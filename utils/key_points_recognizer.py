import torch
from . import mobilenet_v1
import dlib
import cv2
import scipy.io as sio
import numpy as np
from .inference_utils import get_suffix, calc_roi_box, crop_img, predict_68pts, dump_to_ply, dump_vertex, draw_landmarks, predict_dense, dump_key_points_to_ply
import torchvision.transforms as transforms
from .ddfa_utils import ToTensorGjz, NormalizeGjz, str2bool  
from .key_points_renderer import KeyPointsRenderer
from .logging import log_exec_time
import copy

class KeyPointsRecognizer:
    def __init__(self, checkpoint_fp, frames_step=1):
        self.checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
        self.arch = 'mobilenet_1'
        self.model = self.__load_model()
        self.dlib_landmark_model = './models/shape_predictor_68_face_landmarks.dat'
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_regressor = dlib.shape_predictor(self.dlib_landmark_model)
        self.frames_step = frames_step
        
    def __load_model(self):
        model = getattr(mobilenet_v1, self.arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)
        model_dict = model.state_dict()
        # because the model is trained by multiple gpus, prefix module should be removed
        for k in self.checkpoint.keys():
            model_dict[k.replace('module.', '')] = self.checkpoint[k]
        model.load_state_dict(model_dict)
        model.eval()
        return model

    def get_key_points_from_video(self, video_fp, on_frame_processed=None, save_format='./dsg_{}.ply'):
        cap = cv2.VideoCapture(video_fp)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        key_points = []
        pts_dlib = []
        i = 0
        # 3. forward
        while True:
            _, img_ori = cap.read()
            if i%self.frames_step==0:
                try:
                    print('Frame {} is being processed'.format(i))
                    if np.shape(img_ori) == ():
                        break

                    if on_frame_processed:
                        on_frame_processed(img_ori, i, num_frames, cap)

                    pts68 = self.get_image_keypoints(img_ori)
                    dump_key_points_to_ply(pts68, save_format.format(i))
                    key_points.append(pts68)
                    
                    k = cv2.waitKey(5) & 0xFF
                    if k == 27:
                        break

                    i += 1
                except Exception as e:
                    print(e)
            else:
                print("Frame {} skipped".format(i))
                i+=1
        cv2.destroyAllWindows()
        cap.release()
        return np.array(key_points)

    def get_image_keypoints(self, image):
        if np.shape(image) == ():
            return None
        
        rects = self.face_detector(image, 1)
        if not len(rects):
            raise Exception("No faces on image")

        rect = rects[0]
        # landmark & crop
        pts = self.face_regressor(image, rect).parts()
        pts = np.array([[pt.x, pt.y] for pt in pts]).T 

        roi_box = calc_roi_box(pts)
        img = crop_img(image, roi_box)

        # forward: one step
        img = cv2.resize(img, dsize=(120, 120), interpolation=cv2.INTER_LINEAR)
        transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
        input = transform(img).unsqueeze(0)
        with torch.no_grad():
            param = self.model(input)
            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
        
        key_points = predict_68pts(param, roi_box)

        #set (x,y) coords from dlib to resulting key points

        pts[:,:17] = key_points[0:2,:17]
        pts[:,31:36] = key_points[0:2,31:36]
        pts[:,27:31] = key_points[0:2,27:31]

        key_points[0:2][:] = pts[:]

        return key_points






