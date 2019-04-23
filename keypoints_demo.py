import numpy as np
import cv2
from utils.key_points_recognizer import KeyPointsRecognizer
from utils.logging import log_exec_time
import logging
from threading import Thread
from queue import Queue
import concurrent.futures as fut


MODEL = './models/phase1_wpdc_vdc_v2.pth.tar'
KEYPOINT_COLOR = (0, 255, 0)

recognizer = KeyPointsRecognizer(MODEL)
frameQueue = Queue()


def recognize_frame(frame):
    pts68 = log_exec_time()(recognizer.get_image_keypoints)(frame)
    if (pts68 is not None):
        draw_keypts(frame, pts68, KEYPOINT_COLOR)


#     frameQueue.put(frame)

def draw_keypts(frame, pts, color):
    def plot_close(i1, i2): cv2.line(
        frame, (pts[0, i1], pts[1, i1]), (pts[0, i2], pts[1, i2]), color=color)
    nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]
    #nums = [ 27,31]
    plot_close(41, 36)
    plot_close(47, 42)
    plot_close(59, 48)
    plot_close(67, 60)
    for i in range(0, pts.shape[1]):
        cv2.circle(frame, (pts[0, i], pts[1, i]), 2, color=color)

    for ind in range(len(nums) - 1):
        l, r = nums[ind], nums[ind + 1]
        for ii in range(l, r-1):
            cv2.line(frame, (pts[0, ii], pts[1, ii]),
                     (pts[0, ii+1], pts[1, ii+1]), color=color)


def webcam_live_capturing(device_num=0):
    cap = cv2.VideoCapture(device_num)
    print(cap)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        recognize_frame(frame)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def save_frames_wth_pts_from_vid(video_path):
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(cap)
    i = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        print('Frame {} is being processed'.format(i))
        # Our operations on the frame come here
        recognize_frame(frame)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        cv2.imwrite(video_path.split('/')[2].split('.')[0]+'_keypoints_f{}.png'.format(i), frame)
        i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# logging.basicConfig(level=logging.DEBUG)
# webcam_live_capturing()

save_frames_wth_pts_from_vid('./videos/Video_Sample_4.mov')
