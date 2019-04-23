from utils.key_points_recognizer import  KeyPointsRecognizer
from utils.key_points_renderer import KeyPointsRenderer
import os
import cv2

if __name__ == '__main__':
    frames_step = 1
    recognizer = KeyPointsRecognizer(checkpoint_fp='./models/phase1_wpdc_vdc_v2.pth.tar', frames_step=frames_step)
    key_points = recognizer.get_key_points_from_video(video_fp='video_sample_03.MOV')
    renderer = KeyPointsRenderer(frames_step=1)
    print(key_points.shape)
    renderer.frames = key_points
    renderer.animate_key_points_from_frames(key_points)
    
    
