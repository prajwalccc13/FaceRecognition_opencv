import numpy as np
import cv2

from utils import CFEVideoConf, image_resize

cap = cv2.VideoCapture(0)

save_path           = 'saved-media/glasses_and_stash.avi'
frames_per_seconds  = 24
config              = CFEVideoConf(cap, filepath=save_path, res='720p')
out                 = cv2.VideoWriter(save_path, config.video_type, frames_per_seconds, config.dims)
face_cascade        = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
eyes_cascade        = cv2.CascadeClassifier('cascades/third-party/parojosG.xml')
nose_cascade        = cv2.CascadeClassifier('cascades/third-party/Nose18x15.xml')
glasses             = cv2.imread("images/fun/glasses.png", -1)
mustache            = cv2.imread('images/fun/mustache.png',-1)
