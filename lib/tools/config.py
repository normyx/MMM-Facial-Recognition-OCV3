#!/usr/bin/env python
# coding: utf8
"""MMM-Facial-Recognition-OCV3 - MagicMirror Module
The MIT License (MIT)

Copyright (c) 2018 Mathieu Goulène (MIT License)
Based on work by Paul-Vincent Roll (Copyright 2016) (MIT License)
"""
import inspect
import os
import sys
sys.path.append((os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))+ '/common/'))
import platform
import cv2

(CV_MAJOR_VER, CV_MINOR_VER, mv1) = cv2.__version__.split(".")

_platform = platform.system().lower()
path_to_file = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

POSITIVE_THRESHOLD=80





if ('FACE_USERS' in os.environ):
    u = os.environ['FACE_USERS']
    users = u.split(',')
    print(users)
else:
    # NOTE: Substitute your own user names here. These are just
    # placeholders, and you will get errors if your training.xml file
    # has more than 10 user classes.
    users = ["User1", "User2", "User3", "User4", "User5",
             "User6", "User7", "User8", "User9", "User10"]
    print('Remember to set the name list environment variable FACE_USERS')

# File to save and load face recognizer model.
TRAINING_FILE = 'training.xml'
TRAINING_DIR = './training_data/'


# Size (in pixels) to resize images for training and prediction.
# Don't change this unless you also change the size of the training images.
FACE_WIDTH = 92
FACE_HEIGHT = 112

# Face detection cascade classifier configuration.
# You don't need to modify this unless you know what you're doing.
# See: http://docs.opencv.org/modules/objdetect/doc/cascade_classification.html
#HAAR_FACES = 'lib/haarcascade_frontalface_alt.xml'
#HAAR_FACES = 'lib/haarcascade_frontalface_alt2.xml'
#HAAR_FACES = 'lib/haarcascade_frontalface_default.xml'
HAAR_FACES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))+ '/resources/haarcascade_frontalface.xml'
HAAR_EYES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))+ '/resources/haarcascade_eye.xml'
HAAR_SCALE_FACTOR = 1.05
HAAR_MIN_NEIGHBORS_FACE = 4     # 4 or 3 trainer/tester used different values.
HAAR_MIN_NEIGHBORS_EYES = 2
HAAR_MIN_SIZE_FACE = (30, 30)
HAAR_MIN_SIZE_EYES = (20, 20)


def get_camera(preview=True):
    try:
        import picam
        print("Loading PiCamera")
        capture = picam.OpenCVCapture(preview)
        print("PiCamera loaded")
        capture.start()
        return capture
    except Exception as e:
        print(e)
        import webcam
        return webcam.OpenCVCapture(device_id=0)




def is_cv3():
    if CV_MAJOR_VER == '3':
        return True
    else:
        return False


def model(thresh):
    # set the choosen algorithm
    model = None
    if is_cv3():
        model = cv2.face.LBPHFaceRecognizer_create(threshold=thresh)
    else:
        print("FATAL: OpenCV Major Version must be 3")
        os._exit(1)
    return model


def user_label(i):
    """ Generate the user lable. Lables are 1 indexed.
    """
    i = i - 1
    if i < 0 or i > len(users):
        return "User" + str(int(i))
    return users[i]
