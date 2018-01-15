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
import platform
import cv2

HAAR_FACES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))+ '/resources/haarcascade_frontalface.xml'
HAAR_EYES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))+ '/resources/haarcascade_eye.xml'
HAAR_SCALE_FACTOR = 1.05
HAAR_MIN_NEIGHBORS_FACE = 4     # 4 or 3 trainer/tester used different values.
HAAR_MIN_NEIGHBORS_EYES = 2
HAAR_MIN_SIZE_FACE = (30, 30)
HAAR_MIN_SIZE_EYES = (20, 20)



(CV_MAJOR_VER, CV_MINOR_VER, mv1) = cv2.__version__.split(".")

# Size (in pixels) to resize images for training and prediction.
# Don't change this unless you also change the size of the training images.
FACE_WIDTH = 92
FACE_HEIGHT = 112

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


