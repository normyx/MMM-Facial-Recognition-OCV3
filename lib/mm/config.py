#!/usr/bin/python
# coding: utf8
"""MMM-Facial-Recognition-OCV3 - MagicMirror Module
The MIT License (MIT)

Copyright (c) 2018 Mathieu Goulène (MIT License)
Based on work by Paul-Vincent Roll (Copyright 2016) (MIT License)
"""
import inspect
import os
import json
import sys
import platform


def to_node(type, message):
    print(json.dumps({type: message}))
    sys.stdout.flush()

sys.path.append((os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))+ '/common/'))
_platform = platform.uname()[4]
path_to_file = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# Size (in pixels) to resize images for training and prediction.
# Don't change this unless you also change the size of the training images.
FACE_WIDTH = 92
FACE_HEIGHT = 112

# Face detection cascade classifier configuration.
# You don't need to modify this unless you know what you're doing.
# See: http://docs.opencv.org/modules/objdetect/doc/cascade_classification.html
HAAR_FACES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))+ '/resources/haarcascade_frontalface.xml'
HAAR_SCALE_FACTOR = 1.05
HAAR_MIN_NEIGHBORS = 4
HAAR_MIN_SIZE = (30, 30)

CONFIG = json.loads(sys.argv[1]);

def get(key):
    return CONFIG[key]

def get_camera():
    to_node("status", "-" * 20)
    try:
        if get("useUSBCam") == False:
            import picam
            to_node("status", "PiCam ausgewählt...")
            cam = picam.OpenCVCapture()
            cam.start()
            return cam
        else:
            raise Exception
    except Exception:
        import webcam
        to_node("status", "Webcam ausgewählt...")
        return webcam.OpenCVCapture(device_id=0)
    to_node("status", "-" * 20)
