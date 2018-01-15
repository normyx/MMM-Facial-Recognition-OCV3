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




def user_label(i):
    """ Generate the user lable. Lables are 1 indexed.
    """
    i = i - 1
    if i < 0 or i > len(users):
        return "User" + str(int(i))
    return users[i]
