#!/usr/bin/env python
# coding: utf8
"""MMM-Facial-Recognition-OCV3 - MagicMirror Module
The MIT License (MIT)

Copyright (c) 2018 Mathieu Goulène (MIT License)
Based on work by Paul-Vincent Roll (Copyright 2016) (MIT License)
"""
import lib.tools.capture as capture
import lib.tools.config as config

# to install builtins run `pip install future` 
from builtins import input

# set preview to False to disable picamera preview
preview = True

print("What do you want to do?")
print("[1] Capture training images from webcam")
print("[2] Convert '*.jpg' pictures from other cameras to training images")
choice = int(input("--> "))
print("")
print("Enter the name of the person you want to capture or convert images for.")
capture.CAPTURE_DIR = str(input("--> "))
print("Images will be placed in " + config.TRAINING_DIR + capture.CAPTURE_DIR)

if choice == 1:
    print("")
    print('-' * 20)
    print("Starting process...")
    print("")
    capture.capture(preview)
else:
    print("")
    print("Please enter path to images or drag and drop folder into terminal")
    capture.RAW_DIR = str(input("--> "))
    print("")
    print('-' * 20)
    print("Starting process...")
    print("")
    capture.convert()
