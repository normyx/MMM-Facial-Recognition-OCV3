#!/usr/bin/env python
# coding: utf8
"""MMM-Facial-Recognition-OCV3 - MagicMirror Module
The MIT License (MIT)

Copyright (c) 2018 Mathieu Goulène (MIT License)
Based on work by Paul-Vincent Roll (Copyright 2016) (MIT License)
"""
from lib.tools.capture import ToolsCapture
from lib.tools.config import ToolsConfig

# to install builtins run `pip install future` 
from builtins import input


print("What do you want to do?")
print("[1] Capture training images from webcam")
print("[2] Convert '*.jpg' pictures from other cameras to training images")
choice = int(input("--> "))
print("")
print("Enter the name of the person you want to capture or convert images for.")
capName = str(input("--> "))
capture = ToolsCapture(capName)
print("Images will be placed in " + ToolsConfig.TRAINING_DIR + capName)

if choice == 1:
    print("")
    print('-' * 20)
    print("Starting process...")
    print("")
    capture.capture()
else:
    print("")
    print("Please enter path to images or drag and drop folder into terminal")
    rawDir = str(input("--> "))
    print("")
    print('-' * 20)
    print("Starting process...")
    print("")
    capture.convert(rawDir)
