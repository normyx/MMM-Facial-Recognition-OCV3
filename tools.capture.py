#!/usr/bin/env python
# coding: utf8
"""MMM-Facial-Recognition - MagicMirror Module
Face Recognition image capture script
The MIT License (MIT)

Copyright (c) 2016 Paul-Vincent Roll (MIT License)
Based on work by Tony DiCola (Copyright 2013) (MIT License)

Run this script to capture images for the training script.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
