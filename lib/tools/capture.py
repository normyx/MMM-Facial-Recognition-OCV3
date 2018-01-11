"""MMM-Facial-Recognition - MagicMirror Module
Positive Image Capture Script
The MIT License (MIT)

Copyright (c) 2016 Paul-Vincent Roll (MIT License)
Based on work by Tony DiCola (Copyright 2013) (MIT License)

Run this script to capture positive images for training the face recognizer.
"""
from __future__ import division
# need to run `pip install future` for builtins (python 2 & 3 compatibility)
from   builtins import input

import fnmatch
import glob
import os
import sys
import re

import cv2
sys.path.append((os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))+ '/common/'))

import config
from face import FaceDetection

face = FaceDetection(config.HAAR_SCALE_FACTOR,
                     config.HAAR_MIN_NEIGHBORS_FACE,
                     config.HAAR_MIN_SIZE_FACE,
                     config.HAAR_FACES)

def is_letter_input(letter):
    input_char = input()
    return input_char.lower()


def walk_files(directory, match='*'):
    """Generator function to iterate through all files in a directory
    recursively which match the given filename match parameter.
    """
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, match):
            yield os.path.join(root, filename)


def capture(preview):
    camera = config.get_camera(preview)
    # Create the directory for positive training images if it doesn't exist.
    if not os.path.exists(config.TRAINING_DIR + CAPTURE_DIR):
        os.makedirs(config.TRAINING_DIR + CAPTURE_DIR)
    # Find the largest ID of existing positive images.
    # Start new images after this ID value.
    files = sorted(glob.glob(os.path.join(config.TRAINING_DIR + CAPTURE_DIR, '[0-9][0-9][0-9].pgm')))
    count = 0
    if len(files) > 0:
        # Grab the count from the last filename.
        count = int(files[-1][-7:-4]) + 1
    print('Capturing positive training images.')
    print('Press enter to capture an image.')
    print('Press Ctrl-C to quit.')
    while True:
        try:
            input()
            print('Capturing image...')
            image = camera.read()
            # Convert image to grayscale.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Get coordinates of single face in captured image.
            result = face.detect_single(image)
            if result is None:
                print('Could not detect single face!'
                      + ' Check the image in capture.pgm'
                      + ' to see what was captured and try'
                      + ' again with only one face visible.')
                continue
            x, y, w, h = result
            # Crop image as close as possible to desired face aspect ratio.
            # Might be smaller if face is near edge of image.
            crop = face.crop(image, x, y, w, h,int((config.FACE_HEIGHT / float(config.FACE_WIDTH)) * w))
            # Save image to file.
            filename = os.path.join(config.TRAINING_DIR,
                                    CAPTURE_DIR,
                                    '%03d.pgm' % count)
            cv2.imwrite(filename, crop)
            print('Found face and wrote training image', filename)
            count += 1
        except KeyboardInterrupt:
            camera.stop()
            break


def convert():
    # Create the directory for positive training images if it doesn't exist.
    if not os.path.exists(config.TRAINING_DIR + CAPTURE_DIR):
        os.makedirs(config.TRAINING_DIR + CAPTURE_DIR)
    # Find the largest ID of existing positive images.
    # Start new images after this ID value.
    files = sorted(glob.glob(os.path.join(config.TRAINING_DIR,
                                          CAPTURE_DIR, '[0-9][0-9][0-9].pgm')))
    count = 0
    if len(files) > 0:
        # Grab the count from the last filename.
        count = int(files[-1][-7:-4]) + 1
    for filename in walk_files(RAW_DIR, '*'):
        if not re.match('.+\.(jpg|jpeg)$', filename, re.IGNORECASE):
            print("file {0} does not have the correct file extention."
                  .format(filename))
            continue
        print("processing {0}".format(filename))
        image = cv2.imread(filename)
        height, width, channels = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # TODO: check for multiple faces and warn
        # Get coordinates of single face in captured image.
        result = face.detect_single(image)
        if result is None:
            if (height + width > 800):
                # it's a big image resize it and try again
                mult = 0.5
                print('Resizing from ({0},{1}) -> ({2},{3})'
                      .format(height, width,
                              int(mult*height), int(mult*width)))
                image2 = cv2.resize(image, None, fx=mult, fy=mult)
                result = face.detect_single(image2)
                if result is None:
                    mult = 0.25
                    print('Resizing from ({0},{1}) -> ({2},{3})'
                          .format(height, width,
                                  int(mult*height), int(mult*width)))
                    image2 = cv2.resize(image, None, fx=mult, fy=mult)
                    result = face.detect_single(image2)
                if result is not None:
                    print('It worked, found a face in resized image!')
                    image = image2
            if result is None:
                print('No face found')
                continue
        x, y, w, h = result
        # Crop image as close as possible to desired face aspect ratio.
        # Might be smaller if face is near edge of image.
        crop = face.crop(image, x, y, w, h,int((config.FACE_HEIGHT / float(config.FACE_WIDTH)) * w))
        # Save image to file.
        filename = os.path.join(config.TRAINING_DIR,
                                CAPTURE_DIR, '%03d.pgm' % count)
        cv2.imwrite(filename, crop)
        print('Found face and wrote training image', filename)
        count += 1
