# coding: utf8
"""MMM-Facial-Recognition-OCV3 - MagicMirror Module
The MIT License (MIT)

Copyright (c) 2018 Mathieu Goulène (MIT License)
Based on work by Paul-Vincent Roll (Copyright 2016) (MIT License)
"""
from __future__ import division
# need to run `pip install future` for builtins (python 2 & 3 compatibility)
from   builtins import input

import glob
import os
import sys
import re

import cv2
sys.path.append((os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))+ '/common/'))

from config import ToolsConfig
from face import FaceDetection

class ToolsCapture:
    def __init__(self, capName=None):
        self.face = FaceDetection(ToolsConfig.HAAR_SCALE_FACTOR,
                     ToolsConfig.HAAR_MIN_NEIGHBORS_FACE,
                     ToolsConfig.HAAR_MIN_SIZE_FACE,
                     ToolsConfig.HAAR_FACES)
        self.captureName = capName
                     


    def capture(self):
        toolsConfig = ToolsConfig(self.captureName)
        camera = toolsConfig.getCamera()
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
                result = self.face.detect_single(image)
                if result is None:
                    print('Could not detect single face!'
                          + ' Check the image in capture.pgm'
                          + ' to see what was captured and try'
                          + ' again with only one face visible.')
                    continue
                x, y, w, h = result
                # Crop image as close as possible to desired face aspect ratio.
                # Might be smaller if face is near edge of image.
                crop = self.face.crop(image, x, y, w, h,int((ToolsConfig.FACE_HEIGHT / float(ToolsConfig.FACE_WIDTH)) * w))
                # Save image to file.
                filename, count = toolsConfig.getNewCaptureFile()
                cv2.imwrite(filename, crop)
                print('Found face and wrote training image', filename)
            except KeyboardInterrupt:
                camera.stop()
                break


    def convert(self, rawDir):
        toolsConfig = ToolsConfig(self.captureName)
        filename, count = toolsConfig.getNewCaptureFile()
        
        for filename in ToolsConfig.walkFiles(rawDir, '*'):
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
            result = self.face.detect_single(image)
            if result is None:
                if (height + width > 800):
                    # it's a big image resize it and try again
                    mult = 0.5
                    print('Resizing from ({0},{1}) -> ({2},{3})'
                          .format(height, width,
                                  int(mult*height), int(mult*width)))
                    image2 = cv2.resize(image, None, fx=mult, fy=mult)
                    result = self.face.detect_single(image2)
                    if result is None:
                        mult = 0.25
                        print('Resizing from ({0},{1}) -> ({2},{3})'
                              .format(height, width,
                                      int(mult*height), int(mult*width)))
                        image2 = cv2.resize(image, None, fx=mult, fy=mult)
                        result = self.face.detect_single(image2)
                    if result is not None:
                        print('It worked, found a face in resized image!')
                        image = image2
                if result is None:
                    print('No face found')
                    continue
            x, y, w, h = result
            # Crop image as close as possible to desired face aspect ratio.
            # Might be smaller if face is near edge of image.
            crop = self.face.crop(image, x, y, w, h,int((ToolsConfig.FACE_HEIGHT / float(ToolsConfig.FACE_WIDTH)) * w))
            # Save image to file.
            toFilename = os.path.join(ToolsConfig.TRAINING_DIR,
                                    self.captureName, '%03d.pgm' % count)
            cv2.imwrite(toFilename, crop)
            print('Found face and wrote training image', toFilename)
            count += 1
