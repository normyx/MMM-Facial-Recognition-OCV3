#!/usr/bin/env python
# coding: utf8
"""MMM-Facial-Recognition - MagicMirror Module
Face Recognition Training Script
The MIT License (MIT)

Copyright (c) 2016 Paul-Vincent Roll (MIT License)
Based on work by Tony DiCola (Copyright 2013) (MIT License)

Run this script to train the face recognition system with training images from multiple people.
The face recognition model is based on the eigen faces algorithm implemented in OpenCV.
You can find more details on the algorithm and face recognition here:
http://docs.opencv.org/modules/contrib/doc/facerec/facerec_tutorial.html

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import fnmatch
import os

# to install builtins run `pip install future` 
from builtins import input

import cv2
import numpy as np

import lib.tools.config as config
import lib.common.face as face

print("Which algorithm do you want to use?")
print("[1] LBPHF (recommended)")
print("[2] Fisherfaces")
print("[3] Eigenfaces")

algorithm_choice = int(input("--> "))
print('')


def walk_files(directory, match='*'):
    """Generator function to iterate through all files in a directory recursively
    which match the given filename match parameter.
    """
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, match):
            yield os.path.join(root, filename)


def prepare_image(filename):
    """Read an image as grayscale and resize it to the appropriate size for
    training the face recognition model.
    """
    return face.resize(cv2.imread(filename, cv2.IMREAD_GRAYSCALE),config.FACE_WIDTH, config.FACE_HEIGHT)


def normalize(X, low, high, dtype=None):
    """Normalizes a given array in X to a value between low and high.
    Adapted from python OpenCV face recognition example at:
    https://github.com/Itseez/opencv/blob/2.4/samples/python2/facerec_demo.py
    """
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # normalize to [0...1].
    X = X - float(minX)
    X = X / float((maxX - minX))
    # scale to [low...high].
    X = X * (high - low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)


if __name__ == '__main__':
    print("Reading training images...")
    print('-' * 20)
    faces = []
    labels = []
    IMAGE_DIRS_WITH_LABEL = [[0, "negative"]]
    IMAGE_DIRS = os.listdir(config.TRAINING_DIR)
    IMAGE_DIRS = [x for x in IMAGE_DIRS if not x.startswith('.') and not x.startswith('negative')]
    pos_count = 0

    for i in range(len(IMAGE_DIRS)):
        print("Assign label " + str(i + 1) + " to " + IMAGE_DIRS[i])
        IMAGE_DIRS_WITH_LABEL.append([i + 1, IMAGE_DIRS[i]])
    print('-' * 20)
    print('')

    # Für jedes Label/Namen Paar:
    # for every label/name pair:
    for j in range(0, len(IMAGE_DIRS_WITH_LABEL)):
        # Label zu den Labels hinzufügen / Bilder zu den Gesichtern
        for filename in walk_files(config.TRAINING_DIR + str(IMAGE_DIRS_WITH_LABEL[j][1]), '*.pgm'):
            faces.append(prepare_image(filename))
            labels.append(IMAGE_DIRS_WITH_LABEL[j][0])
            if IMAGE_DIRS_WITH_LABEL[j][0] != 0:
                pos_count += 1

    # Print statistic on how many pictures per person we have collected
    print('Read', pos_count, 'positive images and', labels.count(0), 'negative images.')
    print('')
    for j in range(1, max(labels) + 1):
        print(str(labels.count(j)) + " images from subject " + IMAGE_DIRS[j - 1])

    # Train model
    print('-' * 20)
    print('')
    print('Training model type {0} with threshold {1}'
          .format(config.RECOGNITION_ALGORITHM, config.POSITIVE_THRESHOLD))

    model = config.model(config.RECOGNITION_ALGORITHM, config.POSITIVE_THRESHOLD)

    model.train(np.asarray(faces), np.asarray(labels))

    # Save model results
    model.write(config.TRAINING_FILE)
    print('Training data saved to', config.TRAINING_FILE)
    print('')
    print("Please add or update (if you added new people not just new images) " + str(IMAGE_DIRS) + " inside config.js (mirror module) or config.py (model tester). You can change the names to whatever you want, just keep the same order and you'll be fine.")
    print("Please add " + str(algorithm_choice) + " as your choosen algorithm inside config.js (mirror module) or config.py (model tester).")
