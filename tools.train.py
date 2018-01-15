#!/usr/bin/env python
# coding: utf8
"""MMMM-Facial-Recognition-OCV3 - MagicMirror Module
The MIT License (MIT)

Copyright (c) 2018 Mathieu Goulène (MIT License)
Based on work by Paul-Vincent Roll (Copyright 2016) (MIT License)
"""

import fnmatch
import os

# to install builtins run `pip install future` 
from builtins import input

import cv2
import numpy as np

import lib.tools.config as config
from lib.common.face import FaceDetection
import lib.common.commonconfig as commonconfig

face = FaceDetection(commonconfig.HAAR_SCALE_FACTOR,
                     commonconfig.HAAR_MIN_NEIGHBORS_FACE,
                     commonconfig.HAAR_MIN_SIZE_FACE,
                     commonconfig.HAAR_FACES)



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
    return face.resize(cv2.imread(filename, cv2.IMREAD_GRAYSCALE),commonconfig.FACE_WIDTH, commonconfig.FACE_HEIGHT)


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
    print('Training model with threshold {0}'
          .format(config.POSITIVE_THRESHOLD))
    model = commonconfig.model(config.POSITIVE_THRESHOLD)

    model.train(np.asarray(faces), np.asarray(labels))

    # Save model results
    model.write(config.TRAINING_FILE)
    print('Training data saved to', config.TRAINING_FILE)
    print('')
    print("Please add or update (if you added new people not just new images) " + str(IMAGE_DIRS) + " inside config.js (mirror module) or config.py (model tester). You can change the names to whatever you want, just keep the same order and you'll be fine.")
