#!/usr/bin/env python
# coding: utf8
"""MMMM-Facial-Recognition-OCV3 - MagicMirror Module
The MIT License (MIT)

Copyright (c) 2018 Mathieu Goulène (MIT License)
Based on work by Paul-Vincent Roll (Copyright 2016) (MIT License)
"""

# to install builtins run `pip install future` 
from builtins import input

import cv2
import numpy as np
import fnmatch
import os

from lib.tools.config import ToolsConfig
from lib.common.face import FaceDetection

class ToolsTrain:
    def __init__(self):
        
        self.face = FaceDetection(ToolsConfig.HAAR_SCALE_FACTOR,
                     ToolsConfig.HAAR_MIN_NEIGHBORS_FACE,
                     ToolsConfig.HAAR_MIN_SIZE_FACE,
                     ToolsConfig.HAAR_FACES)
    
    def prepareImage(self, filename):
        """Read an image as grayscale and resize it to the appropriate size for
        training the face recognition model.
        """
        return self.face.resize(cv2.imread(filename, cv2.IMREAD_GRAYSCALE),ToolsConfig.FACE_WIDTH, ToolsConfig.FACE_HEIGHT)

    """@classmethod
    def normalize(cls, X, low, high, dtype=None):
        Normalizes a given array in X to a value between low and high.
        Adapted from python OpenCV face recognition example at:
        https://github.com/Itseez/opencv/blob/2.4/samples/python2/facerec_demo.py
        "
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
        return np.asarray(X, dtype=dtype)"""
    
    def train(self):
        print("Reading training images...")
        print('-' * 20)
        faces = []
        labels = []
        imageDirsWithLabel = [[0, "negative"]]
        imageDirs = os.listdir(ToolsConfig.TRAINING_DIR)
        imageDirs = [x for x in imageDirs if not x.startswith('.') and not x.startswith('negative')]
        pos_count = 0

        for i in range(len(imageDirs)):
            print("Assign label " + str(i + 1) + " to " + imageDirs[i])
            imageDirsWithLabel.append([i + 1, imageDirs[i]])
        print('-' * 20)
        print('')

        # Für jedes Label/Namen Paar:
        # for every label/name pair:
        for j in range(0, len(imageDirsWithLabel)):
            # Label zu den Labels hinzufügen / Bilder zu den Gesichtern
            for filename in ToolsConfig.walkFiles(ToolsConfig.TRAINING_DIR + str(imageDirsWithLabel[j][1]), '*.pgm'):
                faces.append(self.prepareImage(filename))
                labels.append(imageDirsWithLabel[j][0])
                if imageDirsWithLabel[j][0] != 0:
                    pos_count += 1

        # Print statistic on how many pictures per person we have collected
        print('Read '  + str(pos_count) + ' positive images and ' + str(labels.count(0)) + ' negative images.')
        print('')
        for j in range(1, max(labels) + 1):
            print(str(labels.count(j)) + " images from subject " + imageDirs[j - 1])

        # Train model
        print('-' * 20)
        print('')
        print('Training model with threshold {0}'
              .format(ToolsConfig.POSITIVE_THRESHOLD))
        model = ToolsConfig.model()

        model.train(np.asarray(faces), np.asarray(labels))

        # Save model results
        model.write(ToolsConfig.TRAINING_FILE)
        print('Training data saved to', ToolsConfig.TRAINING_FILE)
        print('')
        print("Please add or update (if you added new people not just new images) " + str(imageDirs) + " inside config.js (mirror module) or config.py (model tester). You can change the names to whatever you want, just keep the same order and you'll be fine.")



