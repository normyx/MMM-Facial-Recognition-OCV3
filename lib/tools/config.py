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
from commonconfig import CommonConfig
import platform
import cv2
import fnmatch

import glob



#_platform = platform.system().lower()
#path_to_file = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))




class ToolsConfig (CommonConfig):
    # File to save and load face recognizer model.
    TRAINING_FILE = 'training.xml'
    TRAINING_DIR = './training_data/'
    POSITIVE_THRESHOLD=80
    
    # The name of the person to be captured
    captureName = ''


    USERS = ["User1", "User2", "User3", "User4", "User5", "User6", "User7", "User8", "User9", "User10"]

    if ('FACE_USERS' in os.environ):
        u = os.environ['FACE_USERS']
        USERS = u.split(',')
        print(USERS)
    else:
        # NOTE: Substitute your own user names here. These are just
        # placeholders, and you will get errors if your training.xml file
        # has more than 10 user classes.
        print('Remember to set the name list environment variable FACE_USERS')





    def __init__(self, capName=None):
        self.captureName = capName


    def createCaptureDirIfNotExisting(self):
        capturePath = self.getCapturePath()
        if not os.path.exists(capturePath): os.makedirs(capturePath)

    def getCapturePath(self):
        return ToolsConfig.TRAINING_DIR + self.captureName
        
    def getCapturedFiles(self, pattern):
        return os.path.join(self.getCapturePath(), pattern)
    
    def getNewCaptureFile(self):
        self.createCaptureDirIfNotExisting()
        files = sorted(glob.glob(self.getCapturedFiles('[0-9][0-9][0-9].pgm')))
        count = 0
        if len(files) > 0:
            # Grab the count from the last filename.
            count = int(files[-1][-7:-4]) + 1
        return self.getCapturedFiles('%03d.pgm' % count), count

    @classmethod
    def walkFiles(cls, directory, match='*'):
        """Generator function to iterate through all files in a directory
        recursively which match the given filename match parameter.
        """
        for root, dirs, files in os.walk(directory):
            for filename in fnmatch.filter(files, match):
                yield os.path.join(root, filename)

        
    @classmethod
    def getCamera(cls):
        try:
            import picam
            print("Loading PiCamera")
            capture = picam.OpenCVCapture(True)
            print("PiCamera loaded")
            capture.start()
            return capture
        except Exception as e:
            print(e)
            import webcam
            return webcam.OpenCVCapture(device_id=0)

    @classmethod
    def model(cls):
        return CommonConfig.model(cls.POSITIVE_THRESHOLD)

    @classmethod
    def userLabel(cls,i):
        """ Generate the user lable. Lables are 1 indexed.
        """
        i = i - 1
        if i < 0 or i > len(ToolsConfig.USERS):
            return "User" + str(int(i))
        return ToolsConfig.USERS[i]
