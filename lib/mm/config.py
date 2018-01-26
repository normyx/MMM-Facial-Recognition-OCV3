#!/usr/bin/python
# coding: utf8
"""MMM-Facial-Recognition-OCV3 - MagicMirror Module
The MIT License (MIT)

Copyright (c) 2018 Mathieu Goulène (MIT License)
Based on work by Paul-Vincent Roll (Copyright 2016) (MIT License)
"""
import os
import json
import sys
import platform
sys.path.append((os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))+ '/common/'))
from commonconfig import CommonConfig
from face import FaceDetection


class MMConfig (CommonConfig):
    
    CONFIG_DATA = json.loads(sys.argv[1]);
    THRESHOLD_ATTR = 'threshold'
    USE_USB_CAM_ATTR = 'useUSBCam'
    TRAINING_FILE_ATTR = 'trainingFile'
    INTERVAL_ATTR = 'interval'
    LOGOUT_DELAY_ATTR = 'logoutDelay'
    USERS_ATTR = 'users'
    DEFAULT_CLASS_ATTR = 'defaultClass'
    EVERYONE_CLASS_ATTR = 'everyoneClass'
    WELCOME_MESSAGE_ATTR = 'welcomeMessage'
    
    @classmethod
    def toNode(cls, type, message):
        print(json.dumps({type: message}))
        sys.stdout.flush()
    @classmethod
    def getTrainingFile(cls):
        return cls.get(cls.TRAINING_FILE_ATTR)
    @classmethod
    def getInterval(cls):
        return cls.get(cls.INTERVAL_ATTR)
    @classmethod
    def getLogoutDelay(cls):
        return cls.get(cls.LOGOUT_DELAY_ATTR)
    @classmethod
    def getUsers(cls):
        return cls.get(cls.USERS_ATTR)
    @classmethod
    def getDefaultClass(cls):
        return cls.get(cls.DEFAULT_CLASS_ATTR)
    @classmethod
    def getEveryoneClass(cls):
        return cls.get(cls.EVERYONE_CLASS_ATTR)
    @classmethod
    def getWelcomeMessage(cls):
        return cls.get(cls.WELCOME_MESSAGE_ATTR)
    
    @classmethod
    def getUseUSBCam(cls):
        return cls.get(cls.USE_USB_CAM_ATTR)

    @classmethod
    def getThreshold(cls):
        return cls.get(cls.THRESHOLD_ATTR)

    
    @classmethod
    def get(cls,key):
        return cls.CONFIG_DATA[key]
        
    @classmethod
    def getCamera(cls):
        cls.toNode("status", "-" * 20)
        try:
            if cls.get("useUSBCam") == False:
                import picam
                cls.toNode("status", "PiCam loaded...")
                cam = picam.OpenCVCapture()
                cam.start()
                return cam
            else:
                raise Exception
        except Exception:
            import webcam
            cls.toNode("status", "Webcam loaded...")
            return webcam.OpenCVCapture(device_id=0)
        cls.toNode("status", "-" * 20)

    





