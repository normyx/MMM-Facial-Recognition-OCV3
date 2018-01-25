#!/usr/bin/python
# coding: utf8
"""MMM-Facial-Recognition-OCV3 - MagicMirror Module
The MIT License (MIT)

Copyright (c) 2018 Mathieu Goulène (MIT License)
Based on work by Paul-Vincent Roll (Copyright 2016) (MIT License)
"""
import inspect
import os
import json
import sys
import platform
sys.path.append((os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))+ '/common/'))
from commonconfig import CommonConfig

class MMConfig (CommonConfig):
    
    CONFIG_DATA = json.loads(sys.argv[1]);
    
    @classmethod
    def toNode(cls, type, message):
        print(json.dumps({type: message}))
        sys.stdout.flush()
    
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

#_platform = platform.uname()[4]
#path_to_file = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))






