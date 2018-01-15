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


def to_node(type, message):
    print(json.dumps({type: message}))
    sys.stdout.flush()

sys.path.append((os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))+ '/common/'))
_platform = platform.uname()[4]
path_to_file = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


CONFIG = json.loads(sys.argv[1]);

def get(key):
    return CONFIG[key]

def get_camera():
    to_node("status", "-" * 20)
    try:
        if get("useUSBCam") == False:
            import picam
            to_node("status", "PiCam loaded...")
            cam = picam.OpenCVCapture()
            cam.start()
            return cam
        else:
            raise Exception
    except Exception:
        import webcam
        to_node("status", "Webcam loaded...")
        return webcam.OpenCVCapture(device_id=0)
    to_node("status", "-" * 20)
