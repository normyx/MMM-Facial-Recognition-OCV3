"""Raspberry Pi Face Recognition Treasure Box
Face Detection Helper Functions
Copyright 2013 Tony DiCola

Functions to help with the detection and cropping of faces.
"""

import cv2
import sys
import json





def detect_single(image, haar_scale_factor,haar_min_neighbors, haar_min_size, haar_faces_file):
    """Return bounds (x, y, width, height) of detected face in grayscale image.
    If no face or more than one face are detected, None is returned.
    """
    haar_faces = cv2.CascadeClassifier(haar_faces_file)
    faces = haar_faces.detectMultiScale(image, scaleFactor=haar_scale_factor, minNeighbors=haar_min_neighbors, minSize=haar_min_size, flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) != 1:
        return None
    return faces[0]

def detect_faces(image, haar_scale_factor,haar_min_neighbors, haar_min_size, haar_faces_file):
    """Return bounds (x, y, width, height) of detected face in grayscale image.
    return all faces found in the image
    """
    haar_faces = cv2.CascadeClassifier(haar_faces_file)
    faces = haar_faces.detectMultiScale(image,scaleFactor=haar_scale_factor, minNeighbors=haar_min_neighbors, minSize=haar_min_size, flags=cv2.CASCADE_SCALE_IMAGE)
    return faces

def detect_eyes(image,haar_scale_factor,haar_min_neighbors_eyes, haar_min_size_eyes, haar_eyes_file):
    haar_eyes = cv2.CascadeClassifier(haar_eyes_file)
    eyes = haar_eyes.detectMultiScale(image, scaleFactor=haar_scale_factor,minNeighbors=haar_min_neighbors_eyes, minSize=haar_min_size_eyes, flags=cv2.CASCADE_SCALE_IMAGE)
    return eyes


def eyes_to_face(eyes):
    """Return bounds (x, y, width, height) of estimated face location based
    on the location of a pair of eyes.
    TODO: Sort through multiple eyes (> 2) to find pairs and detect multiple
    faces.
    """
    if (len(eyes) != 2):
        print("Don't know what to do with {0} eye(s).".format(len(eyes)))
        for eye in eyes:
            print('{0:4d} {1:4d} {2:3d} {3:3d}'
                  .format(eye[0], eye[1], eye[2], eye[3]))
        return None
    x0, y0, w0, h0 = eyes[0]
    x1, y1, w1, h1 = eyes[1]
    # compute centered coordinates for the eyes and face
    cx0 = x0 + int(0.5*w0)
    cx1 = x1 + int(0.5*w1)
    cy0 = y0 + int(0.5*h0)
    cy1 = y1 + int(0.5*h1)
    left_cx = min(cx0, cx1)
    right_cx = max(cx0, cx1)
    x_face_center = int((left_cx + right_cx)/2)
    y_face_center = int((cy0 + cy1)/2)
    eye_width = right_cx - left_cx
    # eye_width is about 2/5 the total face width
    # and 2/6 the total height
    w = int(5 * eye_width / 2)
    h = int(3 * eye_width)
    x = max(0, x_face_center - int(1.25 * eye_width))
    y = max(0, y_face_center - int(1.5 * eye_width))
    return [[x, y, w, h]]


def crop(image, x, y, w, h, crop_height):
    """Crop box defined by x, y (upper left corner) and w, h (width and height)
    to an image with the same aspect ratio as the face training data.  Might
    return a smaller crop if the box is near the edge of the image.
    """
    midy = y + h / 2
    y1 = max(0, midy - crop_height / 2)
    y2 = min(image.shape[0] - 1, midy + crop_height / 2)
    return image[y1:y2, x:x + w]


def resize(image, face_width, face_height):
    """Resize a face image to the proper size for training and detection.
    """
    return cv2.resize(image, (face_width, face_height), interpolation=cv2.INTER_LANCZOS4)
