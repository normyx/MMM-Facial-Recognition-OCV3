"""Raspberry Pi Face Recognition Treasure Box
Face Detection Helper Functions
Copyright 2013 Tony DiCola

Functions to help with the detection and cropping of faces.
"""

import cv2
import sys


class FaceDetection:
    'Face Detection Class'
    def __init__(self, 
                 haar_scale_factor,
                 haar_min_neighbors_face, 
                 haar_min_size_face,
                 haar_faces_file=None,
                 haar_min_neighbors_eyes=None, 
                 haar_min_size_eyes=None, 
                 haar_eyes_file=None) :
        self.haar_scale_factor = haar_scale_factor
        self.haar_min_neighbors_face = haar_min_neighbors_face
        self.haar_min_size_face = haar_min_size_face
        self.haar_faces_file = haar_min_size_face
        self.haar_min_neighbors_eyes = haar_min_neighbors_eyes
        self.haar_min_size_eyes = haar_min_size_eyes
        self.haar_eyes_file = haar_eyes_file
        if haar_faces_file is not None : self.haar_faces = cv2.CascadeClassifier(haar_faces_file)
        if haar_eyes_file is not None : self.haar_eyes = cv2.CascadeClassifier(haar_eyes_file)

    def detect_single(self, image):
        """Return bounds (x, y, width, height) of detected face in grayscale image.
        If no face or more than one face are detected, None is returned.
        """
        faces = self.haar_faces.detectMultiScale(image, 
                                            scaleFactor=self.haar_scale_factor, 
                                            minNeighbors=self.haar_min_neighbors_face, 
                                            minSize=self.haar_min_size_face, 
                                            flags=cv2.CASCADE_SCALE_IMAGE)
        if len(faces) != 1:
            return None
        return faces[0]

    def detect_faces(self, image):
        """Return bounds (x, y, width, height) of detected face in grayscale image.
        return all faces found in the image
        """
        faces = self.haar_faces.detectMultiScale(image, 
                                            scaleFactor=self.haar_scale_factor, 
                                            minNeighbors=self.haar_min_neighbors_face, 
                                            minSize=self.haar_min_size_face, 
                                            flags=cv2.CASCADE_SCALE_IMAGE)
        return faces

    def detect_eyes(self, image,haar_scale_factor,haar_min_neighbors_eyes, haar_min_size_eyes, haar_eyes_file):
        eyes = self.haar_eyes.detectMultiScale(image, 
                                          scaleFactor=self.haar_scale_factor,
                                          minNeighbors=self.haar_min_neighbors_eyes, 
                                          minSize=self.haar_min_size_eyes, 
                                          flags=cv2.CASCADE_SCALE_IMAGE)
        return eyes


    def eyes_to_face(self, eyes):
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


    def crop(self, image, x, y, w, h, crop_height):
        """Crop box defined by x, y (upper left corner) and w, h (width and height)
        to an image with the same aspect ratio as the face training data.  Might
        return a smaller crop if the box is near the edge of the image.
        """
        midy = y + h / 2
        y1 = int(max(0, midy - crop_height / 2))
        y2 = int(min(image.shape[0] - 1, midy + crop_height / 2))
        return image[y1:y2, x:x + w]


    def resize(self, image, face_width, face_height):
        """Resize a face image to the proper size for training and detection.
        """
        return cv2.resize(image, (face_width, face_height), interpolation=cv2.INTER_LANCZOS4)
