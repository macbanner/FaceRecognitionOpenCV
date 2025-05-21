# FaceRecognitionOpenCV

A simple face recognition project using OpenCV and Python.

## Description

This project demonstrates the use of OpenCV for face detection and recognition using classic algorithms such as Eigenfaces, Fisherfaces, and LBPH (Local Binary Patterns Histograms). It provides a simple CLI interface for training and recognizing faces from a webcam or static images.

## Features

- Train a face recognition model using images
- Recognize faces in real-time using webcam
- Support for multiple face recognition algorithms (LBPH, Eigenfaces, Fisherfaces)
- Save/load models for later use

## Installation

1. Clone the repository:

```bash
git clone https://github.com/macbanner/FaceRecognitionOpenCV.git
cd FaceRecognitionOpenCV 

Important Notice
OpenCV has removed the cv2.face module from version 4.4.0 onwards. This module was used for face recognition and included algorithms such as Eigenfaces, Fisherfaces, and LBPH (Local Binary Patterns Histograms).

Instead of using the cv2.face module, OpenCV now recommends using the face_recognition library, which is a third-party library that provides a simple interface for face recognition tasks. This library uses dlib's implementation of face recognition algorithms, which is more accurate and faster than the algorithms included in the cv2.face module.

You can install the face_recognition library using pip:


pip install face_recognition
⚠️ If you are facing issues due to OpenCV version mismatch, consider downgrading OpenCV or refactoring the project using face_recognition.

