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
```

Important Notice
OpenCV has removed the cv2.face module from version 4.4.0 onwards. (maybe i'll fix it if i feel like it lol) 

Instead of using the cv2.face module, OpenCV now recommends using the face_recognition library, which is a third-party library that provides a simple interface for face recognition tasks. This library uses dlib's implementation of face recognition algorithms, which is more accurate and faster than the algorithms included in the cv2.face module.

You can install the face_recognition library using pip:

```bash
pip install face_recognition
```

