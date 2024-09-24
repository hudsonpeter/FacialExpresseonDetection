"""
**
* Author            : A. Peter Hudson
* Filename          : FaceRecognitionUtility.py
* Functions         : getFaceCoordinates(image), 
					  drawBoundingBox(image, faceCoordinates, Rectangle_Colour)
					  cropFace(image, faceCoordinates), def preprocessImage(image, faceCoordinates, face_shape)
* Global Variables  : CASCADE_PATH
**
"""

import cv2
import numpy as np

CASCADE_PATH = "haarcascade_frontalface_default.xml"

"""
* Function Name : getFaceCoordinates(image)
* Input         : image - input image
* Output        : returns face coordinates from the image
"""

def getFaceCoordinates(image):
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    rectangles = cascade.detectMultiScale(img_gray,scaleFactor=1.1,minNeighbors=3,minSize=(48, 48))

    # For now, algorithm deal with the case that detects one face.
    if(len(rectangles) != 1) :
        return None
    
    face = rectangles[0]
    #print face
    bounding_box = [face[0], face[1], face[0] + face[2], face[1] + face[3]]
    #print bounding_box
    return bounding_box

"""
* Function Name : drawBoundingBox(image, faceCoordinates, Rectangle_Colour = (0, 255, 0))
* Input         : image - input image
				  faceCoordinates - Coordinates of the face in the image
				  Rectangle_Colour - Colour of the bounding box
* Output        : Draws rectangle around the face in passed image using passed faceCoordinates.
"""

def drawBoundingBox(image, faceCoordinates, Rectangle_Colour = (0, 255, 0)):
    cv2.rectangle(np.asarray(image), (faceCoordinates[0], faceCoordinates[1]),(faceCoordinates[2], faceCoordinates[3]), Rectangle_Colour, thickness=2)

"""
* Function Name : cropFace(image, faceCoordinates)
* Input         : image - input image
				  faceCoordinates - Coordinates of the face in the image
* Output        : Crops the image to return only face
"""

def cropFace(image, faceCoordinates):
    return image[faceCoordinates[1]:faceCoordinates[3], faceCoordinates[0]:faceCoordinates[2]]

"""
* Function Name : preprocessImage(image, faceCoordinates, face_shape)
* Input         : image - input image
				  faceCoordinates - Coordinates of the face in the image
				  face_shape - dimension of the image
* Output        : This function will crop face from the original frame.
"""

def preprocessImage(image, faceCoordinates, face_shape=(48, 48)):
    face = cropFace(image, faceCoordinates)
    face_scaled = cv2.resize(face, face_shape)
    face_gray = cv2.cvtColor(face_scaled, cv2.COLOR_BGR2GRAY)
    return face_gray
