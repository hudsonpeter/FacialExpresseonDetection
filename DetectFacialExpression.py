"""
**
* Author            : A. Peter Hudson
* Filename          : DetectFacialExpression.py
* Functions         : refreshFrame(frame, faceCoordinates, index),
					  showAndDetect(capture),
					  getCamStream(), main()					  
* Global Variables  : Screen_name, FACE_SHAPE,model,emotions,colour
**
"""

import cv2
import numpy as np
import FaceRecognitionUtility as fru
import CNN_model as objModel

Screen_name = 'Real-time facial expression recognition'
FACE_SHAPE = (48, 48)
model = objModel.buildModel('mywt.h5')
emotions = ['Happy','Fear','Anger','Neutral','Sad','Surprise']
colour = [(255,0,0),(255,255,0),(0,0,255),(0,255,0),(0,255,255),(255,0,255)]

"""
* Function Name : refreshFrame(frame, faceCoordinates, index)
* Input         : frame - input image
				  faceCoordinates -Coordinates of the bounding box bounding the face
				  index - variable to determine the colour of the bounding box. Different colour for different emotions
* Output        : Redraws the bounding box for every new frame and names the emotion. 
"""

def refreshFrame(frame, faceCoordinates, index):
    if faceCoordinates is not None:
        if index is not None:
            cv2.putText (frame, emotions[index], (faceCoordinates[0] - 45, faceCoordinates[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 2, colour[index],
                         3, cv2.LINE_AA)
            fru.drawBoundingBox (frame, faceCoordinates, colour[index])
        else:
            fru.drawBoundingBox (frame, faceCoordinates)

    cv2.imshow(Screen_name, frame)

"""
* Function Name : showAndDetect(capture)
* Input         : capture - input image
* Output        : Detects and predicts emotions from real time captured data
"""

def showAndDetect(capture):
    while (True):
        flag, frame = capture.read()
        faceCoordinates = fru.getFaceCoordinates(frame)
        refreshFrame(frame, faceCoordinates, None)
        
        if faceCoordinates is not None:
            face_img = fru.preprocessImage(frame, faceCoordinates, face_shape=FACE_SHAPE)
            cv2.imshow(Screen_name, frame)
            fru.drawBoundingBox(face_img, faceCoordinates)

            input_img = np.expand_dims(face_img, axis=0)
            input_img = np.expand_dims(input_img, axis=0)

            result = model.predict(input_img)[0]
            index = np.argmax(result)
            print (emotions[index], 'prob:', max(result))
            refreshFrame (frame, faceCoordinates, index)

        if cv2.waitKey(10) & 0xFF == 27:
            break

"""
* Function Name : getCamStream()
* Input         : none
* Output        : Returns a frame if the camera has successfully captured one
"""

def getCamStream():
    capture = cv2.VideoCapture(0)
    if not capture:
        print("Failed to capture video streaming ")
        sys.exit(1)
    else:
        print("Successed to capture video streaming")
        
    return capture

"""
* Function Name : main()
* Input         : none
* Output        : The start point of the Facial expression recognition
"""

def main():
    capture = getCamStream()

    if capture:
        cv2.startWindowThread()
        cv2.namedWindow(Screen_name, cv2.WND_PROP_FULLSCREEN)
        #cv2.setWindowProperty(Screen_name, cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN)
    
    showAndDetect(capture)
    capture.release()
    cv2.destroyAllWindows()

"""fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('input_video.avi',fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

out.release()"""
if __name__ == '__main__':
    main()
