"""
**
* Author            : A. Peter Hudson
* Filename          : TestImages.py
* Functions         : showImage(frame, faceCoordinates, index)
* Global Variables  : Screen_name, FACE_SHAPE, model, emotions, colour, Path, data_path, data_dir_list
**
"""

import os,cv2
import numpy as np
import FaceRecognitionUtility as fru
import CNN_model as objModel

Screen_name = 'Testing'
FACE_SHAPE = (48, 48)
model = objModel.buildModel('model.h5')
emotions = ['Happy','Fear','Anger','Neutral','Sad','Surprise']
colour = [(255,0,0),(255,255,0),(0,0,255),(0,255,0),(0,255,255),(255,0,255)]

"""
* Function Name : showImage(img, faceCoordinates, index)
* Input         : img - input image
				  faceCoordinates -Coordinates of the bounding box bounding the face
				  index - variable to determine the colour of the bounding box. Different colour for different emotions
* Output        : Redraws the bounding box for every new img and names the emotion. 
"""

def showImage(img, faceCoordinates, index):
    if faceCoordinates is not None:
        if index is not None:
            cv2.putText (img, emotions[index], (faceCoordinates[0] - 45, faceCoordinates[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 2, colour[index],
                         3, cv2.LINE_AA)
            fru.drawBoundingBox (img, faceCoordinates, colour[index])
        else:
            fru.drawBoundingBox (img, faceCoordinates)
    #cv2.imshow('Test', img) #Uncomment if required to check the output
    #cv2.waitKey(0)

PATH = os.getcwd()
# Define data path
data_path = PATH + '/TestData'
data_dir_list = os.listdir(data_path)
print (data_dir_list)

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print img_list
	print ('Testing on the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		input_img = cv2.imread(data_path+'/'+ dataset+'/'+img)
		img = cv2.imread(data_path+'/'+ dataset+'/'+img)
		faceCoordinates = fru.getFaceCoordinates(input_img)
		if faceCoordinates is not None:
			face_img = fru.preprocessImage(input_img, faceCoordinates, face_shape=FACE_SHAPE)
			fru.drawBoundingBox(face_img, faceCoordinates)
			input_img = np.expand_dims(face_img, axis=0)
			input_img = np.expand_dims(input_img, axis=0)
			result = model.predict(input_img)[0]
			#print(result)
			index = np.argmax(result)
			print (emotions[index], 'prob:', max(result))
			showImage (img, faceCoordinates, index)
