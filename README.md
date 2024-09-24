# FacialExpresseonDetection
************************************************************************************

* Project Title : Facial Expression Recognition using Convolutional Neural Network *

************************************************************************************

* 				Author : A. Peter Hudson	      		   *
***************************************************************

*********************

============================================================

The Project folder Contains the following files and folders:

============================================================

1. CNN_model.py

2. DetectFacialExpression.py

3. FaceRecognitionUtility.py

4. TestImages.py

5. TrainDataset.py

6. haar_cascade_frontalface_default.xml

7. model.h5

8. Dataset - zip file

9. TestData - zip file
10. Weights - Folder

11. DatasetGenerator - Folder

============================================================



*=========================*

* File/Folder Description *

*=========================*


================================================================================================================================


1. CNN_model.py : A python file containing the code of the architecture of the CNN. The file is used as a module.



2. DetectFacialExpression.py : Python program for the real-time facial expression detection.
   
   Command to run/execute the program -  "python DetectFacialExpression.py" 



3. FaceRecognitionUtility.py : Python program contains functions used for pre-processing of image. The file is used as a module.



4. TestImages.py: Python program to test the CNN model on images.

   Command to run/execute the program -  "python TestImages.py"    
   
   Note: Make sure the Directory path is correct in the program.



5. TrainDataset.py: Python program to train datasets to update weights for the model.
   
   Command to run/execute the program -  "python TrainDataset.py"    
   
   Note: Make sure the Directory path is correct in the program.



6. haar_cascade_frontalface_default.xml: The xml file used in face detection in an image. The file is used as a module.



7. model.h5: This file contains the precalculated or the newly created weights which is the output of the "TrainDataset.py".



8. Dataset.zip :   
   This zip file contains six directories of 6 different emotions namely - Anger, Happy, Sad, Surprise, Fear, Neutral.
   
   Each directory contains images under that category.
   
   Used for training the CNN.



9. TestData.zip :
   This directory contains six directories of 6 different emotions namely - Anger, Happy, Sad, Surprise, Fear, Neutral.
   
   Each directory contains images under that category.
   
   Used for testing the CNN.



10. Weights - Folder
    
    Contains weights obtained as the size of the dataset was increased.



11. DatasetGenerator - Folder
    
    Contains two files.
    
    1. fer2013.csv : This file contains pixel values of the images used for training.
	
    2. reconstruct.py : Python program to gnerate the images from fer2013.csv file.



===================================================================================================================================
