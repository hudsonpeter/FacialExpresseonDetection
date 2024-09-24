"""
**
* Author            : A. Peter Hudson
* Filename          : TrainDataset.py
* Functions         : None
* Global Variables  : Path, data_path, data_dir_list, num_channel, num_epoch, num_classes
					  img_data, names, labels, Y, x, y, X_train, Y_train, X_test, Y_test
**
"""

# Import libraries
import os,cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras import backend as K
K.set_image_dim_ordering('th')

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,RMSprop,adam

PATH = os.getcwd()
# Define data path
data_path = PATH + '/Dataset'
data_dir_list = os.listdir(data_path)
print (data_dir_list)
num_channel=1
num_epoch=20

# Define the number of classes
num_classes = 6

img_data_list=[]

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loading the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
		input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
		#input_img_resize=cv2.resize(input_img,(128,128))
		#img_data_list.append(input_img_resize)
		img_data_list.append(input_img)

print("Loading Complete...")
#exit()

img_data = np.array(img_data_list)

print (img_data.shape)

if num_channel==1:
	if K.image_dim_ordering()=='th':
		img_data= np.expand_dims(img_data, axis=1) 
		print (img_data.shape)
	else:
		img_data= np.expand_dims(img_data, axis=4) 
		print (img_data.shape)

num_of_samples = img_data.shape[0]

# Assigning Labels
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:3999] = 0
labels[4000:7999] = 1
labels[8000:11999] = 2
labels[12000:15999] = 3
labels[16000:19999] = 4
labels[20000:23999] = 5
	  
names = ['Happy','Fear','Anger','Neutral','Sad','Surprise'] 
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)

# Defining the model
input_shape=img_data[0].shape
					
model = Sequential()

model.add(ZeroPadding2D ((2, 2), input_shape=(1, 48, 48)))
model.add(Conv2D (32, 5, activation='relu'))
model.add(ZeroPadding2D((2,2), input_shape=(1, 48, 48)))
model.add(Conv2D(32, 5, activation='relu'))
model.add(ZeroPadding2D((2,2)))
model.add(Conv2D(32, 5, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add (ZeroPadding2D ((2, 2)))
model.add (Conv2D (64, 5, activation='relu'))
model.add(ZeroPadding2D((2,2)))
model.add(Conv2D(64, 5, activation='relu'))
model.add(ZeroPadding2D((2,2)))
model.add(Conv2D(64, 5, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add (ZeroPadding2D ((2, 2)))
model.add (Conv2D (128, 5, activation='relu'))
model.add(ZeroPadding2D((2,2)))
model.add(Conv2D(128, 5, activation='relu'))
model.add(ZeroPadding2D((2,2)))
model.add(Conv2D(128, 5, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5)) 
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])

# Viewing model_configuration

model.summary()
"""model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape			
model.layers[0].output_shape			
model.layers[0].get_weights()
#np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable"""

# Training
hist = model.fit(X_train, y_train, batch_size=64, epochs=num_epoch, verbose=1, validation_data=(X_test, y_test))
#hist = model.fit(X_train, y_train, batch_size=32, nb_epoch=20,verbose=1, validation_split=0.2)

# Training with callbacks
from keras import callbacks

filename='model_train_new.csv'
csv_log=callbacks.CSVLogger(filename, separator=',', append=False)

early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='min')

filepath="Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"

checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [csv_log,early_stopping,checkpoint]

hist = model.fit(X_train, y_train, batch_size=64, epochs=num_epoch, verbose=1, validation_data=(X_test, y_test),callbacks=callbacks_list)

model.save_weights("model.h5")

print("Finished Training and Building Model")
