import csv
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import random
import os.path
import glob, os
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D,Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
###########################################Path to the data directories########################################
source_dir=os.getcwd()   
data= source_dir+"/data/"
driving_log = data+"driving_log.csv"
###########################################Driving Log Reading functions########################################
def read_csv(driving_log):
    file_paths, measurments = [], []
    with open(driving_log) as input:
        reader = csv.reader(input)
        next(reader, None)
        offset = 0.28
        for center_img, left_img, right_img, measurment, _, _, _ in reader:
            measurment = float(measurment)
            file_paths.append([(data+center_img.strip()), (data+left_img.strip()), (data+right_img.strip())])
            measurments.append([measurment, measurment+offset, measurment-offset])
    return file_paths, measurments  
#################################################Image processing functions########################################
def resize_image(img):
    img=cv2.resize(img, (320,160), interpolation=cv2.INTER_AREA)
    return img
def crop_image(img):
    img=img[40:-20,:]
    return img
def randomize_contrast(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    brightness = 0.20 + np.random.uniform()
    img[:,:,2] = img[:,:,2] * brightness
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img
#################################################Generator########################################################## 
def generator(X_train, y_train, batch_size=64):
    images = np.zeros((batch_size, 160,320,3), dtype=np.float32)
    measurments = np.zeros((batch_size,), dtype=np.float32)
    while 1:
            bias_count = 0
            for i in range(batch_size):
                sample_i = random.randrange(len(X_train))
                img_i = random.randrange(len(X_train[0]))
                img = cv2.imread(str(X_train[sample_i][img_i]))  
                measurment = y_train[sample_i][img_i]
                #############################Reducing Bias############################
                if abs(measurment) < 0.1:
                    bias_count += 1
                if bias_count > (batch_size/2):
                    while abs(y_train[sample_i][img_i]) < 0.1:
                        sample_i = random.randrange(len(X_train))
               #############################Image Processing and Random Flipping############################         
                img = cv2.imread(str(X_train[sample_i][img_i]))  
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                randomize_contrast_img = randomize_contrast(img)
                crop_img = crop_image(randomize_contrast_img)
                img = resize_image(crop_img)
                img = np.array(img, dtype=np.float32)
                if (random.randint(0,1)==1):
                    images[i] = img
                    measurments[i] = measurment
                else:
                    images[i] =  cv2.flip(img, 1)
                    measurments[i] = -measurment
            yield images, measurments
#################################################Data Set Preparation##########################################################               
X_train, y_train  = read_csv(driving_log)
X_train, y_train = shuffle(X_train, y_train, random_state=14)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.1, random_state=14)
############################################Nueral Network - NVidia##############################################################
model= Sequential()
model.add( Cropping2D( cropping=( (50,20), (0,0) ), input_shape=(160,320,3)))
model.add ( Lambda( lambda x: x/255. - 0.5 ) )
model.add( Convolution2D( 24, 5, 5, subsample=(2,2), activation = 'relu' ) )
model.add( Convolution2D( 36, 5, 5, subsample=(2,2), activation = 'relu' ) )
model.add( Convolution2D( 48, 5, 5, subsample=(2,2), activation = 'relu' ) )
model.add( Convolution2D( 64, 3, 3, subsample=(1,1), activation = 'relu' ) )
model.add( Convolution2D( 64, 3, 3, subsample=(1,1), activation = 'relu' ) )
model.add( Flatten() )
model.add(Dropout(0.5)) 
model.add( Dense( 100 ) )
model.add(Dropout(0.5)) 
model.add( Dense( 50 ) )
model.add( Dense( 10 ) )
model.add( Dense( 1 ) )
model.compile(optimizer='adam', loss='mse')
train_steps = np.ceil( len( X_train )/32 ).astype( np.int32 )
validation_steps = np.ceil( len( X_validation )/32 ).astype( np.int32 )
############################################Running and Saving Model##############################################################
model.fit_generator(generator(X_train, y_train), samples_per_epoch=2, nb_epoch=24, validation_data=generator(X_validation, y_validation), nb_val_samples=2)
model.save('model4.h5')  