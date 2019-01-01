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

#################################################Data Set Preparation##########################################################               
X_train, y_train  = read_csv(driving_log)
img = cv2.imread(str(X_train[218][1]))  
randomize_contrast_img = randomize_contrast(img)
crop_img = crop_image(randomize_contrast_img)
resize_image = resize_image(crop_img)

img2 = cv2.imread(str(X_train[218][2]))  
randomize_contrast_img2 = randomize_contrast(img2)
crop_img2 = crop_image(randomize_contrast_img2)
resize_image2 = cv2.resize(crop_img2, (320,160), interpolation=cv2.INTER_AREA)

img3 = cv2.imread(str(X_train[218][0]))  
randomize_contrast_img3 = randomize_contrast(img3)
crop_img3 = crop_image(randomize_contrast_img3)
resize_image3 = cv2.resize(crop_img3, (320,160), interpolation=cv2.INTER_AREA)

img_flipped =  cv2.flip(img, 1)
img2_flipped =  cv2.flip(img2, 1)
img3_flipped =  cv2.flip(img3, 1)
'''
fig = plt.figure()
plt.title('left_randomize_contrast_img')
plt.imshow(randomize_contrast_img)
fig.savefig('left_randomize_contrast_img.png')

fig = plt.figure()
plt.title('left_crop_img')
plt.imshow(crop_img)
fig.savefig('left_crop_img.png')

fig = plt.figure()
plt.title('left_resize_image')
plt.imshow(resize_image)
fig.savefig('left_resize_image.png')



fig = plt.figure()
plt.title('right_randomize_contrast_img')
plt.imshow(randomize_contrast_img2)
fig.savefig('right_randomize_contrast_img.png')

fig = plt.figure()
plt.title('right_crop_img')
plt.imshow(crop_img2)
fig.savefig('right_crop_img.png')

fig = plt.figure()
plt.title('right_resize_image')
plt.imshow(resize_image2)
fig.savefig('right_resize_image.png')

 
fig = plt.figure()
plt.title('center_randomize_contrast_img')
plt.imshow(randomize_contrast_img3)
fig.savefig('center_randomize_contrast_img.png')

fig = plt.figure()
plt.title('center_crop_img')
plt.imshow(crop_img3)
fig.savefig('center_crop_img.png')

fig = plt.figure()
plt.title('center_resize_image')
plt.imshow(resize_image3)
fig.savefig('center_resize_image.png')
'''
fig = plt.figure()
a=fig.add_subplot(1,2,1)
imgplot = plt.imshow(img)
a.set_title('left_img')
a=fig.add_subplot(1,2,2)
imgplot = plt.imshow(img_flipped)
a.set_title('flipped_left_img')
fig.savefig('flipped_left.png')

fig = plt.figure()
a=fig.add_subplot(1,2,1)
imgplot = plt.imshow(img2)
a.set_title('right_img')
a=fig.add_subplot(1,2,2)
imgplot = plt.imshow(img2_flipped)
a.set_title('flipped_right_img')
fig.savefig('flipped_right.png')

fig = plt.figure()
a=fig.add_subplot(1,2,1)
imgplot = plt.imshow(img3)
a.set_title('right_img')
a=fig.add_subplot(1,2,2)
imgplot = plt.imshow(img3_flipped)
a.set_title('flipped_center_img')
fig.savefig('flipped_center.png')