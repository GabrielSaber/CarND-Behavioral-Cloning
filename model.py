# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 23:15:52 2017

@author: gsaber
"""

import csv
import cv2
import numpy as np

images = []
measurements = []

def add_image_steering(image, steering):
    images.append(image)
    measurements.append(steering)


csv_file = 'data/driving_log.csv'
with open(csv_file, 'r') as f:
    f.readline()
    reader = csv.reader(f)
    for row in reader:
        steering_center = float(row[3])
        # create adjusted steering measurements for the side camera images
        correction = 0.2 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras
        path = 'data/' #fill in the path to your training IMG directory
        img_center = cv2.imread(path + row[0].lstrip())
        img_left = cv2.imread(path + row[1].lstrip())
        img_right = cv2.imread(path + row[2].lstrip())
        
        add_image_steering(img_center, steering_center)
        add_image_steering(img_left, steering_left)
        add_image_steering(img_right, steering_right) 
        
x_train = np.array(images)
y_train = np.array(measurements)

print(x_train.shape)
print(y_train.shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, Cropping2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2) ,activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2) ,activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2) ,activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))
        
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')
