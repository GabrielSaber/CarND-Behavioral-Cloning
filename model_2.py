# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 23:22:14 2017

@author: gsaber
"""

import csv

samples = []
with open('data/driving_log.csv') as csvfile:
    csvfile.readline()
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
import random

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                steering_center = float(batch_sample[3])
                # create adjusted steering measurements for the side camera images
                correction = 0.1 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction
        
                # read in images from center, left and right cameras
                path = 'data/' #fill in the path to your training IMG directory
                img_center = cv2.imread(path + batch_sample[0].lstrip())
                img_left = cv2.imread(path + batch_sample[1].lstrip())
                img_right = cv2.imread(path + batch_sample[2].lstrip())
                
                images.append(img_center)
                angles.append(steering_center)
                
                images.append(img_left)
                angles.append(steering_left)
                
                images.append(img_right)
                angles.append(steering_right)
                
                
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 160, 320  # Trimmed image format


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, Cropping2D


model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))
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
model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                    validation_data=validation_generator, 
                    nb_val_samples=len(validation_samples), nb_epoch=6)