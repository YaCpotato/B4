#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import backend as K
K.set_image_data_format('channels_last')
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Conv2D, SpatialDropout2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
import numpy as np
import random
import os
import sys

def model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['categorical_crossentropy', 'accuracy'])
    model.summary()
    return model


if( __name__ == '__main__' ):

	epochs = 100
	batch_size = 100
	
	(X_train, Y_train), (X_test, Y_test) = train1000.cifar10()
	nb_classes = 10
	
	model = model()

	print( 'train data:' )
	eva = model.evaluate( X_train, Y_train, verbose=1 )
	for i in range(1,len(model.metrics_names)):
		print( model.metrics_names[i] + ' : ', eva[i] )

	print()

	print( 'test data:' )
	eva = model.evaluate( X_test, Y_test, verbose=1 )
	for i in range(1,len(model.metrics_names)):
		print( model.metrics_names[i] + ' : ', eva[i] )

