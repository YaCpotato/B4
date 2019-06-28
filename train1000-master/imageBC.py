#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:30:47 2019

学習

@author: shoichi
"""

from PIL import Image
import numpy as np
import cv2
import data
import train1000
import keras
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Conv2D,Activation,MaxPooling2D,BatchNormalization
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn import preprocessing

def model():
	model = Sequential()
	model.add(Conv2D(64, (5,5), padding="same", input_shape=X_train.shape[1:],activation="relu"))                                            #畳み込み１層目
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(64, (5,5), padding="same", activation="relu"))         #畳み込み2層目
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	model.add(Dense(512, activation="relu"))
	model.add(BatchNormalization())
	model.add(Dense(10))                                              #全結合2層目
	model.add(BatchNormalization())
	model.add(Activation("softmax"))
	model.compile(loss='kullback_leibler_divergence', optimizer=Adam(), metrics=['kullback_leibler_divergence', 'acc'])
	
	model.summary()
	return model

def BClearning_generator(X_train,Y_train):
	label = np.argmax(Y_train,axis=1)
	for i in range(X_train.shape[0] -1 ):
		if(label[i+1] < label[i]):
			temp = X_train[i]
			X_train[i] = X_train[i+1]
			X_train[i+1] = temp
			temp2 = Y_train[i]
			Y_train[i] = Y_train[i+1]
			Y_train[i+1] = temp2
		
	#5クラスずつにソート半分ずつに分ける500,500
	sub_X_train1 , sub_X_train2 = np.array_split(X_train,2)
	#各ラベルのXをと、各ラベル+5のXをweight（0.1,0.9）をかけあう
	for i in range(len(sub_X_train1) -1 ):
		np.append(X_train,cv2.addWeighted(X_train[i],0.3,sub_X_train2[i],0.7,0))
		np.append(X_train,cv2.addWeighted(X_train[i+500],0.7,sub_X_train1[i],0.3,0))
	return X_train,Y_train


if __name__ == '__main__':
	(X_train,Y_train),(X_test,Y_test) = train1000.cifar10()
	X_train,Y_train = BClearning_generator(X_train,Y_train)
		
	epochs = 30
	batch_size = 100
	nb_classes = 10
	
	model = model()
	result = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,verbose=1,shuffle=True)
	
	print('train data:')
	eva=model.evaluate(X_train,Y_train,verbose=1)
	for i in range(1,len(model.metrics_names)):
		print( model.metrics_names[i] + ' : ', eva[i] )
		print()
	
	print('test data:')
	eva = model.evaluate(X_test,Y_test,verbose=1)
	for i in range(1,len(model.metrics_names)):
		print( model.metrics_names[i] + ' : ', eva[i] )

