#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:10:07 2019

@author: shoichi
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import time
from keras import backend as K
K.set_image_data_format('channels_last')
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Conv2D,Activation,MaxPooling2D
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam
import train1000

def model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['categorical_crossentropy', 'accuracy'])
    model.summary()
    return model


if __name__ == '__main__':
    epochs = 50
    batch_size = 10
    (X_train, Y_train), (X_test, Y_test) = train1000.cifar10()
    nb_classes = 10
    model = model()
    result = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,verbose=1)
    print('train data:')
    eva=model.evaluate(X_train,Y_train,verbose=1)
    for i in range(1,len(model.metrics_names)):
        print( model.metrics_names[i] + ' : ', eva[i] )
    print()
    print('test data:')
    eva = model.evaluate(X_test,Y_test,verbose=1)
    for i in range(1,len(model.metrics_names)):
        print( model.metrics_names[i] + ' : ', eva[i] )
