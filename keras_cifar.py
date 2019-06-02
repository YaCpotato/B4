#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 23:25:31 2019

@author: not me
"""
import time
import numpy as np
import keras.backend as K
from keras.datasets import cifar10
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#CentOSなら下記はコメントアウト↓
#plt.use('agg')
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

#precision算出関数
def P(y_true, y_pred):
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.20), 'float32'))
    pred_positives = K.sum(K.cast(K.greater(K.clip(y_pred, 0, 1), 0.20), 'float32'))
    precision = true_positives / (pred_positives + K.epsilon())
    return precision

#recall算出関数
def R(y_true, y_pred):
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.20), 'float32'))
    poss_positives = K.sum(K.cast(K.greater(K.clip(y_true, 0, 1), 0.20), 'float32'))
    recall = true_positives / (poss_positives + K.epsilon())
    return recall


#各種評価関数プロット
def plot_acc():
    plt.figure()
    plt.title('Test Accuracy and Validation Accuracy')
    plt.plot(range(1,epoch+1),result.history['acc'],label='acc',color='royalblue')
    plt.plot(range(1,epoch+1),result.history['val_acc'],label='val_acc',color='orange')
    plt.xlabel('Epochs(epoch)')
    plt.legend()
    plt.savefig('cifar_acc.png')

def plot_Precision():
    plt.figure()
    plt.title('Test Precision and Validation Precision')
    plt.plot(range(1,epoch+1),result.history['P'],label='pre',color='royalblue')
    plt.plot(range(1,epoch+1),result.history['val_P'],label='val_pre',color='orange')
    plt.xlabel('Epochs(epoch)')
    plt.legend()
    plt.savefig('cifar_pre.png')

def plot_Recall():
    plt.figure()
    plt.title('Test Recall and Validation Recall')
    plt.plot(range(1,epoch+1),result.history['R'],label='rec',color='royalblue')
    plt.plot(range(1,epoch+1),result.history['val_R'],label='val_rec',color='orange')
    plt.xlabel('Epochs(epoch)')
    plt.legend()
    plt.savefig('cifar_rec.png')

def plot_loss():
    plt.figure()
    plt.title('Test Loss and Validation Loss')
    plt.plot(range(1,epoch+1),result.history['loss'],label='loss',color='royalblue')
    plt.plot(range(1,epoch+1),result.history['val_loss'],label='val_loss',color='orange')
    plt.xlabel('Epochs(epoch)')
    plt.legend()
    plt.savefig('cifar_loss.png')


#モデルの定義
def model():
    model = Sequential()
    
    #畳み込み層　：　出力を入力と同じ32×32にするために、34×34にパディング(0をいれる)、モデル第一層目は入力の形を明示する。
    #34×34×3の画像に3×3のフィルタ32個で畳み込む。
    #パラメータは3×3×3×32+32
    #フィルタをかけるx,y,z(必ずしもRGBの3ではないことに注意)　×　フィルタの数　＋　バイアス項(Convでは=フィルタの数、Denseでは=ユニット数となる)
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=x_train.shape[1:]))
    #活性化関数relu　：　0以下は0にして、それ以外はそのまま出力する
    model.add(Activation('relu'))
    
    #畳み込み層　：　今回は32×32の画像にを3×3のフィルタをかける際にパディングしないので出力ピクセル数は30×30となる、フィルタ数は32個
    ##34×34×32の画像に3×3のフィルタ32個で畳み込む。
    #パラメータは3×3×32×32+32
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    
    #プーリング層　：　pool_sizeの行列をかぶらないようにかけていく。2×2のなかで最大値となる値をとり、1つのピクセルとする。
    #（pool_sizeが正方行列で、 ピクセル数は1/pool_sizeとなる）
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #ドロップアウト層　：　訓練時において入力されたデータをランダムに０にする割合
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    #平滑化層　：　データを一次元化する
    model.add(Flatten())
    
    #全結合層　：　特徴量により512個のユニットにわける
    model.add(Dense(512))
    model.add(Activation('relu'))
    
    model.add(Dropout(0.5))
    
    #最終的に0~9のラベル判定をしたいので10個のユニットにわける
    model.add(Dense(10))
    
    #活性化関数softmax　：　出力結果を正の値に変換し、データの総数で割り、総和を１にする（確率になる）
    model.add(Activation('softmax'))
    
    
    #loss　：　誤差の計算法
    #optimizer　：　学習法
    #metrics　：　評価値（model.historyにいれて後から確認するやつ）
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',P,R])
    
    model.summary()
    return model

if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    #学習用データ、正解データをロードする。（最初はかなり時間がかかる）
    
    #型をfloatにして255で割る。こうすると計算がしやすい
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    #学習数
    epoch = 50
    
    #ラベルをワンホットエンコーディングに変換
    #(ラベルの数分の正方行列、各行は、インデックスがラベルの数、の要素が１、他は０になる。)
    #例：ラベルが0,1,2の場合
    #[[1,0,0],
    # [0,1,0],
    # [0,0,1]]
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    
    
    
    model = model()
    start = time.time()
    
    #学習　※metricsに指定したパラメータ等がmodel.historyに格納される
    #x_train　：　学習用データ
    #y_train　：　学習用ラベル
    #batch_size：バッチサイズ　：　いくつ学習するごとにデータを評価する（acc,loss等の更新）か
    #epochs　：　学習数
    #validation_split　：　学習データのうち、何割を評価用に使うか
    #verbose　：　途中経過の出力
    result = model.fit(x_train, y_train, batch_size=100, epochs=epoch, validation_split=0.1,verbose=1)
    totaltime = time.time() - start
    print(totaltime)
    
    
    #予測された数値をワンホットエンコーディングラベルに変換する
    pred = model.predict(x_test)
    
    
    #ワンホットエンコーディングの予測、正解ラベルを、0~9の並びに戻す。
    pred = np.argmax(pred, axis=1)
    Y = np.argmax(y_test,axis=1)
    
    
    #混合行列の出力
    #列成分：予測ラベル、行成分：正解ラベル
    print(confusion_matrix(pred,Y))
    
    
    #各評価のプロット
    plot_acc()
    plot_Precision()
    plot_Recall()
    plot_loss()
