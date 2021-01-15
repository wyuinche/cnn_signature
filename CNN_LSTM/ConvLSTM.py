# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 21:42:29 2020

@author: minju
"""

from tensorflow.keras import models
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from PIL import Image
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import tensorflow_core as tf
import glob
import numpy as np
import time
import os
import random

os.environ['KMP_WARNINGS'] = 'off'
os.environ['KERAS_BACKEND'] = 'theano'

init = tf.compat.v1.global_variables_initializer()
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=2,
                        inter_op_parallelism_threads=4,
                        allow_soft_placement=True)

sess = tf.compat.v1.Session(config=config)
sess.run(init)

#define variables
train_sample_n = 28
valid_sample_n = 4
test_sample_n = 60
sample_width = 400
sample_height = 300
epoch = 10
batch = 10

# data loding
X_train = []
for img_path in glob.glob("sample/train_2/*.png"):
    X_train.append(np.array(Image.open(img_path).convert('L').getdata()))
    print(img_path)
X_train = np.asarray(X_train).reshape(train_sample_n, 10, sample_width, sample_height)
y_train = np.loadtxt('label_train.txt', delimiter=',', dtype=np.uint8)

X_valid = []
for img_path in glob.glob("sample/valid_2/*.png"):
    X_valid.append(np.array(Image.open(img_path).convert('L').getdata()))
    print(img_path)
X_valid = np.asarray(X_valid).reshape(valid_sample_n, 10, sample_width, sample_height)
y_valid = np.loadtxt('label_valid.txt', delimiter=',', dtype=np.uint8)

X_test = []
for img_path in glob.glob("sample/test_2/*.png"):
    X_test.append(np.array(Image.open(img_path).convert('L').getdata()))
    print(img_path)
X_test = np.asarray(X_test).reshape(test_sample_n, 10, sample_width, sample_height)
y_test = np.loadtxt('label_test.txt', delimiter=',', dtype=np.uint8)

# shuffle
tmp_train = [[x, y] for x, y in zip(X_train, y_train)]
random.shuffle(tmp_train)
X_train = [n[0] for n in tmp_train]
X_train = X_train[0:20]
y_train = [n[1] for n in tmp_train]
y_train = y_train[0:20]

# one-hot encoding
y_train = np.asarray(y_train)
y_train = to_categorical(y_train)
y_valid = np.asarray(y_valid)
y_valid = to_categorical(y_valid)
y_test = np.asarray(y_test)
#y_test = to_categorical(y_test)

# Scale
X_train = (X_train-255) * (-1./255)
X_valid = (X_valid-255) * (-1./255)
X_test = (X_test-255) * (-1./255)

CNN = models.Sequential()
CNN.add(TimeDistributed(Conv2D(96, (11, 11), activation='relu', input_shape=(sample_width, sample_height), strides=(4,4), padding='same'), input_shape=(10, sample_width, sample_height)))
CNN.add(TimeDistributed(MaxPool2D(pool_size=(3,3), strides=(2,2))))
CNN.add(TimeDistributed(Dropout(0.1)))
CNN.add(TimeDistributed(Conv2D(256, (5, 5), activation='relu', padding='same', strides=(1,1))))
CNN.add(TimeDistributed(MaxPool2D(pool_size=(3,3), strides=(2,2))))
CNN.add(TimeDistributed(Dropout(0.3)))
CNN.add(TimeDistributed(Conv2D(384, (3, 3), activation='relu', strides=(1,1), padding='same')))
CNN.add(TimeDistributed(Conv2D(1, (3, 3), activation='relu', padding='same', strides=(1, 1))))
CNN.add(TimeDistributed(Conv2D(1, (3, 3), activation='relu', padding='same', strides=(1, 1))))
CNN.add(TimeDistributed(MaxPool2D(pool_size=(3,3), strides=(2,2))))
CNN.add(TimeDistributed(Dropout(0.5)))
CNN.add(TimeDistributed(Flatten()))
CNN.add(LSTM(128, return_sequences=False))
CNN.add(Dense(2, activation='softmax'))
CNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
CNN.summary()

CNN.save('ConvLSTM.h5')

callback_list = [ModelCheckpoint(filepath='ConvLSTM_checkpoint.h5', monitor='val_loss', save_best_only=True), TensorBoard(log_dir="logs".format(time.asctime()))]
history = CNN.fit(X_train, y_train, batch_size=batch, epochs=epoch, shuffle=True, validation_data=(X_valid, y_valid), callbacks=callback_list)

epochs = np.arange(1, epoch+1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

y_predict = CNN.predict(X_test)
y_predict = np.argmax(y_predict, axis=1)

precision = precision_score(y_test, y_predict)
recall = recall_score(y_test, y_predict)
accuracy = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)

ace_tmp = np.bitwise_xor(y_test, y_predict)
ace = ace_tmp / (2*test_sample_n)

print("ACC: %f\n" %(accuracy))
print("PRE: %f\n" %(precision))
print("REC: %f\n" %(recall))
print("F1: %f\n" %(f1))
print("ACE: %f\n" %(ace))