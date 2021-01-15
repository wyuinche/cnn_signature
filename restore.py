# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 10:12:59 2020

@author: minju
"""
import glob
import numpy as np
from tensorflow.keras.utils import to_categorical
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

restored_model = load_model('NET3_CASE2_pen.h5')
restored_model.load_weights('NET3_CASE2_pen_checkpoint.h5')

X_test = []
for img_path in glob.glob("CASE2_pen/t_sign_out/*.png"):
    X_test.append(np.array(Image.open(img_path).convert('RGBA').getdata()))
    print(img_path)
for img_path in glob.glob("CASE2_pen/f_sign_out/*.png"):
    X_test.append(np.array(Image.open(img_path).convert('RGBA').getdata()))
    print(img_path)
for img_path in glob.glob("CASE2_pen/f_train_sign_out/*.png"):
    X_test.append(np.array(Image.open(img_path).convert('RGBA').getdata()))
    print(img_path)
X_test = np.asarray(X_test).reshape(80, 10, 230, 160, 4)
y_test = np.loadtxt('label_test2.txt', delimiter=',', dtype=np.uint8)

y_test = np.asarray(y_test)
y_test_onehot = to_categorical(y_test)

restored_model.evaluate(X_test, y_test_onehot)
y_predict = restored_model.predict(X_test)
y_predict = np.argmax(y_predict, axis=1)

print("y_test")
print(y_test)
print("y_predict")
print(y_predict)
