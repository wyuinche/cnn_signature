# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 03:49:09 2020

@author: minju
"""
from PIL import Image
import glob
import numpy as np

# data loding

div = 230*160*50*1.0

count = 0
for img_path in glob.glob("CASE2_complexity_normal/*.png"):
    image = np.array(Image.open(img_path).convert('L').getdata())
    image = np.asarray(image).reshape(230, 160)
    print(img_path)
    for i in range(230):
        for j in range(160):
            if image[i, j] != 255:
                count += 1
                
print(count/div)
