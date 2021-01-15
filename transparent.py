# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 22:11:29 2020

@author: minju
"""

from PIL import Image
import os, glob

def trans(file_name):
    img = Image.open(file_name)
    img = img.convert("RGBA")
    datas = img.getdata()
    newData = []
    for item in datas:
        if item[0] > 200 and item[1] > 200 and item[2] > 200:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img.putdata(newData)
    filename = os.path.splitext(file_name)
    img.save(filename[0] + "_trans"  + filename[1], "PNG")
    
for img_path in glob.glob("*.png"):
    trans(img_path)