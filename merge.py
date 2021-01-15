# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:01:12 2020

@author: minju
"""

import glob
from PIL import Image
import numpy as np

def CalculateSize(files):
    size_x=[]
    size_y=[]
    for file in files:
        image =Image.open(file)
        size_x.append(image.size[0])
        size_y.append(image.size[1])
    x_min = min(size_x)
    y_min = min(size_y) 
    total_x_size = x_min * len(files)
    total_y_size = y_min * len(files)
    return x_min,y_min,total_x_size,total_y_size

def To_MIN(files,x_min,y_min,x_size,y_size):
    file_list=[]
    for file in files:
        image = Image.open(file)
        resized_file = image.resize((x_min,y_min))
        file_list.append(resized_file)
    return file_list, x_size, y_size,x_min, y_min

def ImageMerge(seq, file_list,x_size,y_size,x_min,y_min):
    if seq<10 : saveImageFileName= "000" + str(seq) + "_merged1.png"
    elif seq<100 : saveImageFileName = "00" + str(seq) + "_merged1.png"
    else : saveImageFileName= "0" + str(seq) + "_merged1.png"
    new_image = Image.new("RGB",(x_size, y_min),(256,256,256))
    for index in range(len(file_list)):
        area=(x_min*index, 0, x_min*(index+1), y_min) 
        new_image.paste(file_list[index],area) 
    new_image.save(saveImageFileName,"PNG")
    return new_image
 
def Paste_Image(files):
    i = 1
    for idx in range(int(len(files)/5)):
        target_files = files[idx*5:(idx+1)*5]
        x_min,y_min,x_size,y_size=CalculateSize(target_files)
        file_list,x_size,y_size,x_min,y_min = To_MIN(target_files,x_min,y_min,x_size,y_size)
        image=ImageMerge(i, file_list,x_size,y_size,x_min,y_min)
        i += 1     
    return image

def ImageMerge2(seq, file_list,x_size,y_size,x_min,y_min):
    if seq<10 : saveImageFileName= "000" + str(seq) + "_merged2.png"
    elif seq<100 :saveImageFileName = "00" + str(seq) + "_merged2.png"
    else : saveImageFileName= "0" + str(seq) + "_merged2.png"
    new_image = Image.new("RGB",(x_min, y_size),(256,256,256))
    for index in range(len(file_list)):
        area=(0, y_min*index, x_min, y_min*(index+1)) 
        new_image.paste(file_list[index],area) 
    new_image.save(saveImageFileName,"PNG")
    return new_image

def Paste_Image2(files):
    i = 1
    for idx in range(int(len(files)/2)):
        target_files = files[idx*2:(idx+1)*2]
        x_min,y_min,x_size,y_size=CalculateSize(target_files)
        file_list,x_size,y_size,x_min,y_min = To_MIN(target_files,x_min,y_min,x_size,y_size)
        image=ImageMerge2(i, file_list,x_size,y_size,x_min,y_min)
        i += 1     
    return image

files=glob.glob("*.png")
Paste_Image2(files)


