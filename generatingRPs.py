#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import cv2 as cv

# Binarization
def binarization(matrix, threshold):
    matrix[matrix < threshold] = 0.0
    matrix[matrix >= threshold] = 1.0
    return matrix

# Recurrence (Distance) Plot 
def rplot(series, err=0.03, bin=0):
    dim = len(series)
    rp = np.zeros((dim,dim))
    for x in range(dim):
        for y in range(dim):
            rp[x,y] = abs(series[x] - series[y])
    if (bin == 1):
        rp = binarization(rp, err)
    return rp

def Mat2Image(matrix, fileName):
    minimun = np.amin(np.min(matrix))
    maximun = np.amax(np.amax(matrix))
    diff = maximun-minimun
    #print("max= %.1f, min= %.1f"%(minimun,maximun))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i,j] = 255*((matrix[i,j]-minimun)/(diff))
    cv.imwrite(fileName, matrix)



if __name__ == "__main__":
    lag = int(input("Enter Lag size: "))
    folderPath = input("Enter folder path: ")
    symbol = input("Enter stock symbol: ")
    window = 5 #used to label time series

    os.makedirs("./rps/sell",exist_ok=True)
    os.makedirs("./rps/hold",exist_ok=True)
    os.makedirs("./rps/buy",exist_ok=True)
    

    while symbol != '!':
        df = pd.read_csv (folderPath+'/'+symbol+'.csv', engine='python', sep=',')
        ts = df["Adj Close"].values.astype('float32')
        labels = df["Labels"].values
        indexs = df["Date"].values

        for i in range(lag, len(ts)-window):
            subseries = ts[i-lag:i]
            rp = rplot(subseries)
            Mat2Image(rp,"./rps/"+labels[i]+"/"+labels[i]+"."+indexs[i]+"."+symbol+".jpg" )

        symbol = input("Enter stock symbol or ! to stop: ")

