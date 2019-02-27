#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import cv2 as cv
import json
import talib

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
    with open('config.json','r') as f:
        config = json.load(f)

    lag = config['DEFAULT']['LAG_RPS']
    folderPath = "./labels"
    window = config['DEFAULT']['WINDOW']

    os.makedirs("./rps/sell",exist_ok=True)
    os.makedirs("./rps/hold",exist_ok=True)
    os.makedirs("./rps/buy",exist_ok=True)
    

    for symbol in config['STOCKS']['ALL']:
        df = pd.read_csv (folderPath+'/'+symbol+'.csv', engine='python', sep=',')
        ts = df["Adj Close"].values.astype('float32')
        ts_c = np.copy(ts)
        ts_c[lag:] = talib.EMA(ts, timeperiod=lag)
        labels = df["Labels"].values
        indexs = df["Date"].values

        print("Stock: ",symbol)
        for i in range(30):
            print(index[i], ts[i], ts_c[i],labels[i])
        break
        #for i in range(lag, len(ts)-window):
         #   subseries = ts_c[i-lag:i+1]
          #  rp = rplot(subseries)
          #  Mat2Image(rp,"./rps/"+labels[i]+"/"+symbol+"_"+indexs[i]+".png" )

print("Finish :-)")

