from subprocess import Popen as popen
import pandas as pd
import numpy as np
import json
import random
import os

random.seed(7)

DIRECTORY = "experiment3/"
LABEL = 'Labels2'


def createLink(images, orig, dest, trainIndx, tagImage, label):
    destPath = dest + "train/"+ label +"/"
    flag = 0
    #print( "[INFO] dest: ", destPath, ", stock: ", tagImage, ", label", label)
    for img in images:
        if flag > trainIdx:
            destPath = dest + "validation/"+label+"/"
        im = tagImage + '_' + img +".png" 
        popen(['ln', '-s', orig + label +"/"+ im, im], cwd=os.path.abspath(destPath))
        flag += 1

with open('config.json', 'r') as f:
    config = json.load(f)

stock_symbols = config['STOCKS']['ALL']
folds = config['CV']['TRAIN_YEARS']
currentFold = 1

rootPath = "../../../../../images/"
t = ["train/", "validation/"]
l = ["buy", "hold", "sell"]

for fold in folds:
    for dirT in t:
        for dirL in l:
            os.makedirs(DIRECTORY+"fold_"+str(currentFold)+"/dataset/"+dirT+dirL, exist_ok=True)
    for stock in stock_symbols:
        df_stock = pd.read_csv("labels/"+stock+".csv",index_col="Date")
        image_keys = df_stock[fold[0]:fold[1]]
        buy_list = list()
        sell_list = list()
        hold_list = list()
        for img in image_keys.index:
            #print(type(image_keys.loc[img]['Labels']),image_keys.loc[img]['Labels'] )
            if image_keys.loc[img][LABEL] == np.nan:
                continue
            elif image_keys.loc[img][LABEL] == "buy":
                buy_list.append(img)
            elif image_keys.loc[img][LABEL] == "hold":
                hold_list.append(img)
            elif image_keys.loc[img][LABEL] == "sell":
                sell_list.append(img)
        random.shuffle(buy_list)
        random.shuffle(hold_list)
        random.shuffle(sell_list)
        numIdx = np.min(np.array([len(buy_list),len(hold_list),len(sell_list)]))
        print("[INFO] Fold: {0}, stock:{1}, numIdx: {2}".format(currentFold, stock, numIdx))
        trainIdx = int(numIdx*0.75)

        destPath = DIRECTORY+"fold_"+str(currentFold)+"/dataset/" 
        createLink(buy_list[0:numIdx], rootPath, destPath, trainIdx, stock, "buy")
        createLink(hold_list[0:numIdx], rootPath, destPath, trainIdx, stock, "hold")
        createLink(sell_list[0:numIdx], rootPath, destPath, trainIdx, stock, "sell")
    currentFold += 1

    #image_keys.to_csv("labels_f01.csv")


                
