from subprocess import Popen as popen
import pandas as pd
import numpy as np
import json
import random

random.seed(7)

image_keys = pd.read_csv("labels_f01.csv", index_col="Date")
with open('../../config.json', 'r') as f:
    config = json.load(f)

images = image_keys.index
stock_symbols = config['STOCKS']['ALL']
rootPath = "/home/agnaldo/unicamp/projetos/indicators/images/"

for stock in stock_symbols:
    #df_stock = pd.read_csv("../../labels/"+stock+".csv",index_col="Date")
    #image_keys[stock] = df_stock['Labels']
    buy_list = list()
    sell_list = list()
    hold_list = list()
    for img in images:
        #print(type(image_keys.loc[img]['Labels']),image_keys.loc[img]['Labels'] )
        if image_keys.loc[img][stock] == np.nan:
            continue
        elif image_keys.loc[img][stock] == "buy":
            buy_list.append(img)
        elif image_keys.loc[img][stock] == "hold":
            hold_list.append(img)
        elif image_keys.loc[img][stock] == "sell":
            sell_list.append(img)
    random.shuffle(hold_list)
    if len(buy_list) < len(sell_list):
        numIdx = len(buy_list)
    else:
        numIdx = len(sell_list)

    trainIdx = int(numIdx*0.75)
    for i in range(numIdx):
        if i <= trainIdx:
            destPath = "./dataset/train/"
        else:
            destPath = "./dataset/validation/"
        img_buy = "buy/"+stock + '_' +buy_list[i]+".png" 
        img_hold = "hold/"+stock + '_'+ hold_list[i]+".png"
        img_sell = "sell/"+stock + '_' + sell_list[i]+".png"
        #print(rootPath+img_buy)
        popen(["ln","-s",rootPath+img_buy, destPath+img_buy])
        popen(["ln","-s",rootPath+img_hold, destPath+img_hold])
        popen(["ln","-s",rootPath+img_sell, destPath+img_sell])

#image_keys.to_csv("labels_f01.csv")


            
