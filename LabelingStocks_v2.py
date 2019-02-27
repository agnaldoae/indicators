# -*- coding: utf-8 -*-
# Usage: python3 downloadLabelingStocks.py

import json
import pandas as pd
import numpy as np
import talib
import fix_yahoo_finance as yt
import os


with open('config.json', 'r') as f:
    config = json.load(f)

window = config['DEFAULT']['LAG_RPS']
lag = window//2
path = "./labels/"


os.makedirs(path, exist_ok=True)

for stock in config['STOCKS']['ALL']:
    df_stockPrices = pd.read_csv(path+stock+".csv",index_col="Date")

    # Labeling data
    print("Labeling data ...stock:", stock)
    oldLabels = df_stockPrices['Labels']
    labels = list()
    flag = 0
    for i in range(len(oldLabels)):
        if i < lag or i > (len(oldLabels)-lag -1):
            labels.append('')
        else:
            if (oldLabels[i]=='buy' or oldLabels[i] == 'sell'):
                labels.append(oldLabels[i])
            elif (oldLabels[i] == 'hold'):
                flag = 0
                for m in range(1,lag):
                    if (oldLabels[i-m] != 'hold' or oldLabels[i+m] != 'hold'): 
                        flag = 1
                if (flag == 1):
                    labels.append('')
                else:
                    labels.append('hold')
            else:
                labels.append('')
    df_stockPrices["Labels2"] = labels
    df_stockPrices.to_csv(path+stock+".csv")
    
print("Finish. Stocks date are in datasets fold")          
        
