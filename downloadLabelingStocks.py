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

window = config['DEFAULT']['WINDOW']
lag = window//2
path = config['DEFAULT']['HEAD_FILE']
s_date = config['DEFAULT']['DATE_START']
e_date = config['DEFAULT']['DATE_END']

os.makedirs(path, exist_ok=True)

for stock in config['STOCKS']['ALL']:
    # Download time series as DataFrame
    print("Downloading {} stock prices".format(stock))
    df_stockPrices = yt.download(stock, start=s_date, end=e_date)

    # Labeling data
    print("Labeling data ...")
    closePrices = df_stockPrices['Adj Close'].values
    labels = list()
    for i in range(len(closePrices)):
        if i < lag or i > (len(closePrices)-lag -1):
            labels.append('')
        else:
            temp = closePrices[i-lag:i+(lag+1)]
            if (np.amin(temp) == closePrices[i]):
                labels.append('buy')
            elif (np.amax(temp) == closePrices[i]):
                labels.append('sell')
            else:
                labels.append('hold')
    df_stockPrices["Labels"] = labels
    df_stockPrices.to_csv(path+stock+".csv")
    
print("Finish. Stocks date are in datasets fold")          
        
