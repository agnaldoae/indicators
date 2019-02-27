from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import json
import pandas as pd
from sklearn.metrics import confusion_matrix


def annualizedReturns(startMoney, endMoney, numYears):
    m = (endMoney/startMoney)**(1/numYears)
    ar = (m-1)*100
    return ar

def financialEvaluation(prices, signals, capital):
    num_stocks = capital // prices[0]
    new_capital = 0
    num_transactions = 0
    signalLastTrade = 0
    for signal, price in zip(signals, prices):
        # if Buy signal
        if signal == 0 and signalLastTrade != 0:
            num_stocks = new_capital // price
            new_capital = 0
            signalLastTrade = 0
            num_transactions += 1
            #print("buy- ",num_stocks)
        # if Sell signal
        elif signal == 2 and signalLastTrade == 0:
            new_capital = num_stocks * price
            num_stocks = 0
            signalLastTrade = 1
            num_transactions += 1
            #print("sell- $",new_capital)
        #print("price: {0:4.2f}; signal: {1}; num_stock: {2:3.1f}, new_capital: {3:6.2f}".format(price, signal, num_stocks, new_capital))   
        #else: print("hold")
    
    if new_capital == 0 and num_stocks != 0:
        new_capital = num_stocks * prices[-1]
        
    return new_capital, num_transactions

if __name__ == "__main__":
    
    directoryTest = "experiment3"
    #directoryTest = "experimento"
    initialCapital = 10000.0
    trasactionTax = 1.0
    sum_cm = np.zeros([3,3])
    # load .json file
    with open('config.json','r') as f:
        config = json.load(f)

    nFolds = config['CV']['NFOLDS']
    lstocks = list()
    l_ar = list()
    # For each stock
    for stock in config['STOCKS']['ALL']:
        # load .csv
        print("Stock: ", stock)
        df_stockPrices = pd.read_csv('labels/'+stock+'.csv', index_col='Date')
        capital = initialCapital
        ntransactions = 0
        currentFold = 1
    #    For each fold
        for years in config['CV']['TEST_YEARS']:
            print("Years ", years)
            predictions = list()
            actualLabels = list()
        #    load cnn model
            cnn = load_model(directoryTest+'/fold_'+str(currentFold)+'/cnn_model')
        #    get subset date [start:end]
            subset_stockPrices = df_stockPrices.loc[years[0]:years[1]]
        #    For each date
            for day in subset_stockPrices.index:
        #        predict label
                #print("---- DAY: ", day)
                label = subset_stockPrices.loc[day,'Labels']
                imagePath = 'images/'+label+'/'+stock+'_'+day+'.png'
                #print("   imagePath ", imagePath)
                image = cv2.imread(imagePath, 0) # Load an color image in grayscale
                # preprocessing the image for classifcation
                #image = cv2.resize(image, (150,150))
                image = image.astype('float32') / 255.0 # rescale [0,1]
                image = img_to_array(image)
                image = np.expand_dims(image, axis=0)
                p = cnn.predict(image)
                #print(".........Forecast", p)
                predictions.append(np.argmax(p))
                actualLabels.append(label)
        #        includ label into the list
            #for i in range(len(actualLabels)):
             #   print("Label {}, Prediction {};".format(actualLabels[i],(predictions[i])))
            capital, ntransactions = financialEvaluation(subset_stockPrices['Adj Close'].values, predictions, capital)
            #capital -=(ntransactions * trasactionTax)
        #    print Confusion matrix into the respective k-fold directory
            y_test = np.ones(len(actualLabels))
            for i in range(len(actualLabels)):
                if actualLabels[i] == "buy":
                    y_test[i] = 0
                elif actualLabels[i] == "sell":
                    y_test[i] = 2
            cm = confusion_matrix(y_test, predictions)
            #capital, ntransactions = financialEvaluation(subset_stockPrices['Adj Close'].values, y_test, capital)
            if np.shape(cm)[0] == 3:
                sum_cm += cm
            print(cm)
        #    Annualized Return
            print("Capital: {0:.2f}; ntransactions {1}".format(capital,ntransactions))
            currentFold += 1
            #break
        ar = annualizedReturns(initialCapital,capital,nFolds)
        print("Annualized Return: {0:.2f}%".format(ar))
        print("- - - - - - - - - - - - - - - - -")
        lstocks.append(stock)
        l_ar.append(ar)
        #break
    print("Aggregated Accuracy:\n",sum_cm)
    for s,r in zip(lstocks,l_ar):
        print("{0}...{1:.2f}%".format(s,r))
    array = np.array(l_ar)
    print("Average: {0:.2f}, Std: {1:.2f}".format(np.average(array),np.std(array)))    
