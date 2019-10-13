# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 20:05:06 2019

@author: Tiancheng Zhao & Xingyu Bai
"""


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from pandas import read_csv
from statsmodels.tsa.statespace.sarimax import SARIMAX

from xgboost import XGBRegressor

file_path_train = 'Ten-Year-Demand.csv'
file_path_test = 'Two-Year-Test.csv'


data = pd.read_csv(file_path_train)


from datetime import date
data = read_csv('Ten-Year-Demand.csv')
data.columns = ['Year', 'Month', 'Demand']
data = data.fillna(method='ffill')
data_xgb = data.copy()

data['Month'] = data.apply(lambda x: 12 if x['Month']%12 == 0 else x['Month']%12, axis = 1)
data['Month'] = data.apply(lambda x: date(int(x['Year']), int(x['Month']), 1), axis = 1)

del data['Year']

data['Month'] = pd.to_datetime(data['Month'])
data = data.set_index("Month")
data.head()

data.index.freq = 'MS'

final_data = read_csv(file_path_test)


final_data.columns = ['Year', 'Month', 'Demand']
final_data = final_data.fillna(method='ffill')
final_data_xgb = final_data.copy()

final_data['Month'] = final_data.apply(lambda x: 12 if x['Month']%12 == 0 else x['Month']%12, axis = 1)
final_data['Month'] = final_data.apply(lambda x: date(int(x['Year']), int(x['Month']), 1), axis = 1)
del final_data['Year']

final_data['Month'] = pd.to_datetime(final_data['Month'])
final_data = final_data.set_index("Month")
final_data.head()

final_data.index.freq = 'MS'

frames = [data,final_data]
all_data = pd.concat(frames)

start_test = len(all_data)-len(final_data)
test_end = len(all_data)
test_length = len(final_data)

def output(x, q, d1, d2):
    """
    x: starting inventory
    q: arriving order quantity
    d1: demand this month
    d2: predicted demand next month
    return:
    y: ending inventory
    hc: mothly holding cost
    bc: monthly backorder cost
    o: order quantity
    """
    
    y = x + q - d1
    if y>0:
        hc = min(y, 90)*1 + max(0, y-90)*2
        bc = 0
    else:
        bc = -3*y
        hc = 0
    o = max(0,d2-y)+3
    return y, hc, bc, o

def output_cost(hold, backorder):
    total_hc = np.sum(hold) # total holding cost
    ave_hc = np.mean(hold) # average holding cost
    total_bc = np.sum(backorder) # total backorder cost
    ave_bc = np.mean(backorder) # average backorder cost
    total_cost = total_hc + total_bc # total cost
    return total_cost, total_hc, ave_hc, total_bc, ave_bc

start = 73
oq = 0

test_data = all_data[start_test:test_end]

prediction_thismonth_arima = []
for i in range(start_test,test_end):
    train_data_thismonth = data[0:i]
    arima_model_thismonth = SARIMAX(train_data_thismonth['Demand'], order=(3,0,2), seasonal_order=(0,1,1,12))
    arima_result_thismonth = arima_model_thismonth.fit(maxiter=200)
    arima_predict = arima_result_thismonth.predict(start = len(train_data_thismonth),end=len(train_data_thismonth), typ="levels").rename("Arima Prediction")
    prediction_thismonth_arima.append(arima_predict[0])

prediction_thismonth_arima.append(0)


frames = [data_xgb,final_data_xgb]
data = pd.concat(frames)
data.columns = ['year', 'time', 'x']


data['month'] = data['time'].apply(lambda x: 12 if x%12 == 0 else x%12)
ave = []
for i in range(12):
    dd = data[data['month'] == i+1]
    ave.append(np.mean(dd['x']))
data['average'] = data['month'].apply(lambda x: ave[x-1])
pre = list(data['x'])
pre.insert(0,np.mean(ave))
pre = pre[:len(pre)-1]
data['pre1'] = pre
pre.insert(0,np.mean(ave))
pre = pre[:len(pre)-1]
data['pre2'] = pre
data = data.fillna(method='ffill')

X = data[['year','time','month','average','pre1','pre2']].copy()
Y = data[['x']].copy()
X_train = X[:test_end-test_length]
X_test = X[test_end-test_length:test_end]
y_train = Y[:test_end-test_length]
y_test = Y[test_end-test_length:test_end]

arfreg = XGBRegressor(max_depth=6,
min_child_weight=1,
eta=.3,
subsample=1,
colsample_bytree=1,
objective='reg:linear')
arfreg.fit(X_train, y_train)
prediction_thismonth_xgb = arfreg.predict(X_test)
prediction_thismonth_xgb = np.append(prediction_thismonth_xgb,[0])

prediction_ensemble = (prediction_thismonth_arima+prediction_thismonth_xgb)/2


begin_inv = []
order_quan = []
end_inv = []
holding = []
backorder = []

for t in range(len(test_data)):
    begin_inv.append(start)
    start, holdc, backc, oq = output(start, oq, np.array(test_data['Demand'])[t], prediction_ensemble[t+1])
    order_quan.append(oq)
    end_inv.append(start) # Ending inventory this month is the same as beginning inventory next month
    holding.append(holdc)
    backorder.append(backc)

totalc, totalhc, avehc, totalbc, avebc = output_cost(holding, backorder)

summary = open("summary.txt", 'w+')
summary.write("Total cost: " + str(totalc) + "\n")
summary.write("Total holding cost: " + str(totalhc) + "\n")
summary.write("Average holding cost: " + str(avehc) + "\n")
summary.write("Total backorder cost: " + str(totalbc) + "\n")
summary.write("Average backorder cost: " + str(avebc) + "\n")
summary.close()

detail = test_data
detail['start inventory'] = begin_inv
detail['order quantity'] = order_quan
detail['end inventory'] = end_inv
detail['holding cost'] = holding
detail['backorder cost'] = backorder
detail.to_csv(r'summary.csv')
