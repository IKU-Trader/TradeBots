# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 12:30:54 2022

@author: docs9
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

from DataServer import GemforexData, BitflyData
from DataBuffer import DataBuffer
from StubAPI import StubAPI

from Math import Math
from const import BasicConst, IndicatorConst, ParameterConst
from TechnicalAnalysis import Indicator
from CandlePlot import CandlePlot, BandPlot, makeFig, gridFig, array2graphShape
from Strategy import AtrAlternate
from pandas_utility import dic2df

c = BasicConst()
ind = IndicatorConst()
p = ParameterConst()

features1 = {        
                        'ATR': {ind.TYPE:ind.ATR, p.WINDOW: 14},
                        'BBRatio': {ind.TYPE:ind.BB_RATIO, p.WINDOW: 20, p.SIGMA: 2.0},
                        'ADX':{ind.TYPE:ind.ADX, p.WINDOW: 14},
                        'PPO':{ind.TYPE:ind.PPO, p.FAST: 12, p.SLOW: 26},
                        'AroonOSC': {ind.TYPE:ind.AROON_OSC, p.WINDOW: 25},
                        'CCI': {ind.TYPE:ind.CCI, p.WINDOW: 20},
                        'MACDHist': {ind.TYPE:ind.MACD_HIST, p.FAST: 12, p.SLOW: 26, p.SIGNAL: 9},
                        'MFI': {ind.TYPE:ind.MFI, p.WINDOW:14},
                        'MOM': {ind.TYPE:ind.MOMENTUM, p.WINDOW: 26},
                        'RSI': {ind.TYPE:ind.RSI, p.WINDOW: 14},
                        'HTDCPeriod': {ind.TYPE: ind.HT_DC_PERIOD},
                        'HTDCPhase': {ind.TYPE: ind.HT_DC_PHASE},
                        'HTPhasorInphase': {ind.TYPE: ind.HT_PHASOR_INPHASE},
                        'HTPhasorQuadrature': {ind.TYPE: ind.HT_PHASOR_QUADRATURE},
                        'HTTrendmode': {ind.TYPE: ind.HT_TRENDMODE},
                        'HTTrendline': {ind.TYPE: ind.HT_TRENDLINE},
                        'LINEArreg': {ind.TYPE: ind.LINEARREG, p.WINDOW: 14},
                        'LINEArregAngle': {ind.TYPE: ind.LINEARREG_ANGLE, p.WINDOW: 14},
                        'LINEArregIntercept': {ind.TYPE: ind.LINEARREG_INTERCEPT, p.WINDOW: 14},
                        'LINEArregSlope': {ind.TYPE: ind.LINEARREG_SLOPE, p.WINDOW: 14},
                        'Beta': {ind.TYPE: ind.BETA, p.WINDOW: 5},
                        'Kama': {ind.TYPE: ind.KAMA, p.WINDOW: 30},
                        'CandleBody': {ind.TYPE: ind.CANDLE_BODY},
                        'Spike': {ind.TYPE: ind.SPIKE},
                        'Weekday': {ind.TYPE: ind.WEEKDAY},
                        'TimeBand': {ind.TYPE: ind.TIMEBAND}
                       }
    

features0 =          {
                        'ADX':{ind.TYPE:ind.ADX, p.WINDOW: 14},
                        'ADXR':{ind.TYPE:ind.ADXR, p.WINDOW: 14},
                        'APO': {ind.TYPE:ind.APO, p.FAST: 12, p.SLOW: 26},
                        'AROON_DOWN': {ind.TYPE: ind.AROON_DOWN, p.WINDOW: 14},
                        'AROON_UP': {ind.TYPE: ind.AROON_UP, p.WINDOW: 14},
                        'AROON_OSC': {ind.TYPE: ind.AROON_OSC, p.WINDOW: 14},
                        'CCI': {ind.TYPE:ind.CCI, p.WINDOW: 14},
                        'DX': {ind.TYPE:ind.DX, p.WINDOW: 14},
                        'MACD': {ind.TYPE:ind.MACD, p.FAST: 12, p.SLOW: 26, p.SIGNAL: 9 },
                        'MACD_signal': {ind.TYPE:ind.MACD_SIGNAL, p.FAST: 12, p.SLOW: 26, p.SIGNAL: 9 },
                        'MACD_hist': {ind.TYPE:ind.MACD_HIST, p.FAST: 12, p.SLOW: 26, p.SIGNAL: 9 },
                        'MFI': {ind.TYPE:ind.MFI, p.WINDOW:14},
                        'MOM': {ind.TYPE:ind.MOMENTUM, p.WINDOW: 10},
                        'RSI': {ind.TYPE:ind.RSI, p.WINDOW: 14},
                        'STOCHASTIC1_slowk': {ind.TYPE: ind.STOCHASTIC1_SLOWK, p.FASTK: 5, p.SLOWK: 3, p.SLOWD: 3},
                        'STOCHASTIC2_slowd': {ind.TYPE: ind.STOCHASTIC2_SLOWD, p.FASTK: 5, p.SLOWK: 3, p.SLOWD: 3},
                        'STOCHASTIC3_fastk': {ind.TYPE: ind.STOCHASTIC3_FASTK, p.FASTK: 5, p.FASTD: 3},
                        'ULTOSC': {ind.TYPE: ind.ULTOSC, p.FAST: 7, p.MID: 14, p.SLOW: 28},
                        'WILLR': {ind.TYPE: ind.WILLR, p.WINDOW: 14},
                        'HT_DC_period': {ind.TYPE: ind.HT_DC_PERIOD},
                        'HT_DC_phase': {ind.TYPE: ind.HT_DC_PHASE},
                        'HT_PHASOR_inphase': {ind.TYPE: ind.HT_PHASOR_INPHASE},
                        'HT_PHASOR_quadrature': {ind.TYPE: ind.HT_PHASOR_QUADRATURE},
                        'HT_TRENDLINE': {ind.TYPE: ind.HT_TRENDLINE},
                        'HT_TRENDMODE': {ind.TYPE: ind.HT_TRENDMODE},
                        'Beta': {ind.TYPE: ind.BETA, p.WINDOW: 5},
                        'LINEARREG': {ind.TYPE: ind.LINEARREG, p.WINDOW: 14},
                        'LINEARREG_ANGLE': {ind.TYPE: ind.LINEARREG_ANGLE, p.WINDOW: 14},
                        'LINEARREG_INTERCEPT': {ind.TYPE: ind.LINEARREG_INTERCEPT, p.WINDOW: 14},
                        'LINEARREG_SLOPE': {ind.TYPE: ind.LINEARREG_SLOPE, p.WINDOW: 14},
                        'STDDEV': {ind.TYPE: ind.STDDEV, p.WINDOW: 5, p.SIGMA: 1},
                        'BB_UPPER': {ind.TYPE: ind.BB_UPPER, p.WINDOW: 5, p.SIGMA: 2},
                        'BB_LOWER': {ind.TYPE: ind.BB_LOWER, p.WINDOW: 5, p.SIGMA: 2},
                        'BB_MID': {ind.TYPE: ind.BB_MID, p.WINDOW: 5, p.SIGMA: 2},
                        'DEMA': {ind.TYPE: ind.DEMA, p.WINDOW: 30},
                        'EMA': {ind.TYPE: ind.EMA, p.WINDOW: 30},
                        'KAMA': {ind.TYPE: ind.KAMA, p.WINDOW: 30},
                        'MA': {ind.TYPE: ind.MA, p.WINDOW: 30},
                        'MIDPOINT': {ind.TYPE: ind.MIDPOINT, p.WINDOW: 14},
                        'T3': {ind.TYPE: ind.T3, p.WINDOW: 5},
                        'TEMA': {ind.TYPE: ind.TEMA, p.WINDOW: 30},
                        'TRIMA': {ind.TYPE: ind.TRIMA, p.WINDOW: 30},
                        'WMA': {ind.TYPE: ind.WMA, p.WINDOW: 30}
                       }
    
def trade(server, atr_coef, atr_window):
    indicator_param = {'ATR': {ind.TYPE:ind.ATR, p.WINDOW:atr_window}}
    indicators = Indicator.makeIndicators(indicator_param)

    api = StubAPI(server)    
    buffer = DataBuffer(indicators)    
    buffer.loadData(api.allData())
    #print('len: ', buffer.length(), buffer.technical.keys())
    #print('from:', buffer.fromTime(), 'to: ', buffer.lastTime())
    #tohlcv, technical_data = buffer.dataByDate(2020, 7, 5)
    tohlcv = buffer.tohlcv
    technical_data = buffer.technical

    atr = technical_data['ATR']
    atralt = AtrAlternate(atr_coef, 1, indicators)

    summary = atralt.simulateEveryBar(tohlcv)
    #print(tohlcv.keys())
    #save('./trade_gold.csv', tohlcv, [c.TIME, c.OPEN, c.HIGH, c.LOW, c.CLOSE, 'ATR', 'long_price_open', 'long_price_close', 'short_price_open', 'short_price_close'])

    time = array2graphShape(tohlcv, [c.TIME])
    long_profit_acc = tohlcv['long_ror_acc']
    short_profit_acc = tohlcv['short_ror_acc']
    
    
    
   
    (fig, axes) = gridFig([1], (12, 4))
    fig.subplots_adjust(hspace=0.6, wspace=0.4)
    #graph1 = CandlePlot(fig, axes[0], server.title())
    #keys = [c.OPEN, c.HIGH, c.LOW, c.CLOSE]
    #graph1.drawCandle(time, tohlcv, keys)
   #graph1.drawLine(time, long_price)
   #graph1.drawLine(time, short_price, color='blue')
      
    #graph2 = CandlePlot(fig, axes[0], 'ATR')
    #graph2.drawLine(time, atr, color='red')
    graph3 = CandlePlot(fig, axes[0], 'Profit of ' + server.title() + '  coef: ' + str(atr_coef) + ' window: ' + str(atr_window))
    graph3.drawLine(time, long_profit_acc, color='red')
    graph3.drawLine(time, short_profit_acc, color='blue')    
    
    return tohlcv


def isnanInVector(vector):
    for v in vector:
        if np.isnan(v):
            return True
        elif v is None:
            return True
    return False
       
def vectorize(data: dict, x_keys, y_key):
    d = data[x_keys[0]]
    n = len(d)
    m = len(x_keys) - 1
    x_data = []
    y_data = data[y_key]
    for x_key in x_keys:
        x_data.append(data[x_key])
    x = []
    y = []
    indices = []
    for i in range(n):
        vector = []
        if i < n - 1:
            vector.append(y_data[i + 1])
        else:
            vector.append(0)
        for j in range(m + 1):
            vector.append(x_data[j][i])
        if  isnanInVector(vector) == False:
            y.append(vector[0])
            x.append(vector[1:])
            indices.append(i)
    return (np.array(x), np.array(y), indices)

def splitData(size, n_splits):
    l = int(size / n_splits)
    out = []
    for i in range(n_splits):
        train = []
        test = []
        begin = i * l
        stop = (i + 1) * l
        if i == n_splits - 1:
            stop = size
        for j in range(begin, stop):
            test.append(j)
        for j in range(0, begin):
            train.append(j)
        for j in range(stop, size):
            train.append(j)
        out.append([train, test])
    return out


def crossValidation(model, x, y, features):
    length = y.shape[0]
    indices = splitData(length, 5)
    pred = np.full(y.shape, np.nan)
    for [train_index, test_index] in indices:
        model.fit(x[train_index], y[train_index], feature_name=features)
        pred[test_index] = model.predict(x[test_index])
        #lgb.plot_importance(model, figsize=(8, 6))
    
    return pred


def predict(model, x, y, features, rate, debug=False):
    n = y.shape[0]
    m = int(float(n) * rate)
    pred = np.full(n, np.nan)
    trainX = x[:m]
    trainY = y[:m]
    testX = x[m:]
    model.fit(trainX, trainY, feature_name=features)
    pred[m:] = model.predict(testX)
    
    if debug:
        saveArray(trainX, '../report/mytrade_trainX.csv')
        saveArray(trainY, '../report/mytrade_trainY.csv')
        saveArray(testX, '../report/mytrade_testX.csv')
        saveArray(pred, '../report/mytrade_pred.csv')
    #lgb.plot_importance(model, figsize=(8, 6)
    
    
    return pred
    

def filtered(ror, predict, is_long, threshold=None):
    out = [0.0]
    for i in range(1, len(ror)):
        value = predict[i - 1]
        if is_long:
            if threshold is not None:
                if value < threshold:
                    out.append(0.0)
                    continue
            if value > 0:
                out.append(ror[i])
            else:
                out.append(0.0)
        else:
            if threshold is not None:
                if value > threshold * -1.0:
                    out.append(0.0)
                    continue
            if value < 0:
                out.append(ror[i])
            else:
                out.append(0.0)
    return out


def index(array, indices):
    out = []
    for index in indices:
        out.append(array[index])
    return out


def saveArray(array, filepath):
    rows = array.shape[0]
    if len(array.shape) > 1:
        cols = array.shape[1]
    else:
        cols = 1
    f = open(filepath, mode='w')
    for row in range(rows):
        s = ''
        for col in range(cols):
            if cols == 1:
                s += str(array[row]) + ','
            else:
                s += str(array[row, col]) + ','
        f.write(s + '\n')
    f.close()
    
def isNan(vector):
    for v in vector:
        if v is None:
            return True
        if np.isnan(v):
            return True
    return False


def dropNan(data: dict, target_keys=None) ->dict:
    arrays = []
    target_index = []
    if target_keys is None:
        target_keys = data.keys()
    for i, key in enumerate(data.keys()):
        arrays.append(data[key])
        if key in target_keys:
            target_index.append(i)

    n = len(arrays[0])
    rows = []
    for row in range(n):
        vector = []
        for i in target_index:
            vector.append(arrays[i][row])
        if not isNan(vector):
            rows.append(row)

    slices = []
    begin = None
    old = None
    for row in rows:
        if begin is None:
            begin = row
            old = row
        else:
            if row > old + 1:
                slices.append([begin, old])
                begin = None
            else:
                old = row
    if begin is not None:
        slices.append([begin, old])
            
    dic = {}
    for key in data.keys():
        d = data[key]
        ary = np.array([])
        for b, e in slices:
            d0 = d[b: e + 1]
            ary = np.hstack([ary, d0])
        dic[key] = ary
    return dic
    

# short/long別々に学習して予測する。
def fitting0(name, features:dict, data:dict):
    

    indicators = Indicator.makeIndicators(features)
    for indicator in indicators:
        indicator.calc(data)
        
    ind2 = Indicator(ind.ROR, 'ROR')
    ind2.calc(data)
    
    #dic2df(data).to_csv('../report/mytrade_before_drop.csv', index=False)
    
    keys = sorted(list(features.keys()))
    data = dropNan(data, target_keys = keys)
    #dic2df(data1).to_csv('../report/mytrade_after_drop.csv', index=False)
    
    
    
    model = lgb.LGBMRegressor(n_jobs=-1, random_state=1)
    
    y_key = 'short_ror'
    x, y, indices = vectorize(data, keys, y_key)
    #short_predict = crossValidation(model, x, y, list(features.keys()))
    short_predict = predict(model, x, y, list(features.keys()), 0.5, debug=True)
    evaluate(y, short_predict)
    
    short_ror = data['short_ror']
    ml_short_ror = filtered(short_ror[indices], short_predict, False)
    ml_short_ror_acc = Math.accumulate(ml_short_ror)

    y_key = 'long_ror'
    x, y, indices = vectorize(data, keys, y_key)
    #long_predict = crossValidation(model, x, y, list(features.keys()))
    long_predict = predict(model, x, y, list(features.keys()), 0.5)
    evaluate(y, long_predict)

    long_ror = data['long_ror']
    ml_long_ror = filtered(long_ror[indices], long_predict, True)
    ml_long_ror_acc = Math.accumulate(ml_long_ror)
    
    
    time = index(data['time'], indices)
    (fig, axes) = gridFig([1], (12, 4))
    fig.subplots_adjust(hspace=0.6, wspace=0.4)
    graph1= CandlePlot(fig, axes[0], 'ror acc fitted')
    graph1.drawLine(time, ml_long_ror_acc, color='red')
    graph1.drawLine(time, ml_short_ror_acc, color='blue') 
    
    
    df = dic2df(data)
    df2 = df.iloc[indices]
    df2['short_predict'] = short_predict
    df2['long_predict'] = long_predict
    df2['ml_short_ror'] = ml_short_ror
    df2['ml_long_ror'] = ml_long_ror
    
    columns = ['time', 'open', 'high', 'low', 'close', 'ATR', 'buy_price', 'sell_price', 'buy_signal', 'long_open', 'long_close', 'long_ror', 'long_ror_acc', 'sell_signal', 'short_open', 'short_close', 'short_ror', 'short_ror_acc', 'short_predict', 'long_predict', 'ml_short_ror', 'ml_long_ror'  ]
    
    feature_name = sorted(list(features.keys()))
    df3 = df2[columns + feature_name]
    df3.to_csv('../report/ml_mytrade_' + name + '.csv', index=False)
    

# long/short一緒に学習する。
def fitting1(name, features:dict, data:dict):
    indicators = Indicator.makeIndicators(features)
    for indicator in indicators:
        indicator.calc(data)
        
    ind2 = Indicator(ind.ROR, 'ROR')
    ind2.calc(data)
    
    keys = sorted(list(features.keys()))
    x, y, indices = vectorize(data, keys, 'ROR')
    
    model = lgb.LGBMRegressor(n_jobs=-1, random_state=1)
    predict = crossValidation(model, x, y, keys)
    evaluate(y, predict)
    
    short_ror = data['short_ror']
    new_short_ror = filtered(short_ror[indices], predict, False, threshold=0.005)
    new_short_ror_acc = Math.accumulate(new_short_ror)

    long_ror = data['long_ror']
    new_long_ror = filtered(long_ror[indices], predict, True, threshold=0.005)
    new_long_ror_acc = Math.accumulate(new_long_ror)
    
    
    time = index(data['time'], indices)
    (fig, axes) = gridFig([1], (12, 4))
    fig.subplots_adjust(hspace=0.6, wspace=0.4)
    graph1= CandlePlot(fig, axes[0], 'ror acc fitted')
    graph1.drawLine(time, new_long_ror_acc, color='red')
    graph1.drawLine(time, new_short_ror_acc, color='blue') 
    
    
    df = dic2df(data)
    df2 = df.iloc[indices]
    df2['predict'] = predict
    df2['short_ror'] = new_short_ror
    df2['long_ror'] = new_long_ror
    df2.to_csv('../report/ml_mytrade2_' + name + '.csv', index=False)


def MAE(vector1, vector2):
    s = 0.0
    count = 0
    for v1, v2 in zip(vector1, vector2):
        if np.isnan(v1) == False and np.isnan(v2) == False:
            s += np.abs(v1 - v2)
            count += 1
    if count == 0:
        return -1.0
    else:
        return s / float(count)
        

def MSE(vector1, vector2):
    s = 0.0
    count = 0
    for v1, v2 in zip(vector1, vector2):
        if np.isnan(v1) == False and np.isnan(v2) == False:
            s += np.power(v1 - v2, 2.0)
            count += 1
    if count == 0:
        return -1.0
    else:
        return s / float(count)    
    
def RMSE(vector1, vector2):
    mse = MSE(vector1, vector2)
    if mse < 0:
        return -1.0
    else:
        return np.sqrt(mse)
    
    
    
    
    
def evaluate(y_test, y_pred):
    mae = MAE(y_test, y_pred)
    mse = MSE(y_test, y_pred)
    rmse = RMSE(y_test, y_pred)
    model = LinearRegression(
                         fit_intercept=True,
                         normalize=False,
                         copy_X=True, 
                         n_jobs=1,
                         positive=True
                        )

    cor = np.corrcoef(y_test, y_pred)
    er = error_rate(y_test, y_pred)
    
    print('MAE: ', mae)
    print('MSE: ', mse)
    print('RMSE: ', rmse)
    print('Error Rate: ', er)
    print('correlation coef: ', cor[0][1])
    
    fig, ax = makeFig(1, 1, (8, 8))
    ax.scatter(y_test, y_pred, alpha=0.5, s=5)
    ax.set_xlim(-0.1, 0.2)
    ax.set_ylim(-0.1, 0.2)
    
def error_rate(test, pred):
    count = 0
    for t, p in zip(test, pred):
        if t * p < 0:
            count += 1
    return count / len(test)
    
    
def testGemforex1():
    title = 'Gold'
    interval = 'M5'
    csv_path = '../data/gemforex/XAUUSD_M5_202010262200_202203281055.csv'
    server = GemforexData(title, interval)
    server.loadFromCsv(csv_path)
    for window in [7, 10, 14, 21]:
        for coef in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:    
            trade(server, coef, window)

def testGemforex2():
    title = 'Gold'
    interval = 'M5'
    csv_path = '../data/gemforex/XAUUSD_M5_202010262200_202203281055.csv'
    server = GemforexData(title, interval)
    server.loadFromCsv(csv_path)
    coef = 0.5
    window = 14
    data = trade(server, coef, window)    
    fitting1(features1, data)
    
    
def testBitfly():
    server = BitflyData('btcjpy', 'M15')
    path = '../data/bitflyer/btcjpy_m15.csv'
    dirpath, filepath = os.path.split(path)
    name, _ = os.path.splitext(filepath)
    
    server.loadFromCsv(path) 
    data = trade(server, 0.5, 14)
    df = dic2df(data)
    print(df.columns)
    columns = ['time', 'open', 'high', 'low', 'close', 'ATR', 'buy_price', 'sell_price', 'buy_signal', 'long_open', 'long_close', 'long_ror', 'long_ror_acc', 'sell_signal', 'short_open', 'short_close', 'short_ror', 'short_ror_acc']
    df2 = df[columns]
    df2.to_csv('../report/mytrade2_' + name + '.csv')
    fitting1(name, features1, data)
    
    #print(data.keys())
    
    
if __name__ == '__main__':
    testBitfly()