# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 12:30:54 2022

@author: docs9
"""

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
    

features2 =          {
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
                        'STOCHASTIC_slowk': {ind.TYPE: ind.STOCHASTIC_SLOWK, p.FASTK: 5, p.SLOWK: 3, p.SLOWD: 3},
                        'STOCHASTIC_slowd': {ind.TYPE: ind.STOCHASTIC_SLOWD, p.FASTK: 5, p.SLOWK: 3, p.SLOWD: 3},
                        'STOCHASTIC_fastk': {ind.TYPE: ind.STOCHASTIC_FASTK, p.FASTK: 5, p.FASTD: 3},
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
                        'BB_UP': {ind.TYPE: ind.BB_UP, p.WINDOW: 5, p.SIGMA: 2},
                        'BB_DOWN': {ind.TYPE: ind.BB_DOWN, p.WINDOW: 5, p.SIGMA: 2},
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
    long_profit_acc = tohlcv['long_profit_acc']
    short_profit_acc = tohlcv['short_profit_acc']
   
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
       
def vectorize(data: dict, keys):
    d = data[keys[0]]
    n = len(d)
    m = len(keys) - 1
    data_list = []
    for key in keys:
        data_list.append(data[key])
    x = []
    y = []
    indices = []
    for i in range(n):
        vector = []
        for j in range(m + 1):
            vector.append(data_list[j][i])
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
        lgb.plot_importance(model, figsize=(12, 6))
    
    return pred
    
def forceZero(predict, y):
    length = y.shape[0]
    for i in range(length):
        if y[i] == 0.0:
            predict[i] = 0.0
    return
    

def filtered(ror, predict, is_long):
    out = [0.0]
    for i in range(1, len(ror)):
        if is_long:
            if predict[i - 1] > 0:
                out.append(ror[i])
            else:
                out.append(0.0)
        else:
            if predict[i - 1] < 0:
                out.append(ror[i])
            else:
                out.append(0.0)
    return out


def index(array, indices):
    out = []
    for index in indices:
        out.append(array[index])
    return out

def fitting(features:dict, data:dict):
    indicators = Indicator.makeIndicators(features)
    for indicator in indicators:
        indicator.calc(data)
        
    ind2 = Indicator(ind.ROR, 'ROR')
    ind2.calc(data)
    
    keys = ['ROR'] + list(features.keys())
    x, y, indices = vectorize(data, keys)
    
    model = lgb.LGBMRegressor(n_jobs=-1, random_state=1)
    predict = crossValidation(model, x, y, list(features.keys()))
    forceZero(predict, y)
    evaluate(y, predict)
    
    short_ror = data['short_ror']
    new_short_ror = filtered(short_ror[indices], predict, False)
    new_short_ror_acc = Math.accumulate(new_short_ror)

    long_ror = data['long_ror']
    new_long_ror = filtered(long_ror[indices], predict, True)
    new_long_ror_acc = Math.accumulate(new_long_ror)
    
    
    time = index(data['time'], indices)
    (fig, axes) = gridFig([1], (12, 4))
    fig.subplots_adjust(hspace=0.6, wspace=0.4)
    graph1= CandlePlot(fig, axes[0], 'ror acc fitted')
    graph1.drawLine(time, new_long_ror_acc, color='red')
    graph1.drawLine(time, new_short_ror_acc, color='blue') 
    
    
def evaluate(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
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
    fitting(features1, data)
    
    
def testBitfly():
    server = BitflyData('btcjpy', 'M15')
    server.loadFromCsv('../data/bitflyer/btcjpy_m15.csv') 
    data = trade(server, 0.5, 14)
    fitting(features2, data)
    #print(data.keys())
    
    
if __name__ == '__main__':
    testBitfly()