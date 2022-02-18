# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 14:20:18 2022

@author: docs9
"""

import numpy as np
import pandas as pd
from const import BasicConst, IndicatorConst, ParameterConst
import talib as ta

c = BasicConst()
t = IndicatorConst()
p = ParameterConst()



    
def SMA(ohlcv: dict, window, key=c.CLOSE):
    return ta.SMA(ohlcv[key], timeperiod=window)

def EMA(ohlcv: dict, window, key=c.CLOSE):
    return ta.EMA(ohlcv[key], timeperiod=window)    

def BB(ohlcv: dict, window, sigma, key=c.CLOSE):
    (up, mid, low) = ta.BBANDS(ohlcv[key], timeperiod=window, nbdevup=sigma, nbdevdn=sigma, matype=0)
    return (up, mid, low)

def BBRATIO(ohlcv: dict, window, sigma, key=c.CLOSE):
    price = ohlcv[key]
    (up, mid, low) = ta.BBANDS(price, timeperiod=window, nbdevup=sigma, nbdevdn=sigma, matype=0)
    width = up - low
    level = price - low
    ratio = level / width
    return ratio     
    
def ATR(ohlcv:dict, window):
    return ta.ATR(ohlcv[c.HIGH], ohlcv[c.LOW], ohlcv[c.CLOSE], timeperiod=window)

def TRANGE(ohlcv:dict):
    return ta.TRANGE(ohlcv[c.HIGH], ohlcv[c.LOW], ohlcv[c.CLOSE])
    
def ADX(ohlcv:dict, window):
    return ta.ADX(ohlcv[c.HIGH], ohlcv[c.LOW], ohlcv[c.CLOSE], timeperiod=window)

def ADXR(ohlcv:dict, window):
    return ta.ADXR(ohlcv[c.HIGH], ohlcv[c.LOW], ohlcv[c.CLOSE], timeperiod=window)

def PLUS_DI(ohlcv:dict, window):
    ta.PLUS_DI(ohlcv[c.HIGH], ohlcv[c.LOW], ohlcv[c.CLOSE], timeperiod=window)
    
def MINUS_DI(ohlcv:dict, window):
    ta.MINUS_DI(ohlcv[c.HIGH], ohlcv[c.LOW], ohlcv[c.CLOSE], timeperiod=window)
    
def AROON(ohlcv:dict, window):
    return ta.AROON(ohlcv[c.HIGH], ohlcv[c.LOW], timeperiod=window)

def RSI(ohlcv:dict, window):
    return ta.RSI(ohlcv[c.CLOSE], timeperiod=window)

def MACD(ohlcv:dict, fast, slow, signal):
    return ta.MACD(ohlcv[c.CLOSE], fastperiod=fast, slowperiod=slow, signalperiod=signal)

def ROC(ohlcv:dict, window):
    return ta.ROC(ohlcv[c.CLOSE], timeperiod=window)

def MOM(ohlcv:dict, window):
    return ta.MOM(ohlcv[c.CLOSE], timeperiod=window)

def APO(ohlcv:dict, fast, slow):
    return ta.APO(ohlcv[c.CLOSE], fastperiod=fast, slowperiod=slow, matype=0)

def SAR(ohlcv:dict, acc, maxv):
    return ta.SAR(ohlcv[c.HIGH], ohlcv[c.LOW], acceleration=acc, maximum=maxv)

def RCI(ohlcv:dict, window):
    close = ohlcv[c.CLOSE]
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(n):
        if (i + window)> n:
            break
        y = close[i:i + window]
        x_rank = np.arange(len(y))
        y_rank = rankdata(y, method='ordinal') - 1
        sum_diff = sum((x_rank - y_rank)**2)
        rci = (1 - ((6 * sum_diff) / (window**3 - window))) * 100
        out.append(rci)
    return out

def fillZero(array):
    for i in range(len(array)):
        if np.isnan(array[i]):
            array[i] = 0
            
def movingAverage(array, window, fill_zero=True):
    d = pd.Series(array)
    d.rolling(window).mean()
    if fill_zero:
        fillZero(d)
    return d

def ADXDMI(ohlcv: dict, adx_window, ma_window):
    adx = ADX(ohlcv, adx_window)
    ma = movingAverage(adx, ma_window, fill_zero=False)
    return ma

def VQ(ohlcv: dict):
    trange = TRANGE(ohlcv)
    high = ohlcv[c.HIGH]
    low = ohlcv[c.LOW]
    open = ohlcv[c.OPEN]
    close = ohlcv[c.CLOSE]
    n = len(close)
    vo = np.zeros(n)
    q = np.zeros(n)
    for i in range(1, n):
        if trange[i] != 0 and (high[i] - low[i]) != 0:
            vo[i] = (close[i] - close[i - 1]) / trange[i] + (close[i] - open[i]) / (high[i] - low[i]) * 0.5
        else:
            vo[i] = vo[1]
    for i in range(1, n):
        q[i] = np.abs(vo[i]) * ((close[i] - close[i - 1]) + (close[i] -open[i])) * 0.5
    vq = np.zeros(n)
    s = q[0]
    for i in range(1, n):
        s += q[i]
        vq[i] = s
    return vq


class Indicator:
    
    @classmethod
    def makeIndicators(cls, indicators:dict):
        out = []
        for key, values in indicators.items():
            ind = Indicator(values[t.TYPE], key)
            for k, v in values.items():
                ind.param(k, v)
            out.append(ind)
        return out 
    
    def __init__(self, typ, name):
        self.typ = typ.strip().lower()
        self.name = name.strip()
        self.params = {}
        
    def param(self, key, value):
        self.params[key] = value
        
    def calc(self, ohlcv: dict):
        if self.typ == t.SMA:
            return SMA(ohlcv, self.params[p.WINDOW])
        if self.typ == t.EMA:
            return EMA(ohlcv, self.params[p.WINDOW])
        if self.typ == t.BBUP:
            (up, down, mid) = BB(ohlcv, self.params[p.WINDOW], self.params[p.SIGMA])
            return up
        if self.typ == t.BBDOWN:
            (up, down, mid) = BB(ohlcv, self.params[p.WINDOW], self.params[p.SIGMA])
            return down
        if self.typ == t.BBRATIO:
            return BBRATIO(ohlcv, self.params[p.WINDOW], self.params[p.SIGMA])
        if self.typ == t.ATR:
            return ATR(ohlcv, self.params[p.WINDOW])
        if self.typ == t.TRANGE:
            return TRANGE(ohlcv)
        if self.typ == t.ADX:
            return ADX(ohlcv, self.params[p.WINDOW])
        if self.typ == t.PLUS_DI:
            return PLUS_DI(ohlcv, self.params[p.WINDOW])
        if self.typ == t.MINUS_DI:
            return MINUS_DI(ohlcv, self.params[p.WINDOW])       
        if self.typ == t.ADXR:
            return ADXR(ohlcv)
        if self.typ == t.AROON:
            return AROON(ohlcv, self.params[p.WINDOW])       
        if self.typ == t.RSI:
            return RSI(ohlcv, self.params[p.WINDOW]) 
        if self.typ == t.MACD:
            return MACD(ohlcv, self.params[p.FAST], self.params[p.SLOW], self.params[p.SINGAL])
        if self.typ == t.ROC:
            return ROC(ohlcv, self.params[p.WINDOW])
        if self.typ == t.MOMENTUM:
            return MOM(ohlcv, self.parms[p.WINDOW])
        if self.typ == t.APO:
            return APO(ohlcv, self.params[p.FAST], self.params[p.SLOW])
        if self.typ == t.VQ:
            return VQ(ohlcv)
        if self.typ == t.SAR:
            return SAR(ohlcv, self.params[p.ACC], self.params[p.MAX])
        if self.typ == t.RCI:
            return RCI(ohlcv, self.params[p.WINDOW])
        
    def description(self):
        out = "[" + self.name + "] " + self.typ + "  "
        for key, value in self.params.item():
            out +=  (key + ":" + str(value) + " ")
        return out
        
        
    