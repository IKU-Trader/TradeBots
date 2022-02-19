# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 16:50:07 2022

@author: docs9
"""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from TechnicalAnalysis import SMA, RSI, ATR, movingAverage
from const import BasicConst, IndicatorConst, ParameterConst
c = BasicConst()
t = IndicatorConst()
p = ParameterConst()

def crossOver(vector, ref):
    n = len(vector)
    out = np.zeros(n)
    for i in range(1, n):
        if ref[i - 1] <= vector[i - 1] and ref[i] > vector[i]:
            out[i] = 1
    return out
    
def crossUnder(vector, ref):
    n = len(vector)
    out = np.zeros(n)
    for i in range(1, n):
        if ref[i - 1] >= vector[i - 1] and ref[i] < vector[i]:
            out[i] = 1
    return out

def greater(vector, value):
    n = len(vector)
    out = np.zeros(n)
    for i in range(n):
        if vector[i] > value:
            out[i] = 1
    return out

def smaller(vector, value):
    n = len(vector)
    out = np.zeros(n)
    for i in range(n):
        if vector[i] < value:
            out[i] = 1
    return out

def logicalAnd(vector1, vector2):
    n = len(vector1)
    out = np.zeros(n)
    for i in range(n):
        if vector1[i] > 0 and vector2[i] > 0:
            out[i] = 1
    return out
    
def fillZero(array):
    for i in range(len(array)):
        if np.isnan(array[i]):
            array[i] = 0

def isNan(vector):
    for v in vector:
        if np.isnan(v):
            return True
    return False   

def RegisterSupportPrice(array, n):
    X = []
    for v in array:
        X.append([v, 1.0])
    cluster = AgglomerativeClustering(n_clusters=n, affinity='euclidean', linkage='ward')
    cluster.fit_predict(X)
    return cluster.labels_


def shift(vector, offset):
    n = len(vector)
    out = np.full(n, np.nan)
    if offset > 0:
        for i in range(0, n - offset):
            out[i + offset] = vector[i]
    else:
        for i in range(offset, n):
            out[i - offset] = vector[i]
    return out

def equal(vector, value):
    n = len(vector)
    out = np.Zeros(n)
    for i in range(n):
        if vector[i] == value:
            out[i] = 1
    return out

def positiveEdge(vector):
    s = shift(vector, 1)
    dif = vector - s
    out = equal(dif, 1)
    return out

def limitUnder(vector, value):
    n = len(vector)
    out = np.Zeros(n)
    for i in range(n):
        if vector[i] > value:
            out[i] = value
        else:
            out[i] = vector[i]
    return out

    
# Trend following
class PerfectOrder():
    def __init__(self, windows):
        self.windows = windows
        return
    
    # return signal 1: order 0: not order
    def marketOrder(self, ohlcv: dict):
        sig1 = self.perfectOrder(ohlcv, self.windows, True)
        long = positiveEdge(sig1)
        sig2 = self.perfectOrder(ohlcv, self.windows, False)    
        short = positiveEdge(sig2)
        return (long, short)
    
    def inOrder(self, vector, is_ascend):
        if isNan(vector):
             return None    
        before = None
        for v in vector:
            if before is None:
                before = v
            else:
                if is_ascend:
                    if v <= before:
                        return False
                else:
                    if v >= before:
                        return False
            before = v
        return True
        
    def perfectOrder(self, ohlcv:dict, windows, is_ascend):
        mas = []
        for window in windows:
            v = SMA(ohlcv, window)
            mas.append(v)
        out = []
        n = len(mas[0])
        for i in range(n):
            v = []
            for j in range(len(mas)):
                v.append(mas[j][i])
            ret = self.inOrder(v, is_ascend)
            if ret is None:
                out.append(np.nan)
            else:
                if ret:
                    out.append(1)
                else:
                    out.append(0)
        return out  
    

# Counter Trend
class RSICounter():
    
    def __init__(self, rsi_window, ma_window, upper, lower):
        self.rsi_window = rsi_window
        self.ma_window = ma_window
        self.upper = upper
        self.lower = lower
        
    # return signal 1: order 0: not order
    def marketOrder(self, ohlcv:dict):
        rsi = RSI(ohlcv, self.rsi_window)
        ma_rsi = movingAverage(rsi, self.ma_window)
        long = self.long(rsi, ma_rsi)
        short = self.short(rsi, ma_rsi)
        return (long, short)
        
    def long(self, rsi, rsi_ma):
        v1 = greater(rsi, self.upper)
        v2 = crossOver(rsi, rsi_ma)
        out = logicalAnd(v1, v2)
        return out
        
    def short(self, rsi, rsi_ma):
        v1 = smaller(rsi, self.lower)
        v2 = crossUnder(rsi, rsi_ma)
        out = logicalAnd(v1, v2)
        return out   
    
        
class ATRCounter():
    def __init(self, atr_window, coeff):
        self.atr_window = atr_window
        self.coeff = coeff;
        
    # return order price
    def limitOrder(self, ohlcv:dict):
        close = ohlcv[c.CLOSE]
        atr = ATR(ohlcv, self.atr_window)
        atr = limitUnder(atr, 1.0)
        upper = close + atr * self.coeff     
        long_price = shift(upper, 1)
        lower = close - atr * self.coeff
        short_price = shift(lower, 1)
        return (long_price, short_price)

def test():
    return
    
    
    

if __name__ == '__main__':
    test()