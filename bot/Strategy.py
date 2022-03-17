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
ind = IndicatorConst()
prm = ParameterConst()



STATUS_WAIT = 0
STATUS_WAIT_ENTRY = 1
STATUS_ENTRIED = 2
STATUS_WAIT_EXIT = 3
STATUS_EXIT = 4
STATUS_ENTRY_FAIL = 5


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
    out = np.zeros(n)
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
    out = np.zeros(n)
    for i in range(n):
        if vector[i] > value:
            out[i] = value
        else:
            out[i] = vector[i]
    return out

class Position:
    def __init__(self, kind):
        self.is_open = False
        self.time_open = None
        self.time_close = None
        self.lot = None
        self.price_open = None
        self.price_close = None
        self.profit = None
        self.kind = kind
        self.count = 0
        pass
        
    def judgeClose(self, time, open, high, low, close, price):
        if self.is_open != True:
            return
        
        if self.kind == c.LONG:
            if high < price:
                return
        else:
            if low < price:
                return    
        self.time_close = time
        self.price_close = price
        self.profit = self.price_close - self.price_open
        if self.kind == c.SHORT:
            self.profit *= -1
        self.is_open = False
    

# Trend following
class PerfectOrder:
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
class RSICounter:
    
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
    
    
class ATRAleternatePosition(Position):
    
    def __init__(self, kind, horizon):
        self.kind = kind
        self.horizon = horizon
        self.status = STATUS_WAIT   
        self.count = None
        self.index_open = None
        self.index_close = None
        self.price_open = None
        self.price_close = None
        self.time_open = None
        self.time_close = None
        self.profit = None
    
    
    # STATUS_WAIT
    #  
    #  V  if buy_signal or sell_signal ON, entry price is decided
    #
    # STATUS_WAIT_ENTRY
    #
    #  V  if canBuy  or canSell
    #
    # STATUS_ENTRIED
    #
    #  v  if sell_signl or buy_singnal ON, exit price is decided
    #
    # STATUS_WAIT_EXIT
    #
    # v if canSEll or canBuy
    #
    # STATS_CLOSED
    
    
    
    def update(self, index, time, high, low, price):
        self.current_index = index
        if self.status == STATUS_EXIT:
            return self.status

        
                
        if self.status == STATUS_WAIT_ENTRY:
            if self.kind == c.LONG and self.canBuy(high, price):
                    self.time_open = time
                    self.index_open = index
                    self.price_open = price
                    self.status = STATUS_ENTRIED
                    self.count = 0
            else:
                self.status = STATUS_ENTRY_FAIL
            
            if self.kind == c.SHORT and self.canSell(low, price):
                    self.time_open = time
                    self.index_open = index
                    self.price_open = price
                    self.status == STATUS_ENTRIED
                    self.count = 0
            else:
                self.tatus = STATUS_ENTRY_FAIL
            return self.status
 
        
        if self.status == STATUS_ENTRIED:
            self.count += 1
            if self.count < self.horizon:
                return self.status
            else:
                self.status = STATUS_WAIT_EXIT
                
        if self.status == STATUS_WAIT_EXIT:
            if self.kind == c.LONG and self.canSell(low, price):
                self.time_close = time
                self.index_close = index
                self.price_close = price
                self.profit = self.price_close - self.price_open
                self.status = STATUS_EXIT
            elif self.kind == c.SHORT and self.canBuy(high, self.close):
                self.time_close = time
                self.index_close = index
                self.price_close = price
                self.profit = self.price_open - self.price_close  
                self.status = STATUS_EXIT
                
        return self.status
            


class AtrAlternate:
    def __init__(self, coeff, horizon):
        self.coeff = coeff
        self.horizon = horizon
        self.positions = []
        self.finished_postions = []
        self.time = []
        self.open = []
        self.high = []
        self.low = []
        self.close = []
        self.atr = []
        
    def tohlcvDic(self):
        dic = {}
        dic[c.TIME] = self.time
        dic[c.OPEN] = self.open
        dic[c.HIGH] = self.high
        dic[c.LOW] = self.low
        dic[c.CLOSE] = self.close
        return dic        
                
    def updateAtr(self):
        dic = self.tohlcvDic()
        atr = ATR(dic)
        self.atr = atr
        
    def updatePosition(self, index):
        time = self.time[-1]
        high = self.high[-1]
        low = self.low[-1]
        close = self.close[-1]
        price = self.atr[-2]
        upper = close + self.coeff * price
        lower = close - self.coeff * price
        positions = []
        for i in range(self.positions):
            pos = self.positions[i]
            if (pos.kind == c.LONG and pos.status == STATUS_WAIT_ENTRY) or (pos.kind == c.SHORT and pos.status == STATUS_WAIT_EXIT):
                status = pos.update(time, high, low, upper)
            if (pos.kind == c.SHORT and pos.status == STATUS_WAIT_ENTRY) or (pos.kind == c.LONG and pos.stauus == STATUS_WAIT_EXIT):
                status = pos.update(time, high, low, lower)
                
            if status == STATUS_ENTRY_FAIL:
                continue
            elif status == STATUS_EXIT:
                self.finished_postions.append(pos)
            else:
                positions.append(pos)
        self.positions = positions
        
                
        
    def update(self, index, tohlc:list):
        t = tohlc[0]
        op = tohlc[1]
        hi = tohlc[2]
        lo = tohlc[3]
        cl = tohlc[4]
        
        self.time.append(t)
        self.open.append(op)
        self.high.append(hi)
        self.low.append(lo)
        self.close.append(cl)       
        self.updateAtr()
        self.updatePosition(index)
        
        # Order
        pos1 = ATRAleternatePosition(c.LONG, self.horizon)
        self.positions.append(pos1)
        pos2 = ATRAleternatePosition(c.SHORT, self.horizon)
        self.positions.append(pos2)
    
        
    # return order price
    def limitOrder(self, ohlcv:dict):
        atr = ohlcv[ind.ATR]
        
        close = ohlcv[c.CLOSE]
        upper = close + atr * self.coeff     
        long_price = shift(upper, 1)
        lower = close - atr * self.coeff
        short_price = shift(lower, 1)
        return (long_price, short_price)
    
    def summary(self, length):
        long = []
        short = []
        for pos in self.finisehd_positions:
            if pos.kind == c.LONG:                
                long.append([pos.index_close, pos.time_open, pos.price_open, pos.price_close, pos.profit, pos.count])
            else:
                short.append([pos.index_close, pos.time_open, pos.price_open, pos.price_close, pos.profit, pos.count])
                 
        long_profit = np.zeros(length)
        long_profit_acc = np.zeros(length)
        acc = 0
        for index, _, _, _, profit, _ in long:
            acc += profit
            long_profit[index] = profit
            long_profit_acc[index] = acc
            
        short_profit = np.zeros(length)
        short_profit_acc = np.zeros(length)
        acc = 0
        for index, _, _, _, profit, _ in short:
            acc += profit
            short_profit[index] = profit
            short_profit_acc[index] = acc
            
        return (long_profit, long_profit_acc, short_profit, short_profit_acc)
    
    
    def position2array(self, kind, length, key:str):
        value = np.zeros(length)
        for position in self.finished_positions:
            if position.kind == kind:
                if key == 'price_open':
                    value[position.index_open] = position.price_open
                elif key == 'price_close':
                    value[position.index_close] = position.price_close        
        return value
    
    def summary2(self, tohlcv:dict):
        time = tohlcv[c.TIME]
        n = len(time)
        tohlcv['long_price_open'] = self.position2array(c.LONG, n, 'price_open')
        tohlcv['long_price_close'] = self.position2array(c.LONG, n, 'price_close')
        tohlcv['short_price_open'] = self.position2array(c.SHORT, n, 'price_open')
        tohlcv['short_price_close'] = self.position2array(c.SHORT, n, 'price_close')        
    
    def simulateEveryBar(self, tohlcv:dict, atr):
        time = tohlcv[c.TIME]
        open = tohlcv[c.OPEN]
        high = tohlcv[c.HIGH]
        low = tohlcv[c.LOW]
        close = tohlcv[c.CLOSE]
        
        for i, tohlc in enumerate(zip(time, open, high, low, close)):
            self.update(i, tohlc)
            
        self.summary2(tohlcv)
        return
    
    def allPosition(self):
        return self.positions

def test():
    return
    
    
    

if __name__ == '__main__':
    test()