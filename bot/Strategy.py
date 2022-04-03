# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 16:50:07 2022

@author: docs9
"""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from TechnicalAnalysis import TaLib
from TechnicalLibrary import S_ATR
from const import BasicConst, IndicatorConst, ParameterConst
c = BasicConst()
ind = IndicatorConst()
prm = ParameterConst()

STATUS_NONE = 0
STATUS_ENTRY_ORDER = 1
STATUS_ENTRIED = 2
STATUS_EXIT_ORDER = 3
STATUS_DONE = 4
STATUS_ENTRY_ORDER_FAIL = 5


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
            v = TaLib.SMA(ohlcv, window)
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
        rsi = TaLib.RSI(ohlcv, self.rsi_window)
        ma_rsi = TaLib.movingAverage(rsi, self.ma_window)
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
    
    def __init__(self, idno, kind, horizon):
        self.idno = idno
        self.kind = kind
        self.horizon = horizon
        self.status = STATUS_ENTRY_ORDER
        self.count = None
        self.index_open = None
        self.index_close = None
        self.price_open = None
        self.price_close = None
        self.time_open = None
        self.time_close = None
        self.profit = None
    
    #                         if entry failed
    # (1) STATUS_ENTRY_ORDER       --->       (5) STATUS_ENTRY_FAIL
    #  
    #  V  if entry is success   
    #
    # (2) STATUS_ENTRIED
    #
    #  V  After horizon time 
    #
    # (3) STATUS_EXIT_ORDER
    #
    #  v  if position is closed
    #
    # (4) STATUS_DONE

    def update(self, index, time, high, low, price):
        self.current_index = index
        if self.status == STATUS_DONE:
            return self.status

        if self.status == STATUS_ENTRY_ORDER:
            if self.kind == c.LONG:
                if self.canBuy(low, price):
                    self.time_open = time
                    self.index_open = index
                    self.price_open = price
                    self.status = STATUS_ENTRIED
                    self.count = 0    
                else:
                    self.status = STATUS_ENTRY_ORDER_FAIL
                    
            elif self.kind == c.SHORT:
                if self.canSell(high, price):
                    self.time_open = time
                    self.index_open = index
                    self.price_open = price
                    self.status = STATUS_ENTRIED
                    self.count = 0
                else:
                    self.status = STATUS_ENTRY_ORDER_FAIL
            return self.status
 
        ###
        
        if self.status == STATUS_ENTRIED:
            self.count += 1
            if self.count < self.horizon:
                return self.status
            else:
                self.status = STATUS_EXIT_ORDER
            return self.status
                
        if self.status == STATUS_EXIT_ORDER:
            if self.kind == c.LONG and self.canSell(high, price):
                self.time_close = time
                self.index_close = index
                self.price_close = price
                self.profit = self.price_close - self.price_open
                self.ror = self.profit / self.price_open
                self.status = STATUS_DONE
            elif self.kind == c.SHORT and self.canBuy(low, price):
                self.time_close = time
                self.index_close = index
                self.price_close = price
                self.profit = self.price_open - self.price_close
                self.ror = self.profit / self.price_open
                self.status = STATUS_DONE
            return self.status    
    
    def canBuy(self, low, price):
        return price >= low

    def canSell(self, high, price):
        return price <= high
    
    def desc(self):
        s = 'ID: ' + str(self.idno)
        if self.kind == c.LONG:
            s += ' kind: Long '
        else:
            s += ' kind: Short '
        s += ' status: ' + str(self.status)
        s += ' open: ' + str(self.price_open)  + ' @' +  str(self.time_open) 
        s += ' close: ' + str(self.price_close)  + ' @' +  str(self.time_close) 
        print(s)

class AtrAlternate:
    def __init__(self, coeff, horizon, indicators):
        self.coeff = coeff
        self.horizon = horizon
        self.indicators = indicators
        self.positions = []
        self.finished_positions = []
        self.time = []
        self.open = []
        self.high = []
        self.low = []
        self.close = []
        self.atr = []
        self.current_position_id = 0
        ind = self.indicators[0]
        window = ind.params[prm.WINDOW]
        self.s_atr = S_ATR(window)
        
    def idno(self):
        self.current_position_id += 1
        return self.current_position_id
        
            
    def sliceDic(self, data:dict, begin, stop):
        time = data[c.TIME]
        n = len(time)
        if begin < 0:
            return None
        if stop > n:
            return None
    
        dic = {}
        dic[c.TIME] = data[c.TIME][begin:stop]
        dic[c.OPEN] = data[c.OPEN][begin:stop]
        dic[c.HIGH] = data[c.HIGH][begin:stop]
        dic[c.LOW] = data[c.LOW][begin:stop]
        dic[c.CLOSE] = data[c.CLOSE][begin:stop]
        return dic       
                
    def updateAtr(self, high, low, close):
        v = self.s_atr.update(high, low, close)
        self.atr.append(v)
        
    def updatePosition(self, index):
        if len(self.time) < 2:
            return
        
        time = self.time[-1]
        high = self.high[-1]
        low = self.low[-1]
        close = self.close[-1]
        price = self.atr[-2]
        
        if np.isnan(price):
            positions = []
            for pos in self.positions:
                if pos.status != STATUS_ENTRY_ORDER:
                    positions.append(pos)
            self.positions = positions
            return
        
        upper = int(self.close[-2] + self.coeff * price + 0.5)
        lower = int(self.close[-2] - self.coeff * price + 0.5)
        
        positions = []
        for i in range(len(self.positions)):
            pos = self.positions[i]
                
            if pos.status == STATUS_ENTRIED:
                status = pos.update(index, time, high, low, 0)
                if status == STATUS_ENTRIED:
                    positions.append(pos)
                    continue

            if pos.status == STATUS_ENTRY_ORDER:
                if pos.kind == c.LONG:
                    price = lower
                else:
                    price = upper
            elif pos.status == STATUS_EXIT_ORDER:
                if pos.kind == c.LONG:
                    price = upper
                else:
                    price = lower
            
            status = pos.update(index, time, high, low, price)
            if status == STATUS_ENTRY_ORDER_FAIL:
                continue
            
            #pos.desc()
            if status == STATUS_DONE:
                self.finished_positions.append(pos)
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
        self.updateAtr(hi, lo, cl)
        
        if len(self.time) < 2:
            return
        self.updatePosition(index)
        
        # Order
        pos1 = ATRAleternatePosition(self.idno(), c.LONG, self.horizon)
        self.positions.append(pos1)
        pos2 = ATRAleternatePosition(self.idno(), c.SHORT, self.horizon)
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
    
    def summary(self, tohlcv:dict):
        long_ror = []
        short_ror = []
        long_profit = []
        short_profit = []
        for pos in self.finished_positions:
            if pos.kind == c.LONG:
                long_ror.append(pos.ror)
                long_profit.append(pos.profit)
            else:
                short_ror.append(pos.ror)
                short_profit.append(pos.profit)
        
        result = {}
        result['long_ror_num'] = len(long_ror)
        result['long_ror_min'] = np.min(long_ror)
        result['long_ror_max'] = np.max(long_ror)
        result['long_ror_mean'] = np.mean(long_ror)
        result['short_ror_num'] = len(short_ror)
        result['short_ror_min'] = np.min(short_ror)
        result['short_ror_max'] = np.max(short_ror)
        result['short_ror_mean'] = np.mean(short_ror)
        result['long_profit_max'] = np.max(tohlcv['long_profit_acc'])
        result['long_draw_down'] = np.min(tohlcv['long_profit_acc'])
        result['short_profit_max'] = np.max(tohlcv['short_profit_acc'])
        result['short_draw_down'] = np.min(tohlcv['short_profit_acc'])        
        
        
        return result
    
    def acc(self, array):
        s = 0.0
        out = []
        for a in array:
            s += a
            out.append(s)
        return out
    
    def position2array(self, kind, length, key:str):
        value = np.zeros(length)
        for position in self.finished_positions:
            if position.kind == kind:
                if key == 'price_open':
                    value[position.index_open] = position.price_open
                elif key == 'price_close':
                    value[position.index_open] = position.price_close
                elif key == 'profit':
                    value[position.index_open] = position.profit
                elif key == 'ror':
                    value[position.index_open] = position.ror
        return value
    
    def makeTradeData(self, tohlcv:dict):
        tohlcv['ATR'] = self.atr
        time = tohlcv[c.TIME]
        n = len(time)
        tohlcv['long_price_open'] = self.position2array(c.LONG, n, 'price_open')
        tohlcv['long_price_close'] = self.position2array(c.LONG, n, 'price_close')
        tohlcv['short_price_open'] = self.position2array(c.SHORT, n, 'price_open')
        tohlcv['short_price_close'] = self.position2array(c.SHORT, n, 'price_close')
        tohlcv['long_profit'] = self.position2array(c.LONG, n, 'profit')
        tohlcv['short_profit'] = self.position2array(c.SHORT, n, 'profit')
        tohlcv['long_ror'] = self.position2array(c.LONG, n, 'ror')
        tohlcv['short_ror'] = self.position2array(c.SHORT, n, 'ror')

        tohlcv['long_profit_acc'] = self.acc(tohlcv['long_profit'])
        tohlcv['short_profit_acc'] = self.acc(tohlcv['short_profit'])
        tohlcv['long_ror_acc'] = self.acc(tohlcv['long_ror'])
        tohlcv['short_ror_acc'] = self.acc(tohlcv['short_ror'])


    
    def simulateEveryBar(self, tohlcv:dict):
        time = tohlcv[c.TIME]
        open = tohlcv[c.OPEN]
        high = tohlcv[c.HIGH]
        low = tohlcv[c.LOW]
        close = tohlcv[c.CLOSE]
        
        for i, tohlc in enumerate(zip(time, open, high, low, close)):
            self.update(i, tohlc)
            
        #print('Total Trade num:', len(self.finished_positions), ' not closed num: ', len(self.positions))
        self.makeTradeData(tohlcv)
        return self.summary(tohlcv)
    
    def allPosition(self):
        return self.positions

def test():
    return
    
    

if __name__ == '__main__':
    test()