

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 17:37:53 2022

@author: docs9
"""
import numpy as np


class SMA:
    def __init__(self, window: int):
        self.window = window

    def calc(self, values: list):
        n = len(values)
        out = []
        for i in range(n):
            if i < self.window - 1:
                out.append(np.nan)
            else:
                d = values[i - self.window + 1: i + 1]
                out.append(np.mean(d))
        return out
            
            
class S_SMA:
    def __init__(self, window: int):
        self.window = window
        self.values = []
                
    def update(self, data: float):
        self.values.append(data)
        n = len(self.values)
        if n == self.window:
            ma = np.mean(self.values)
            self.values = self.values[1:]
            return ma
        else:
            return np.nan
        
class TR:
    def __init__(self):
        pass
        
    def calc(self, high: list, low: list, close: list):
        n = len(high)
        out = [high[0] - low[0]]
        for i in range(1, n):
            r = []
            r.append(high[i] - low[i])
            r.append(np.abs(high[i] - close[i - 1]))
            r.append(np.abs(close[i - 1] - low)[i])
            tr = np.max(r)
            out.append(tr)
        return out        
    
class S_TR:
    def __init__(self, window: int):
        self.window = window
        self.last_high = None
        self.last_close = None
                
    def update(self, high: float, low: float, close: float):
        if self.last_high is None:
            tr = high - low
        else:
            r = []
            r.append(high - low)
            r.append(np.abs(high - self.last_close))
            r.append(np.abs(self.last_close - low))
            tr = np.max(r)
        self.last_high = high
        self.last_close = close
        return tr
  
class ATR:
    def __init__(self, window: int):
        self.window = window
                
    def calc(self, high: list, low: list, close: list):
        tr = TR()
        sma = SMA(self.window)
        vector = tr.calc(high, low, close)
        return sma.calc(vector)
    
class S_ATR:
    def __init__(self, window: int):
        self.window = window
        self.s_tr = S_TR(window)
        self.s_sma = S_SMA(window)
        self.last_atr = np.nan
        
    def update(self, high: float, low: float, close: float):
        tr = self.s_tr.update(high, low, close)
        atr = self.s_sma.update(tr)
        return atr
    
    def update_old(self, high: float, low: float, close: float):
        tr = self.s_tr.update(high, low, close)
        if np.isnan(self.last_atr):
            atr = self.s_sma.update(tr)
        else:
            atr = (self.last_atr * (self.window - 1) + tr) / self.window
        self.last_atr = atr
        return atr
    
