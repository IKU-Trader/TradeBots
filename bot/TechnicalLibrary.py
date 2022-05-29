

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 17:37:53 2022

@author: docs9
"""
import numpy as np
import talib as ta

class S_ATR:
    def __init__(self, window):
        self.high = []
        self.low = []
        self.close = []
        self.window = window
        return
    
    
    def update(self, h, l, c):
        self.high.append(h)
        self.low.append(l)
        self.close.append(c)
        n = len(self.high)
        if n > self.window:
            atr = ta.ATR(np.array(self.high, dtype='float'), np.array(self.low, dtype='float'), np.array(self.close, dtype='float'), timeperiod=self.window)
            if n > 500:
                self.high = self.high[1:]
                self.low = self.low[1:]
                self.close = self.close[1:]
            return atr[-1]
        else:
            return np.nan
    
