# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 17:50:29 2022

@author: docs9
"""
import numpy as np
from datetime import datetime, timedelta
import pytz
jp = pytz.timezone('Asia/Tokyo')
from const import BasicConst, IndicatorConst, ParameterConst

c = BasicConst()
ind = IndicatorConst()
p = ParameterConst()


class DataBuffer:
    def __init__(self, indicators: list):
        self.indicators = indicators
        self.tohlcv = []
                
    def loadData(self, tohlcv: dict):
        self.tohlcv = tohlcv
        self.calc()
        pass

    def addData(self, tohlcv: dict):
        for key, values in tohlcv.items():
            data = self.tohlcv[key]
            for v in values:
                data = np.append(data, v)
            self.tohlcv[key] = data
        self.calc()
        
    def fromTime(self):
        if len(self.tohlcv) == 0:
            return None
        time = self.tohlcv[c.TIME][0]
        return time
        
    def lastTime(self):
        if len(self.tohlcv) == 0:
            return None
        time = self.tohlcv[c.TIME][-1]
        return time

    def calc(self):
        technical = {}
        for indicator in self.indicators:
            technical[indicator.name] = indicator.calc(self.tohlcv)        
        self.technical = technical
        return technical
    
    def length(self):
       return len(self.tohlcv[c.TIME]) 
    
    def dict2list(self):
        out = []
        keys = self.tohlcv.keys()
        for key in keys:
            out.append(self.tohlcv[key])
        return out
    
    def dataSlice(self, data:dict, begin, end):
        out = {}
        for key in data:
            d = data[key]
            out[key] = d[begin: end + 1]
        return out
        
    
    def dataByDate(self, year:int, month: int, day: int, size=None):
        t0 = jp.localize(datetime(year, month, day))
        t1 = t0 + timedelta(days=1)
        time = self.tohlcv[c.TIME]
        count = 0
        begin = None
        end = None
        for key in self.tohlcv.keys():
            for i in range(len(time)):
                t = time[i]
                if t >= t0 and t < t1:
                    if begin is None:
                        begin = i
                    else:
                        end = i
                    count += 1
                    if size is not None:
                        if count >= size:
                            break
        if begin is None:
            return (None, None)
        return (self.dataSlice(self.tohlcv, begin, end), self.dataSlice(self.technical, begin, end))   
        