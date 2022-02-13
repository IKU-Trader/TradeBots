# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 12:30:54 2022

@author: docs9
"""

import numpy as np
from datetime import datetime
import pytz
jp = pytz.timezone('Asia/Tokyo')

from DataServer import BitflyData
from StubAPI import StubAPI

from const import BasicConst, IndicatorConst, ParameterConst
from TechnicalAnalysis import Indicator
from CandlePlot import CandlePlot, BandPlot, makeFig, gridFig

c = BasicConst()
t = IndicatorConst()
p = ParameterConst()

indicator_list = {'MA5': {t.TYPE:t.SMA, p.WINDOW:5},
            'MA20': {t.TYPE:t.SMA, p.WINDOW:20},
            'MA60': {t.TYPE:t.SMA, p.WINDOW:60},
            'MA100': {t.TYPE:t.SMA, p.WINDOW:100},
            'MA200': {t.TYPE:t.SMA, p.WINDOW:200},
            'BBRATIO': {t.TYPE:t.BBRATIO, p.WINDOW: 20, p.SIGMA: 2.0},
            'VQ': {t.TYPE:t.VQ}}
    
class DataBuffer:
    
    @classmethod
    def array2graphShape(cls, data:dict, keys):
        if len(keys) == 1:
            return data[keys[0]]
        arrays = []
        for key in keys:
            arrays.append(data[key])
        n = len(arrays[0])
        out = []
        for i in range(n):
            v = []
            for array in arrays:
                v.append(array[i])
            out.append(v)
        return out
    
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
    
    def createModel(self):    
        return

def step():
    server = BitflyData('bitfly', 'M15')
    server.loadFromCsv('./data/bitflyer_btcjpy_m15.csv') 
    api = StubAPI(server)
    data = api.initialData(0)
    if data is None:
        return
    
    indicators = Indicator.makeIndicators(indicator_list)
    buffer = DataBuffer(indicators)
    buffer.loadData(data)
    
    print("last: ", buffer.lastTime())
    
    d = api.nextData()
    while d is not None:
        #print(d[c.TIME])
        buffer.addData(d)
        d = api.nextData()        


    
def test():
    server = BitflyData('bitfly', 'M15')
    server.loadFromCsv('./data/bitflyer_btcjpy_m15.csv') 
    api = StubAPI(server)
    t = jp.localize(datetime(2020, 7, 5, 7, 0))
    data = api.server.dataFrom(t,  2 * 4 * 24)
    indicators = Indicator.makeIndicators(indicator_list)
    buffer = DataBuffer(indicators)
    buffer.loadData(data)
    print('len: ', buffer.length(), buffer.technical.keys())
    
    time = buffer.array2graphShape(data, [c.TIME])
    ohlc = buffer.array2graphShape(data, [c.OPEN, c.HIGH, c.LOW, c.CLOSE])

    (fig, axes) = gridFig([5, 4, 4], (15, 15))
    graph1 = CandlePlot(fig, axes[0], 'btcjpy')
    graph1.xlimit((time[0], time[-1]))
    graph1.drawCandle(time, ohlc, timerange=(time[0], time[-1]))
    
    ma5 = buffer.technical['MA20']
    
    bbr = buffer.technical['BBRATIO']
    graph2 = BandPlot(fig, axes[1], 'bbratio')
    graph2.xlimit((time[0], time[-1]))
    graph2.drawLine(time, bbr, color='red', timerange=(time[0], time[-1]))
    
    vq = buffer.technical['VQ']
    graph3 = BandPlot(fig, axes[2], 'vq')
    graph3.drawLine(time, vq, color='blue', timerange=(time[0], time[-1]))
    graph3.xlimit((time[0], time[-1]))
    
    
    
    
if __name__ == '__main__':
    test()