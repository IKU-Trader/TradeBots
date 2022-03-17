# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 12:30:54 2022

@author: docs9
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
jp = pytz.timezone('Asia/Tokyo')

from DataServer import BitflyData
from StubAPI import StubAPI

from const import BasicConst, IndicatorConst, ParameterConst
from TechnicalAnalysis import Indicator
from CandlePlot import CandlePlot, BandPlot, makeFig, gridFig, array2graphShape
from Strategy import AtrAlternate

c = BasicConst()
ind = IndicatorConst()
p = ParameterConst()

indicator_list = {'MA5': {ind.TYPE:ind.SMA, p.WINDOW:5},
            'MA20': {ind.TYPE:ind.SMA, p.WINDOW:20},
            'MA60': {ind.TYPE:ind.SMA, p.WINDOW:60},
            'MA100': {ind.TYPE:ind.SMA, p.WINDOW:100},
            'MA200': {ind.TYPE:ind.SMA, p.WINDOW:200},
            'BBRATIO': {ind.TYPE:ind.BB_RATIO, p.WINDOW: 20, p.SIGMA: 2.0},
            'VQ': {ind.TYPE:ind.VQ}}



        
            
    
      
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


def save(filepath, data:dict, columns):
    time = data[c.TIME]
    n = len(time)
    arrays = []
    for column in columns:
        arrays.append(data[column])
        
    out = []
    for i in range(n):
        d = []
        for j in range(len(columns)):
            d.append(arrays[j][i])
        out.append(d)
    
    df = pd.DataFrame(data=out, columns=columns)
    df.to_csv(filepath, index=False)
    
    

def trade():
    indicator_param = {'ATR': {ind.TYPE:ind.ATR, p.WINDOW:20}}
    indicators = Indicator.makeIndicators(indicator_param)
    server = BitflyData('bitfly', 'M15')
    server.loadFromCsv('../data/bitflyer_btcjpy_m15.csv') 
    api = StubAPI(server)    
    buffer = DataBuffer(indicators)    
    buffer.loadData(api.allData())
    print('len: ', buffer.length(), buffer.technical.keys())
    print('from:', buffer.fromTime(), 'to: ', buffer.lastTime())
    tohlcv, technical_data = buffer.dataByDate(2020, 7, 5)
    

    atr = technical_data[ind.ATR]
    atralt = AtrAlternate(0.5, 1)

    

    time = array2graphShape(tohlcv, [c.TIME])
    (fig, axes) = gridFig([4, 2, 1], (15, 8))
    fig.subplots_adjust(hspace=0.6, wspace=0.4)
    graph1 = CandlePlot(fig, axes[0], 'btcjpy')
    keys = [c.OPEN, c.HIGH, c.LOW, c.CLOSE]
    graph1.drawCandle(time, tohlcv, keys)
    #graph1.drawLine(time, long_price)
    #graph1.drawLine(time, short_price, color='blue')
    
   
    graph2 = CandlePlot(fig, axes[1], 'ATR')
    graph2.drawLine(time, atr, color='red')
    
    atralt.simulateEveryBar(tohlcv, atr)
    save('./trade.csv', tohlcv, [c.TIME, c.OPEN, c.HIGH, c.LOW, c.CLOSE, ind.ATR, 'long_price_open', 'long_price_close', 'short_price_open', 'short_price_close'])



    #graph3 = CandlePlot(fig, axes[2], 'Profit')
    #graph3.drawLine(time, long_profit_acc, color='red')
    
   
def test():
    server = BitflyData('bitfly', 'M15')
    server.loadFromCsv('../data/bitflyer_btcjpy_m15.csv') 
    api = StubAPI(server)
    t = jp.localize(datetime(2020, 7, 5, 7, 0))
    data = api.server.dataFrom(t,  2 * 4 * 24)
    indicators = Indicator.makeIndicators(indicator_list)
    buffer = DataBuffer(indicators)
    buffer.loadData(data)
    print('len: ', buffer.length(), buffer.technical.keys())


    
    time = array2graphShape(data, [c.TIME])
    (fig, axes) = gridFig([5, 4, 4], (15, 15))
    fig.subplots_adjust(hspace=0.6, wspace=0.4)
    graph1 = CandlePlot(fig, axes[0], 'btcjpy')
    keys =  [c.OPEN, c.HIGH, c.LOW, c.CLOSE]
    graph1.drawCandle(time, data, keys)

    ma5 = buffer.technical['MA20']
    
    bbr = buffer.technical['BBRATIO']
    graph2 = CandlePlot(fig, axes[1], 'bbratio')
    graph2.drawLine(time, bbr, color='red')
    
    vq = buffer.technical['VQ']
    graph3 = CandlePlot(fig, axes[2], 'vq')
    graph3.drawLine(time, vq, color='blue')

    
    
    
if __name__ == '__main__':
    trade()