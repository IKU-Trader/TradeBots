# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 21:44:01 2022

@author: docs9
"""
import pandas as pd
from datetime import datetime
from const import BasicConst 
from pandas_utility import df2dic

c = BasicConst()


def timestamp2pydatetime(array):
    out = []
    for a in array:
        out.append(a.to_pydatetime())
    return out

def datetime64pydatetime(array):
    out = []
    for a in array:
        out.append(a.astype(datetime))
    return out

def string2pydatetime(array:list, form='%Y-%m-%d %H:%M:%S%z', localize=True):
    out = []
    for s in array:
        t = datetime.strptime(s, form)
        if localize:
            t = t.astimezone()
        out.append(t)
    return out    
    
    
class DataServer:
    def __init__(self, name, interval):
        self.name = name
        self.interval = interval
    
class BitflyData(DataServer):
    
    def loadFromCsv(self, filepath):
        df = pd.read_csv(filepath)
        self.length = len(df)
        t = df['timestamp'].values
        time = string2pydatetime(t)
        df1 = df[['op', 'hi', 'lo', 'cl', 'volume']]
        keys={'op':c.OPEN, 'hi':c.HIGH, 'lo':c.LOW, 'cl':c.CLOSE}
        data = df2dic(df1, convert_keys=keys)
        data[c.TIME] = time
        self.data = data

    def sliceData(self, begin, end):
        out = {}
        for key, value in self.data.items():
            out[key] = value[begin:end]
        return out

    def dataRange(self, rng: range):
        begin = rng[0]
        end = rng[1]
        if begin < 0 or begin >= self.length:
            return None
        if end < 0 or end >= self.length:
            return None
        return self.sliceData(begin, end)
        
    def dataAll(self):
        return self.data
    
    def dataFrom(self, time_from, size):
        time = self.data[c.TIME]
        n = len(time)
        begin = None
        for i in range(n):
            if time[i] >= time_from:
                begin = i
                break
        if begin is None:
            return None
        end = begin + size 
        if end >= n:
            end = n - 1
        return self.sliceData(begin, end)
    
    def dataFromTo(self, time_from, time_to):
        time = self.data[c.TIME]
        n = len(time)
        begin = None
        for i in range(n):
            if time[i] >= time_from:
                begin = i
                break
        if begin is None:
            return None
        end = None
        for i in range(begin, n):
            if time[i] <= time_to:
                end = i
        if end is None:
            return None
        return self.sliceData(begin, end)

def test():
    data = BitflyData('bitfly', 'M15')
    data.loadFromCsv()
    
    
    return


if __name__ == '__main__':
    test()