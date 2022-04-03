# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 21:44:01 2022

@author: docs9
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), './'))

import pandas as pd
from datetime import datetime, timedelta, timezone
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
        
    def title(self):
        return self.name + '-' + self.interval
        
    # begin, end : index
    def sliceData(self, begin, end):
        out = {}
        for key, value in self.data.items():
            out[key] = value[begin:end]
        return out

    # rng: (begin, end)
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

class GemforexData(DataServer):
    
    def lastDay(self, year, month, weekday):
        t = datetime(year, month, 1) + timedelta(month=1)
        for i in range(15):
            t -= timedelta(days=1)
            if t.weekday() == weekday:
                return t
        return None
    
    def weekday(self, year, month, weekday, n):
        t = datetime(year, month, 1)
        count = 0
        for i in range(31):
            if t.weekday() == weekday:
                count += 1
                if count == n:
                    return t
            t += timedelta(days=1)
        return None
        
    def isSummerTime(self, date_str):
        date = datetime.strptime(date_str, '%Y.%m.%d')
        spring = self.weekday(date.year, 3, 6, 2)
        autumn = self.weekday(date.year, 10, 6, 2)
        if date > spring and date < autumn :
            return True
        else:
            return False
        
    def str2pyTime(self, date_str, time_str):
        jst = timedelta(hours=9)
        if self.isSummerTime(date_str):
            zone =  timedelta(hours=2)
            dt = jst - zone
        else:
            zone = timedelta(hours=3)
            dt = jst - zone
        s = date_str + ' ' + time_str
        t1 = datetime.strptime(s, '%Y.%m.%d %H:%M:%S') + dt
        t2 = t1.astimezone(timezone(jst, name='JST'))
        return t2
        
    def loadFromCsv(self, filepath):
        f = open(filepath)
        l = f.readline()
        l = f.readline()
        data = []
        for i in range(7):
            d = []
            data.append(d)
        while(l):
            values = l.split('\t')
            if len(values) != 9:
                l = f.readline()
                continue
            date = values[0]
            time = values[1]
            data[0].append(self.str2pyTime(date, time))
            opn = float(values[2])
            data[1].append(opn)
            high = float(values[3])
            data[2].append(high)
            low = float(values[4])
            data[3].append(low)
            clse = float(values[5])
            data[4].append(clse)
            volume = float(values[6])
            data[5].append(volume)
            spread = float(values[8])
            data[6].append(spread)
            l = f.readline()
        f.close()
        
        dic = {}
        dic[c.TIME] = data[0]
        dic[c.OPEN] = data[1]
        dic[c.HIGH] = data[2]
        dic[c.LOW] = data[3]
        dic[c.CLOSE] = data[4]
        dic[c.VOLUME] = data[5]
        dic[c.SPREAD] = data[6]
        self.data = dic
        return dic

def test():
    data = BitflyData('bitfly', 'M15')
    data.loadFromCsv()
    
    
    return


if __name__ == '__main__':
    test()