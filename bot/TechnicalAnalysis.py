# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 14:20:18 2022

@author: docs9
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import rankdata
from const import BasicConst, IndicatorConst, ParameterConst
import talib as ta
from pandas_utility import df2dic
from TechnicalLibrary import S_ATR, ATR

import matplotlib.pyplot as plt

c = BasicConst()
t = IndicatorConst()
p = ParameterConst()

class TaLib:
    @classmethod
    def ADX(cls, ohlcv:dict, window):
        return ta.ADX(np.array(ohlcv[c.HIGH]), np.array(ohlcv[c.LOW]), np.array(ohlcv[c.CLOSE]), timeperiod=window)
    @classmethod
    def ADXDMI(cls, ohlcv: dict, adx_window, ma_window):
        adx = TaLib.ADX(ohlcv, adx_window)
        ma = movingAverage(adx, ma_window, fill_zero=False)
        return ma

    @classmethod
    def ADXR(cls, ohlcv:dict, window):
        return ta.ADXR(np.array(ohlcv[c.HIGH]), np.array(ohlcv[c.LOW]), np.array(ohlcv[c.CLOSE]), timeperiod=window)

    @classmethod
    def APO(cls, ohlcv:dict, fast, slow):
        return ta.APO(np.array(ohlcv[c.CLOSE]), fastperiod=fast, slowperiod=slow, matype=0)

    @classmethod
    def AROON(cls, ohlcv:dict, window):
        return ta.AROON(np.array(ohlcv[c.HIGH]), np.array(ohlcv[c.LOW]), timeperiod=window)

    @classmethod
    def AROONOSC(cls, ohlcv:dict, window):
        return ta.AROONOSC(np.array(ohlcv[c.HIGH]), np.array(ohlcv[c.LOW]), timeperiod=window)

    @classmethod
    def ATR(cls, ohlcv:dict, window):
        return ta.ATR(np.array(ohlcv[c.HIGH]), np.array(ohlcv[c.LOW]), np.array(ohlcv[c.CLOSE]), timeperiod=window)  

    @classmethod
    def BB(cls, ohlcv: dict, window, sigma, key=c.CLOSE):
        hilo =  (np.array(ohlcv[c.HIGH]) - np.array(ohlcv[c.LOW])) / 2.0
        (up, mid, low) = ta.BBANDS(np.array(ohlcv[key]), timeperiod=window, nbdevup=sigma, nbdevdn=sigma, matype=0)
        return (up - hilo, mid - hilo, low - hilo)

    @classmethod
    def BBRATIO(cls, ohlcv: dict, window, sigma, key=c.CLOSE):
        price = np.array(ohlcv[key])
        (up, mid, low) = ta.BBANDS(price, timeperiod=window, nbdevup=sigma, nbdevdn=sigma, matype=0)
        width = up - low
        level = price - low
        ratio = level / width
        return ratio

    @classmethod
    def BETA(cls, ohlcv: dict, window):
        return ta.BETA(np.array(ohlcv[c.HIGH]), np.array(ohlcv[c.LOW]), timeperiod=window)

    @classmethod
    def CCI(cls, ohlcv:dict, window):
        return ta.CCI(np.array(ohlcv[c.HIGH]), np.array(ohlcv[c.LOW]), np.array(ohlcv[c.CLOSE]), timeperiod=window)

    @classmethod
    def DEMA(cls, ohlcv:dict, window):
        hilo =  (np.array(ohlcv[c.HIGH]) - np.array(ohlcv[c.LOW])) / 2.0
        return ta.DEMA(np.array(ohlcv[c.CLOSE]), timeperiod=window) - hilo

    @classmethod
    def DI_PLUS(cls, ohlcv:dict, window):
        return ta.PLUS_DI(np.array(ohlcv[c.HIGH]), np.array(ohlcv[c.LOW]), np.array(ohlcv[c.CLOSE]), timeperiod=window)
    
    @classmethod
    def DI_MINUS(cls, ohlcv:dict, window):
        return ta.MINUS_DI(np.array(ohlcv[c.HIGH]), np.array(ohlcv[c.LOW]), np.array(ohlcv[c.CLOSE]), timeperiod=window)
    
    @classmethod
    def DX(cls, ohlcv:dict, window):
        return ta.DX(np.array(ohlcv[c.HIGH]), np.array(ohlcv[c.LOW]), np.array(ohlcv[c.CLOSE]), timeperiod=window)

    @classmethod
    def EMA(cls, ohlcv: dict, window, key=c.CLOSE):
        hilo =  (np.array(ohlcv[c.HIGH]) - np.array(ohlcv[c.LOW])) / 2.0
        return ta.EMA(np.array(ohlcv[key]), timeperiod=window) - hilo

    @classmethod
    def HT_DC_PERIOD(cls, ohlcv: dict):
        return ta.HT_DCPERIOD(np.array(ohlcv[c.CLOSE]))

    @classmethod
    def HT_DC_PHASE(cls, ohlcv: dict):
        return ta.HT_DCPHASE(np.array(ohlcv[c.CLOSE]))

    @classmethod
    def HT_PHASOR(cls, ohlcv: dict):
        return ta.HT_PHASOR(np.array(ohlcv[c.CLOSE]))

    @classmethod
    def HT_TRENDMODE(cls, ohlcv: dict):
        return ta.HT_TRENDMODE(np.array(ohlcv[c.CLOSE]))

    @classmethod
    def HT_TRENDLINE(cls, ohlcv: dict):
        hilo =  (np.array(ohlcv[c.HIGH]) - np.array(ohlcv[c.LOW])) / 2.0
        return ta.HT_TRENDLINE(np.array(ohlcv[c.CLOSE])) - hilo

    @classmethod
    def KAMA(cls, ohlcv: dict, window):
        hilo =  (np.array(ohlcv[c.HIGH]) - np.array(ohlcv[c.LOW])) / 2.0
        return ta.KAMA(np.array(ohlcv[c.CLOSE]), timeperiod=window) - hilo

    
    @classmethod
    def LINEARREG(cls, ohlcv: dict, window):
        return ta.LINEARREG(np.array(ohlcv[c.CLOSE]), timeperiod=window) - np.array(ohlcv[c.CLOSE])

    @classmethod
    def LINEARREG_ANGLE(cls, ohlcv: dict, window):
        return ta.LINEARREG_ANGLE(np.array(ohlcv[c.CLOSE]), timeperiod=window)

    @classmethod
    def LINEARREG_INTERCEPT(cls, ohlcv: dict, window):
        return ta.LINEARREG_INTERCEPT(np.array(ohlcv[c.CLOSE]), timeperiod=window)

    @classmethod
    def LINEARREG_SLOPE(cls, ohlcv: dict, window):
        return ta.LINEARREG_SLOPE(np.array(ohlcv[c.CLOSE]), timeperiod=window)

    @classmethod
    def MA(cls, ohlcv: dict, window, key=c.CLOSE):
        hilo =  (np.array(ohlcv[c.HIGH]) - np.array(ohlcv[c.LOW])) / 2.0
        return ta.MA(np.array(ohlcv[key]), timeperiod=window) - hilo
    
    @classmethod
    def MACD(cls, ohlcv:dict, fast, slow, signal):
        return ta.MACD(np.array(ohlcv[c.CLOSE]), fastperiod=fast, slowperiod=slow, signalperiod=signal)

    @classmethod
    def MFI(cls, ohlcv:dict, window):
        return ta.MFI(np.array(ohlcv[c.HIGH]), np.array(ohlcv[c.LOW]), np.array(ohlcv[c.CLOSE]), np.array(ohlcv[c.VOLUME]), timeperiod=window)

    @classmethod
    def MIDPOINT(cls, ohlcv:dict, window):
        hilo =  (np.array(ohlcv[c.HIGH]) - np.array(ohlcv[c.LOW])) / 2.0
        return ta.MIDPOINT(np.array(ohlcv[c.CLOSE]), timeperiod=window) - hilo

    @classmethod
    def MOM(cls, ohlcv:dict, window):
        return ta.MOM(np.array(ohlcv[c.CLOSE]), timeperiod=window)
    
    @classmethod
    def PPO(cls, ohlcv:dict, fast: int, slow: int):
        return ta.PPO(np.array(ohlcv[c.CLOSE]), fastperiod=fast, slowperiod=slow)

    @classmethod
    def RCI(cls, ohlcv:dict, window):
        close = ohlcv[c.CLOSE]
        n = len(close)
        out = np.full(n, np.nan)
        for i in range(n):
            if (i + window)> n:
                break
            y = close[i:i + window]
            x_rank = np.arange(len(y))
            y_rank = rankdata(y, method='ordinal') - 1
            sum_diff = sum((x_rank - y_rank)**2)
            rci = (1 - ((6 * sum_diff) / (window**3 - window))) * 100
            out.append(rci)
        return out

    @classmethod
    def ROC(cls, ohlcv:dict, window):
        return ta.ROC(np.array(ohlcv[c.CLOSE]), timeperiod=window)

    @classmethod
    def RSI(cls, ohlcv:dict, window):
        return ta.RSI(np.array(ohlcv[c.CLOSE]), timeperiod=window)

    @classmethod
    def SAR(cls, ohlcv:dict, acc, maxv):
        return ta.SAR(np.array(ohlcv[c.HIGH]), np.array(ohlcv[c.LOW]), acceleration=acc, maximum=maxv)

    @classmethod
    def SMA(cls, ohlcv: dict, window, key=c.CLOSE):
        hilo =  (np.array(ohlcv[c.HIGH]) - np.array(ohlcv[c.LOW])) / 2.0
        return ta.SMA(np.array(ohlcv[key]), timeperiod=window) - hilo

    @classmethod
    def STDDEV(cls, ohlcv: dict, window: int, sigma):
        return ta.STDDEV(np.array(ohlcv[c.CLOSE]), timeperiod=window, nbdev=sigma)        
    
    @classmethod
    def STOCHASTIC(cls, ohlcv: dict, fastk, slowk, slowd):
        return ta.STOCH(np.array(ohlcv[c.HIGH]), np.array(ohlcv[c.LOW]), np.array(ohlcv[c.CLOSE]), fastk_period = fastk, slowk_period=slowk, slowk_matype=0, slowd_period=slowd, slowd_matype=0)

    @classmethod
    def STOCHASTICFAST(cls, ohlcv: dict, fastk, fastd):
        return ta.STOCHF(np.array(ohlcv[c.HIGH]), np.array(ohlcv[c.LOW]), np.array(ohlcv[c.CLOSE]), fastk_period = fastk, fastd_period=fastd, fastd_matype=0)

    @classmethod
    def T3(cls, ohlcv: dict, window, vfactor=0):
        hilo =  (np.array(ohlcv[c.HIGH]) - np.array(ohlcv[c.LOW])) / 2.0
        return ta.T3(np.array(ohlcv[c.CLOSE]), timeperiod=window, vfactor=vfactor) - hilo

    @classmethod
    def TEMA(cls, ohlcv: dict, window):
        hilo =  (np.array(ohlcv[c.HIGH]) - np.array(ohlcv[c.LOW])) / 2.0
        return ta.TEMA(np.array(ohlcv[c.CLOSE]), timeperiod=window) - hilo
                    
    @classmethod
    def TRIMA(cls, ohlcv: dict, window):
        hilo =  (np.array(ohlcv[c.HIGH]) - np.array(ohlcv[c.LOW])) / 2.0
        return ta.TRIMA(np.array(ohlcv[c.CLOSE]), timeperiod=window) - hilo
  
    @classmethod
    def TRIX(cls, ohlcv: dict, window):
        return ta.TRIX(np.array(ohlcv[c.CLOSE]), timeperiod=window)

    @classmethod
    def TRANGE(cls, ohlcv:dict):
        return ta.TRANGE(np.array(ohlcv[c.HIGH]), np.array(ohlcv[c.LOW]), np.array(ohlcv[c.CLOSE]))
    
    @classmethod
    def ULTOSC(cls, ohlcv:dict, fast, mid, slow):
        return ta.ULTOSC(np.array(ohlcv[c.HIGH]), np.array(ohlcv[c.LOW]), np.array(ohlcv[c.CLOSE]), timeperiod1=fast, timeperiod2=mid, timeperiod3=slow)

    @classmethod
    def VQ(cls, ohlcv: dict):
        trange = TaLib.TRANGE(ohlcv)
        high = ohlcv[c.HIGH]
        low = ohlcv[c.LOW]
        open = ohlcv[c.OPEN]
        close = ohlcv[c.CLOSE]
        n = len(close)
        vo = np.zeros(n)
        q = np.zeros(n)
        for i in range(1, n):
            if trange[i] != 0 and (high[i] - low[i]) != 0:
                vo[i] = (close[i] - close[i - 1]) / trange[i] + (close[i] - open[i]) / (high[i] - low[i]) * 0.5
            else:
                vo[i] = vo[1]
        for i in range(1, n):
            q[i] = np.abs(vo[i]) * ((close[i] - close[i - 1]) + (close[i] -open[i])) * 0.5
            vq = np.zeros(n)
            s = q[0]
        for i in range(1, n):
            s += q[i]
            vq[i] = s
        return vq

    @classmethod
    def WILLR(cls, ohlcv: dict, window):
        return ta.WILLR(np.array(ohlcv[c.HIGH]), np.array(ohlcv[c.LOW]), np.array(ohlcv[c.CLOSE]), timeperiod=window)

    @classmethod
    def WMA(cls, ohlcv: dict, window):
        hilo =  (np.array(ohlcv[c.HIGH]) - np.array(ohlcv[c.LOW])) / 2.0
        return ta.WMA(np.array(ohlcv[c.HIGH]), timeperiod=window) - hilo

def fillZero(array):
    for i in range(len(array)):
        if np.isnan(array[i]):
            array[i] = 0
            
def movingAverage(array, window, fill_zero=True):
    d = pd.Series(array)
    d.rolling(window).mean()
    if fill_zero:
        fillZero(d)
    return d


class Indicator:
    
    @classmethod
    def makeIndicators(cls, indicators:dict):
        out = []
        for key, values in indicators.items():
            ind = Indicator(values[t.TYPE], key)
            for k, v in values.items():
                ind.param(k, v)
            out.append(ind)
        return out 
    
    def __init__(self, typ, name):
        self.typ = typ.strip().lower()
        self.name = name.strip()
        self.params = {}
        
    def param(self, key, value):
        self.params[key] = value
        
    def calc(self, ohlcv: dict):
        ret = self.calc2(ohlcv)
        ohlcv[self.name] = ret
        
    def calc2(self, ohlcv: dict):
        if self.typ == t.ROR:
            return self.ror(ohlcv)
        if self.typ == t.SMA:
            return TaLib.SMA(ohlcv, self.params[p.WINDOW])
        if self.typ == t.EMA:
            return TaLib.EMA(ohlcv, self.params[p.WINDOW])
        if self.typ == t.BB_UP:
            (up, down, mid) = TaLib.BB(ohlcv, self.params[p.WINDOW], self.params[p.SIGMA])
            return up
        if self.typ == t.BB_DOWN:
            (up, down, mid) = TaLib.BB(ohlcv, self.params[p.WINDOW], self.params[p.SIGMA])
            return down
        if self.typ == t.BB_MID:
            (up, down, mid) = TaLib.BB(ohlcv, self.params[p.WINDOW], self.params[p.SIGMA])
            return mid
        if self.typ == t.BB_RATIO:
            return TaLib.BBRATIO(ohlcv, self.params[p.WINDOW], self.params[p.SIGMA])
        if self.typ == t.ATR:
            return TaLib.ATR(ohlcv, self.params[p.WINDOW])
        if self.typ == t.TRANGE:
            return TaLib.TRANGE(ohlcv)
        if self.typ == t.ADX:
            return TaLib.ADX(ohlcv, self.params[p.WINDOW])
        if self.typ == t.DI_PLUS:
            return TaLib.DI_PLUS(ohlcv, self.params[p.WINDOW])
        if self.typ == t.DI_MINUS:
            return TaLib.DI_MINUS(ohlcv, self.params[p.WINDOW])       
        if self.typ == t.ADXR:
            return TaLib.ADXR(ohlcv, self.params[p.WINDOW])
        if self.typ == t.AROON_DOWN:
            aroon_down, aroon_up = TaLib.AROON(ohlcv, self.params[p.WINDOW])
            return aroon_down
        if self.typ == t.AROON_UP:
            aroon_down, aroon_up = TaLib.AROON(ohlcv, self.params[p.WINDOW])
            return aroon_up
        if self.typ == t.AROON_OSC:
            return TaLib.AROONOSC(ohlcv, self.params[p.WINDOW])
        if self.typ == t.RSI:
            return TaLib.RSI(ohlcv, self.params[p.WINDOW]) 
        if self.typ == t.MA:
            return TaLib.MA(ohlcv, self.params[p.WINDOW])
        if self.typ == t.MACD:
            macd, macd_signal, macd_hist =  TaLib.MACD(ohlcv, self.params[p.FAST], self.params[p.SLOW], self.params[p.SIGNAL])
            return macd
        if self.typ == t.MACD_SIGNAL:
            macd, macd_signal, macd_hist = TaLib.MACD(ohlcv, self.params[p.FAST], self.params[p.SLOW], self.params[p.SIGNAL])
            return macd_signal
        if self.typ == t.MACD_HIST:
            macd, macd_signal, macd_hist =  TaLib.MACD(ohlcv, self.params[p.FAST], self.params[p.SLOW], self.params[p.SIGNAL])
            return macd_hist                
        if self.typ == t.PPO:
            return TaLib.PPO(ohlcv, self.params[p.FAST], self.params[p.SLOW])
        if self.typ == t.ROC:
            return TaLib.ROC(ohlcv, self.params[p.WINDOW])
        if self.typ == t.MOMENTUM:
            return TaLib.MOM(ohlcv, self.params[p.WINDOW])
        if self.typ == t.APO:
            return TaLib.APO(ohlcv, self.params[p.FAST], self.params[p.SLOW])
        if self.typ == t.VQ:
            return TaLib.VQ(ohlcv)
        if self.typ == t.SAR:
            return TaLib.SAR(ohlcv, self.params[p.ACC], self.params[p.MAX])
        if self.typ == t.RCI:
            return TaLib.RCI(ohlcv, self.params[p.WINDOW])
        if self.typ == t.CCI:
            return TaLib.CCI(ohlcv, self.params[p.WINDOW])
        if self.typ == t.DX:
            return TaLib.DX(ohlcv, self.params[p.WINDOW])
        if self.typ == t.MFI:
            return TaLib.MFI(ohlcv, self.params[p.WINDOW])
        if self.typ == t.STDDEV:
            return TaLib.STDDEV(ohlcv, self.params[p.WINDOW], self.params[p.SIGMA])
        if self.typ == t.STOCHASTIC_SLOWD:
            slowk, slowd = TaLib.STOCHASTIC(ohlcv, self.params[p.FASTK], self.params[p.SLOWK], self.params[p.SLOWD])
            return slowd
        if self.typ == t.STOCHASTIC_SLOWK:
            slowk, slowd = TaLib.STOCHASTIC(ohlcv, self.params[p.FASTK], self.params[p.SLOWK], self.params[p.SLOWD])
            return slowk
        if self.typ == t.STOCHASTIC_FASTK:
            fastk, fastd = TaLib.STOCHASTICFAST(ohlcv, self.params[p.FASTK], self.params[p.FASTD])
            return fastk
        if self.typ == t.STOCHASTIC_FASTD:
            fastk, fastd = TaLib.STOCHASTICFAST(ohlcv, self.params[p.FASTK], self.params[p.FASTD])
            return fastd
        if self.typ == t.ULTOSC:
            return TaLib.ULTOSC(ohlcv, self.params[p.FAST], self.params[p.MID], self.params[p.SLOW])
        if self.typ == t.WILLR:
            return TaLib.WILLR(ohlcv, self.params[p.WINDOW])
        if self.typ == t.HT_DC_PERIOD:
            return TaLib.HT_DC_PERIOD(ohlcv)
        if self.typ == t.HT_DC_PHASE:
            return TaLib.HT_DC_PHASE(ohlcv)
        if self.typ == t.HT_PHASOR_INPHASE:
            inphase, quadrature = TaLib.HT_PHASOR(ohlcv)
            return inphase
        if self.typ == t.HT_PHASOR_QUADRATURE:
            inphase, quadrature = TaLib.HT_PHASOR(ohlcv)
            return quadrature
        if self.typ == t.HT_TRENDMODE:
            return TaLib.HT_TRENDMODE(ohlcv)
        if self.typ == t.HT_TRENDLINE:
            return TaLib.HT_TRENDLINE(ohlcv)
        if self.typ == t.BETA:
            return TaLib.BETA(ohlcv, self.params[p.WINDOW])
        if self.typ == t.LINEARREG:
            return TaLib.LINEARREG(ohlcv, self.params[p.WINDOW])
        if self.typ == t.LINEARREG_ANGLE:
            return TaLib.LINEARREG_ANGLE(ohlcv, self.params[p.WINDOW])
        if self.typ == t.LINEARREG_INTERCEPT:
            return TaLib.LINEARREG_INTERCEPT(ohlcv, self.params[p.WINDOW])
        if self.typ == t.LINEARREG_SLOPE:
            return TaLib.LINEARREG_SLOPE(ohlcv, self.params[p.WINDOW])
        if self.typ == t.DEMA:
            return TaLib.DEMA(ohlcv, self.params[p.WINDOW])
        if self.typ == t.KAMA:
            return TaLib.KAMA(ohlcv, self.params[p.WINDOW])
        if self.typ == t.MIDPOINT:
            return TaLib.MIDPOINT(ohlcv, self.params[p.WINDOW])
        if self.typ == t.T3:
            return TaLib.T3(ohlcv, self.params[p.WINDOW])
        if self.typ == t.TEMA:
            return TaLib.TEMA(ohlcv, self.params[p.WINDOW])
        if self.typ == t.TRIMA:
            return TaLib.TRIMA(ohlcv, self.params[p.WINDOW])
        if self.typ == t.TRIX:
            return TaLib.TRIX(ohlcv, self.params[p.WINDOW])  
        if self.typ == t.WMA:
            return TaLib.WMA(ohlcv, self.params[p.WINDOW])     
        if self.typ == t.WEEKDAY:
            return self.weekday(ohlcv)
        if self.typ == t.TIMEBAND:
            return self.timeband(ohlcv)
        if self.typ == t.CANDLE_BODY:
            return self.candleBody(ohlcv)
        if self.typ == t.SPIKE:
            return self.spike(ohlcv)
        
    def ror(self, tohlcv:dict):
        out = []
        out.append(np.nan)
        close = tohlcv[c.CLOSE]
        for i in range(1, len(close)):
            out.append((close[i] - close[i - 1])/ close[i - 1])
        return np.array(out)
    
    def weekday(self, tohlcv):
        time = tohlcv[c.TIME]
        out = []
        for t0 in time:
            t = t0 - timedelta(hours=7)
            out.append(float(t.weekday()))
        return np.array(out)
    
    def timeband(self, tohlcv):
        time = tohlcv[c.TIME]
        out = []
        for t0 in time:
            hour = t0.hour
            if hour < 7:
                b = 0
            elif hour < 15:
                b = 1
            elif hour < 17:
                b = 2
            elif hour < 21:
                b = 3
            else:
                b = 0
            out.append(float(b))
        return np.array(out)
    
    def candleBody(self, tohlcv):
        op = tohlcv[c.OPEN]
        close = tohlcv[c.CLOSE]
        out = []
        for o, cl in zip(op, close):
            out.append(float(cl - o) / float(o))
        return np.array(out)
    
    def spike(self, tohlcv):
        open = tohlcv[c.OPEN]
        high = tohlcv[c.HIGH]
        low = tohlcv[c.LOW]
        close = tohlcv[c.CLOSE]
        out = []
        for op, hi, lo, cl in zip(open, high, low, close):
            if (cl - op ) > 0:
                out.append(float(hi - cl) / float(cl))
            else:
                out.append(float(lo - cl) / float(cl))
        return np.array(out)        
        
    def description(self):
        out = "[" + self.name + "] " + self.typ + "  "
        for key, value in self.params.item():
            out +=  (key + ":" + str(value) + " ")
        return out
        
###
    
def string2pydatetime(array:list, form='%Y-%m-%d %H:%M:%S%z', localize=True):
    out = []
    for s in array:
        t = datetime.strptime(s, form)
        if localize:
            t = t.astimezone()
        out.append(t)
    return out 

def loadFromCsv(filepath):
    df = pd.read_csv(filepath)
    t = df['Time'].values
    tim = string2pydatetime(t, form='%d-%m-%Y')
    df1 = df[['Time', 'High','Low','Close']]
    keys={'Time':c.TIME, 'High':c.HIGH, 'Low':c.LOW, 'Close':c.CLOSE}
    data = df2dic(df1, convert_keys=keys)
    data[c.TIME] = tim
    return data
            
            
def sliceDic(data:dict, begin, stop):
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
    
    
def test1(data:dict):
    atr = TaLib.ATR(data, 14)
    return atr

def test2(data:dict):
    import ta.volatility as ta
    high = pd.Series(data=data[c.HIGH])
    low = pd.Series(data=data[c.LOW])
    close = pd.Series(data=data[c.CLOSE])
    atr = ta.AverageTrueRange(high, low, close, window=14, fillna=True)
    return atr.average_true_range() 
    
def test3(data:dict):
    high = pd.Series(data=data[c.HIGH])
    low = pd.Series(data=data[c.LOW])
    close = pd.Series(data=data[c.CLOSE])
    atr = ATR(14)
    vector = atr.calc(high, low, close)
    return vector 

def test4(data:dict):
    high = data[c.HIGH]
    low = data[c.LOW]
    close = data[c.CLOSE]
    out = []
    satr = S_ATR(14)
    for hi, lo, cl in zip(high, low, close):
        a = satr.update(hi, lo, cl)
        out.append(a)
    
    return out

if __name__ == '__main__':
    data = loadFromCsv('../data/test-atr.csv')
    atr_ref1 = test1(data)
    atr_ref2 = test2(data)
    atr1 = test3(data)
    atr2 = test4(data)
    
    t = np.arange(len(atr1))
    #plt.plot(t, atr_ref1, color='red')
    #plt.plot(t, atr_ref2, color='green')
    plt.plot(t, atr1, color='blue')
    plt.plot(t, atr2, color='yellow')
    
    print(atr1)