# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 14:20:18 2022

@author: docs9
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from const import BasicConst, IndicatorConst, ParameterConst
import talib as ta

c = BasicConst()
t = IndicatorConst()
p = ParameterConst()

    
def ADX(ohlcv:dict, window):
    return ta.ADX(ohlcv[c.HIGH], ohlcv[c.LOW], ohlcv[c.CLOSE], timeperiod=window)

def ADXDMI(ohlcv: dict, adx_window, ma_window):
    adx = ADX(ohlcv, adx_window)
    ma = movingAverage(adx, ma_window, fill_zero=False)
    return ma

def ADXR(ohlcv:dict, window):
    return ta.ADXR(ohlcv[c.HIGH], ohlcv[c.LOW], ohlcv[c.CLOSE], timeperiod=window)

def APO(ohlcv:dict, fast, slow):
    return ta.APO(ohlcv[c.CLOSE], fastperiod=fast, slowperiod=slow, matype=0)

def AROON(ohlcv:dict, window):
    return ta.AROON(ohlcv[c.HIGH], ohlcv[c.LOW], timeperiod=window)

def AROONOSC(ohlcv:dict, window):
    return ta.AROONOSC(ohlcv[c.HIGH], ohlcv[c.LOW], timeperiod=window)

def ATR(ohlcv:dict, window):
    return ta.ATR(ohlcv[c.HIGH], ohlcv[c.LOW], ohlcv[c.CLOSE], timeperiod=window)  

def BB(ohlcv: dict, window, sigma, key=c.CLOSE):
    (up, mid, low) = ta.BBANDS(ohlcv[key], timeperiod=window, nbdevup=sigma, nbdevdn=sigma, matype=0)
    return (up, mid, low)

def BBRATIO(ohlcv: dict, window, sigma, key=c.CLOSE):
    price = ohlcv[key]
    (up, mid, low) = ta.BBANDS(price, timeperiod=window, nbdevup=sigma, nbdevdn=sigma, matype=0)
    width = up - low
    level = price - low
    ratio = level / width
    return ratio

def BETA(ohlcv: dict, window):
    return ta.BETA(ohlcv[c.HIHG], ohlcv[c.LOW], timeperiod=window)

def CCI(ohlcv:dict, window):
    return ta.CCI(ohlcv[c.HIGH], ohlcv[c.LOW], ohlcv[c.CLOSE], timeperiod=window)

def DEMA(ohlcv:dict, window):
    return ta.DEMA(ohlcv[c.close], timeperiod=window)

def DI_PLUS(ohlcv:dict, window):
    return ta.PLUS_DI(ohlcv[c.HIGH], ohlcv[c.LOW], ohlcv[c.CLOSE], timeperiod=window)
    
def DI_MINUS(ohlcv:dict, window):
    return ta.MINUS_DI(ohlcv[c.HIGH], ohlcv[c.LOW], ohlcv[c.CLOSE], timeperiod=window)
    
def DX(ohlcv:dict, window):
    return ta.DX(ohlcv[c.HIGH], ohlcv[c.LOW], ohlcv[c.CLOSE], timeperiod=window)

def EMA(ohlcv: dict, window, key=c.CLOSE):
    return ta.EMA(ohlcv[key], timeperiod=window)  

def HT_DC_PERIOD(ohlcv: dict):
    return ta.HT_DCPERIOD(ohlcv[c.CLOSE])

def HT_DC_PHASE(ohlcv: dict):
    return ta.HT_DCPHASE(ohlcv[c.CLOSE])

def HT_PHASOR(ohlcv: dict):
    return ta.HT_PHASOR(ohlcv[c.CLOSE])

def HT_TRENDMODE(ohlcv: dict):
    return ta.HT_TRENDMODE(ohlcv[c.CLOSE])

def HT_TRENDLINE(ohlcv: dict):
    return ta.HT_TRENDLINE(ohlcv[c.CLOSE])

def KAMA(ohlcv: dict, window):
    return ta.KAMA(ohlcv[c.CLOSE], timeperiod=window)

def LINEARREG(ohlcv: dict, window):
    return ta.LINEARREG(ohlcv[c.CLOSE], timeperiod=window) - ohlcv[c.CLOSE]

def LINEARREG_ANGLE(ohlcv: dict, window):
    return ta.LINEARREG_ANGLE(ohlcv[c.CLOSE], timeperiod=window)

def LINEARREG_INTERCEPT(ohlcv: dict, window):
    return ta.LINEARREG_INTERCEPT(ohlcv[c.CLOSE], timeperiod=window)

def LINEARREG_SLOPE(ohlcv: dict, window):
    return ta.LINEARREG_SLOPE(ohlcv[c.CLOSE], timeperiod=window)

def MACD(ohlcv:dict, fast, slow, signal):
    return ta.MACD(ohlcv[c.CLOSE], fastperiod=fast, slowperiod=slow, signalperiod=signal)

def MFI(ohlcv:dict, window):
    return ta.MFI(ohlcv[c.HIGH], ohlcv[c.LOW], ohlcv[c.CLOSE], ohlcv[c.VOLUME], timeperiod=window)

def MIDPOINT(ohlcv:dict, window):
    return ta.MIDPOINT(ohlcv[c.CLOSE], timeperiod=window)

def MOM(ohlcv:dict, window):
    return ta.MOM(ohlcv[c.CLOSE], timeperiod=window)

def RCI(ohlcv:dict, window):
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

def ROC(ohlcv:dict, window):
    return ta.ROC(ohlcv[c.CLOSE], timeperiod=window)

def RSI(ohlcv:dict, window):
    return ta.RSI(ohlcv[c.CLOSE], timeperiod=window)

def SAR(ohlcv:dict, acc, maxv):
    return ta.SAR(ohlcv[c.HIGH], ohlcv[c.LOW], acceleration=acc, maximum=maxv)

def SMA(ohlcv: dict, window, key=c.CLOSE):
    return ta.SMA(ohlcv[key], timeperiod=window)

def STOCHASTIC(ohlcv: dict, fastk, slowk, slowd):
    return ta.STOCH(ohlcv[c.HIGH], ohlcv[c.LOW], ohlcv[c.CLOSE], fastk_window = fastk, fastk_matype=0, slowk_window=slowk, slowk_matype=0, slowd_window=slowd, slowd_matype=0)

def STOCHASTICFAST(ohlcv: dict, fastk, fastd):
    return ta.STOCHF(ohlcv[c.HIGH], ohlcv[c.LOW], ohlcv[c.CLOSE], fastk_window = fastk, fastk_matype=0, fastd_window=fastd, fastd_matype=0)

def T3(ohlcv: dict, window, vfactor=0):
    return ta.T3(ohlcv[c.CLOSE], timeperiod=window, vfactor=vfactor) + (ohlcv[c.HIGH] + ohlcv[c.LOW]) / 2

def TEMA(ohlcv: dict, window):
    return ta.TEMA(ohlcv[c.CLOSE], timeperiod=window)
                    
def TRIMA(ohlcv: dict, window):
    return ta.TRIMA(ohlcv[c.CLOSE], timeperiod=window)
  
def TRIX(ohlcv: dict, window):
    return ta.TRIX(ohlcv[c.CLOSE], timeperiod=window)

def TRANGE(ohlcv:dict):
    return ta.TRANGE(ohlcv[c.HIGH], ohlcv[c.LOW], ohlcv[c.CLOSE])
    
def ULTOSC(ohlcv:dict, fast, mid, slow):
    return ta.ULTOSC(ohlcv[c.HIGH], ohlcv[c.LOW], ohlcv[c.CLOSE], timeperiod1=fast, timeperiod2=mid, timeperiod3=slow)

def VQ(ohlcv: dict):
    trange = TRANGE(ohlcv)
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

def WILLR(ohlcv: dict, window):
    return ta.WILLR(ohlcv[c.HIGH], ohlcv[c.LOW], ohlcv[c.CLOSE], timeperiod=window)



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
        if self.typ == t.SMA:
            return SMA(ohlcv, self.params[p.WINDOW])
        if self.typ == t.EMA:
            return EMA(ohlcv, self.params[p.WINDOW])
        if self.typ == t.BB_UP:
            (up, down, mid) = BB(ohlcv, self.params[p.WINDOW], self.params[p.SIGMA])
            return up
        if self.typ == t.BB_DOWN:
            (up, down, mid) = BB(ohlcv, self.params[p.WINDOW], self.params[p.SIGMA])
            return down
        if self.typ == t.BB_RATIO:
            return BBRATIO(ohlcv, self.params[p.WINDOW], self.params[p.SIGMA])
        if self.typ == t.ATR:
            return ATR(ohlcv, self.params[p.WINDOW])
        if self.typ == t.TRANGE:
            return TRANGE(ohlcv)
        if self.typ == t.ADX:
            return ADX(ohlcv, self.params[p.WINDOW])
        if self.typ == t.DI_PLUS:
            return DI_PLUS(ohlcv, self.params[p.WINDOW])
        if self.typ == t.DI_MINUS:
            return DI_MINUS(ohlcv, self.params[p.WINDOW])       
        if self.typ == t.ADXR:
            return ADXR(ohlcv)
        if self.typ == t.AROON_DOWN:
            aroon_down, aroon_up = AROON(ohlcv, self.params[p.WINDOW])
            return aroon_down
        if self.typ == t.AROON_UP:
            aroon_down, aroon_up = AROON(ohlcv, self.params[p.WINDOW])
            return aroon_up
        if self.typ == t.AROON_OSC:
            return AROONOSC(ohlcv, self.params[p.WINDOW])
        if self.typ == t.RSI:
            return RSI(ohlcv, self.params[p.WINDOW]) 
        if self.typ == t.MACD:
            macd, macd_signal, macd_hist =  MACD(ohlcv, self.params[p.FAST], self.params[p.SLOW], self.params[p.SINGAL])
            return macd
        if self.typ == t.MACD_SIGNAL:
            macd, macd_signal, macd_hist =  MACD(ohlcv, self.params[p.FAST], self.params[p.SLOW], self.params[p.SINGAL])
            return macd_signal
        if self.typ == t.MACD_HIST:
            macd, macd_signal, macd_hist =  MACD(ohlcv, self.params[p.FAST], self.params[p.SLOW], self.params[p.SINGAL])
            return macd_hist                
        if self.typ == t.ROC:
            return ROC(ohlcv, self.params[p.WINDOW])
        if self.typ == t.MOMENTUM:
            return MOM(ohlcv, self.parms[p.WINDOW])
        if self.typ == t.APO:
            return APO(ohlcv, self.params[p.FAST], self.params[p.SLOW])
        if self.typ == t.VQ:
            return VQ(ohlcv)
        if self.typ == t.SAR:
            return SAR(ohlcv, self.params[p.ACC], self.params[p.MAX])
        if self.typ == t.RCI:
            return RCI(ohlcv, self.params[p.WINDOW])
        if self.typ == t.CCI:
            return CCI(ohlcv, self.params[p.WINDOW])
        if self.typ == t.DX:
            return DX(ohlcv, self.params[p.WINDOW])
        if self.typ == t.MFI:
            return MFI(ohlcv, self.params[p.WINDOW])
        if self.typ == t.STOCHASTIC_SLOWD:
            slowk, slowd = STOCHASTIC(ohlcv, self.params[p.FASTK], self.params[p.SLOWK], self.params[p.SLOWD])
            return slowd
        if self.typ == t.STOCHASTIC_SLOWK:
            slowk, slowd = STOCHASTIC(ohlcv, self.params[p.FASTK], self.params[p.SLOWK], self.params[p.SLOWD])
            return slowk
        if self.typ == t.STOCHASTIC_FASTK:
            fastk, fastd = STOCHASTIC(ohlcv, self.params[p.FASTK], self.params[p.FASTD])
            return fastk
        if self.typ == t.STOCHASTIC_FASTD:
            fastk, fastd = STOCHASTICFAST(ohlcv, self.params[p.FASTK], self.params[p.FASTD])
            return fastd
        if self.typ == t.ULTOSC:
            return ULTOSC(ohlcv, self.params[p.FAST], self.params[p.MID], self.params[p.SLOW])
        if self.typ == t.WILLR:
            return WILLR(ohlcv, self.params[p.WINDOW])
        if self.typ == t.HT_DC_PERIOD:
            return HT_DC_PERIOD(ohlcv)
        if self.typ == t.HT_DC_PHASE:
            return HT_DC_PHASE(ohlcv)
        if self.typ == t.HT_PHASOR_INPHASE:
            inphase, quadrature = HT_PHASOR(ohlcv)
            return inphase
        if self.typ == t.HT_PHASOR_QuADRATURE:
            inphase, quadrature = HT_PHASOR(ohlcv)
            return quadrature
        if self.typ == t.HT_TRENDMODE:
            return HT_TRENDMODE(ohlcv)
        if self.typ == t.HT_TRENDLINE:
            return HT_TRENDLINE(ohlcv)
        if self.typ == t.BETA:
            return BETA(ohlcv)
        if self.typ == t.LINEARREG:
            return LINEARREG(ohlcv, self.params[p.WINDOW])
        if self.typ == t.LINEARREG_ANGLE:
            return LINEARREG_ANGLE(ohlcv, self.params[p.WINDOW])
        if self.typ == t.LINEARREG_INTERCPT:
            return LINEARREG_INTERCEPT(ohlcv, self.params[p.WINDOW])
        if self.typ == t.LINEARREG_SLOPE:
            return LINEARREG_SLOPE(ohlcv, self.params[p.WINDOW])
        if self.typ == t.DEMA:
            return DEMA(ohlcv, self.params[p.WINDOW])
        if self.typ == t.KAMA:
            return KAMA(ohlcv, self.params[p.WINDOW])
        if self.typ == t.MIDPOINT:
            return MIDPOINT(ohlcv, self.params[p.WINDOW])
        if self.typ == t.T3:
            return T3(ohlcv, self.params[p.WINDOW])
        if self.typ == t.TEMA:
            return TEMA(ohlcv, self.params[p.WINDOW])
        if self.typ == t.TRIMA:
            return TRIMA(ohlcv, self.params[p.WINDOW])
        if self.typ == t.TRIX:
            return TRIX(ohlcv, self.params[p.WINDOW])        
        
    def description(self):
        out = "[" + self.name + "] " + self.typ + "  "
        for key, value in self.params.item():
            out +=  (key + ":" + str(value) + " ")
        return out
        
        
    