# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 09:49:46 2022

@author: docs9
"""


class BasicConst:
    def __init__(self):
        self.TIME = 'time'
        self.OPEN = 'open'
        self.HIGH = 'high'
        self.LOW = 'low'
        self.CLOSE = 'close'
        self.VOLUME = 'volume'
        self.BUY_VOL = 'buy_vol'
        self.SELL_VOL = 'sell_vol'
        
        self.LONG = 1
        self.SHORT = -1


class IndicatorConst:
# Techinical indicators
    def __init__(self):
        self.TYPE = 'type'
        self.SMA = 'sma'
        self.EMA = 'ema'
        self.ADX = 'adx'
        self.PLUS_DI = 'plus_di'
        self.MINUS_DI = 'minus_di'
        self.ADXR = 'adxr'
        self.BBUP = 'bbup'
        self.BBDOWN = 'bbdown'
        self.BBRATIO = 'bbratio'
        self.AROON = 'aroon'
        self.ATR = 'atr'
        self.TRANGE = 'trange'
        self.RSI = 'rsi'
        self.MACD = 'macd'
        self.ROC = 'roc'
        self.MOMENTUM = 'momentum'
        self.APO = 'apo'
        self.ADXDMI = 'adxdmi'
        self.VQ = 'vq'
        self.SAR = 'sar'
        self.RCI = 'rci'

class ParameterConst:
# parameters
    def __init__(self):
        self.WINDOW = 'window'
        self.SLOW = 'slow'
        self.FAST = 'fast'
        self.SIGNAL = 'signal'
        self.SIGMA = 'sigma'
        self.ACC = 'acc'
        self.MAX = 'max'

