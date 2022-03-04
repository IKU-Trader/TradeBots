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
        
        self.ADX = 'adx'
        self.ADXR = 'adxr'
        self.ADXDMI = 'adxdmi'
        self.APO = 'apo'
        self.AROON_DOWN = 'aroon_down'
        self.AROON_UP = 'aroon_up'
        self.AROON_OSC = 'aroon_osc'
        self.ATR = 'atr'
        self.BB_DOWN = 'bb_down'
        self.BB_UP = 'bb_up'
        self.BB_RATIO = 'bb_ratio'
        self.CCI = 'cci'
        self.DI_PLUS = 'di_plus'
        self.DI_MINUS = 'di_minus'
        self.DX = 'dx'
        self.EMA = 'ema'
        self.MACD = 'macd'
        self.MACD_SIGNAL = 'macd_signal'
        self.MACD_HIST = 'macd_hist'
        self.MFI = 'mfi'
        self.MOMENTUM = 'momentum'
        self.RCI = 'rci'
        self.ROC = 'roc'
        self.RSI = 'rsi'
        self.SAR = 'sar'
        self.SMA = 'sma'
        self.STOCHASTIC_SLOWK = 'stochastic_slowk'
        self.STOCHASTIC_SLOWD = 'stochastic_slowd'
        self.STOCHASTICF_FASTK = 'stochastic_fastk'
        self.STOCHASTICF_FASTD = 'stochastic_fastd'
        self.TRANGE = 'trange'
        self.VQ = 'vq'

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
        self.FASTK = 'fastk'
        self.SLOWK = 'slowk'
        self.SLOWD = 'slowd'

