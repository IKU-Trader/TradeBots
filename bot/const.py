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
        self.SPREAD = 'spread'
        self.BUY_VOL = 'buy_vol'
        self.SELL_VOL = 'sell_vol'
        
        self.LONG = 1
        self.SHORT = -1


class IndicatorConst:
# Techinical indicators
    def __init__(self):
        self.TYPE = 'type'
        self.ROR = 'ror'
        
        self.ADX = 'adx'
        self.ADXR = 'adxr'
        self.ADXDMI = 'adxdmi'
        self.APO = 'apo'
        self.AROON_DOWN = 'aroon_down'
        self.AROON_UP = 'aroon_up'
        self.AROON_OSC = 'aroon_osc'
        self.ATR = 'atr'
        self.BB_LOWER = 'bb_lower'
        self.BB_UPPER = 'bb_upper'
        self.BB_MID = 'bb_mid'
        self.BB_RATIO = 'bb_ratio'
        self.BETA = 'beta'
        self.CCI = 'cci'
        self.DEMA = 'dema'
        self.DI_PLUS = 'di_plus'
        self.DI_MINUS = 'di_minus'
        self.DX = 'dx'
        self.EMA = 'ema'
        self.HT_DC_PERIOD = 'ht_dc_period'
        self.HT_DC_PHASE = 'ht_dc_phase'
        self.HT_PHASOR_INPHASE = 'ht_phasor_inphase'
        self.HT_PHASOR_QUADRATURE = 'ht_phasor_quadrature'
        self.HT_TRENDMODE = 'ht_trendmode'
        self.HT_TRENDLINE = 'ht_trendline'
        self.KAMA = 'kama'
        self.LINEARREG = 'linearreg'
        self.LINEARREG_ANGLE = 'linearreg_angle'
        self.LINEARREG_INTERCEPT = 'linearreg_intercept'
        self.LINEARREG_SLOPE = 'linearreg_slope'
        self.MA = 'ma'
        self.MACD = 'macd'
        self.MACD_SIGNAL = 'macd_signal'
        self.MACD_HIST = 'macd_hist'
        self.MFI = 'mfi'
        self.MIDPOINT = 'midpoint'
        self.MOMENTUM = 'momentum'
        self.PPO = 'ppo'
        self.RCI = 'rci'
        self.ROC = 'roc'
        self.RSI = 'rsi'
        self.SAR = 'sar'
        self.SMA = 'sma'
        self.STDDEV = 'stddev'
        self.STOCHASTIC1_SLOWK = 'stochastic1_slowk'
        self.STOCHASTIC2_SLOWD = 'stochastic2_slowd'
        self.STOCHASTIC3_FASTK = 'stochastic3_fastk'
        self.STOCHASTIC4_FASTD = 'stochastic4_fastd'
        self.T3 = 't3'
        self.TEMA = 'tema'
        self.TRIMA = 'trima'
        self.TRIX = 'trix'
        self.TRANGE = 'trange'
        self.ULTOSC = 'ultosc'
        self.VQ = 'vq'
        self.WILLR = 'willr'
        self.WEEKDAY = 'weekday'
        self.WMA = 'wma'
        self.TIMEBAND = 'timeband'
        self.CANDLE_BODY = 'candle_body'
        self.SPIKE = 'spike'

class ParameterConst:
# parameters
    def __init__(self):
        self.WINDOW = 'window'
        self.SLOW = 'slow'
        self.MID = 'mid'
        self.FAST = 'fast'
        self.SIGNAL = 'signal'
        self.SIGMA = 'sigma'
        self.ACC = 'acc'
        self.MAX = 'max'
        self.FASTK = 'fastk'
        self.FASTD = 'fastd'
        self.SLOWK = 'slowk'
        self.SLOWD = 'slowd'

