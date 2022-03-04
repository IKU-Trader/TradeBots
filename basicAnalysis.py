#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 11:37:39 2022

@author: iku
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), './bot'))

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
jp = pytz.timezone('Asia/Tokyo')

import statsmodels.graphics.api as smg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

from CandlePlot import CandlePlot, BandPlot, makeFig, gridFig, array2graphShape, awarePyTimeList2Float
from bot.DataServer import BitflyData
from const import BasicConst, IndicatorConst, ParameterConst

c = BasicConst()
t = IndicatorConst()
p = ParameterConst()

def rateOfReturn(vector):
    n = len(vector)
    out = np.zeros(n)
    for i in range(1, n):
        if np.isnan(vector[i]) or np.isnan(vector[i - 1]):
            continue
        if vector[i - 1] == 0.0:
            continue
        out[i] = (vector[i] - vector[i - 1]) / vector[i - 1] * 100.0
    return out

def analyze(tohlc: dict):
    time = tohlc[c.TIME]
    close = tohlc[c.CLOSE]
    ror = rateOfReturn(close)
    (fig, axes) = gridFig([5, 3], (15, 5))
    fig.subplots_adjust(hspace=0.6, wspace=0.4)
    fig.tight_layout()
    graph1 = CandlePlot(fig, axes[0], "")
    keys = [c.OPEN, c.HIGH, c.LOW, c.CLOSE]
    graph1.drawCandle(time, tohlc, keys)
    graph2 = CandlePlot(fig, axes[1], "")
    graph2.drawLine(time, ror)
    
    fig, ax = makeFig(1, 1, (10, 3))
    ax.hist(ror, bins=10)
    
    ax.set_title('Rate of Return  sigma: ' + '{:7}'.format(np.std(ror)) + ' mean: ' + '{:7}'.format(np.mean(ror)))
    
    
    plot_acf(ror, lags=29);
    plot_pacf(ror, lags=29);
    
    lbvalues, pvalues = acorr_ljungbox(ror, lags=10)
    lag = 1
    for lb, p in zip(lbvalues, pvalues):
        print(lag, lb, p)
        lag += 1
        
        
    return

if __name__ == '__main__':
    server = BitflyData('bitfly', 'M15')
    server.loadFromCsv('./data/bitflyer_btcjpy_m15.csv')
    t = jp.localize(datetime(2020, 7, 5, 7, 0))
    data = server.dataFrom(t,  2 * 4 * 24)
    analyze(data)