#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 11:37:39 2022

@author: iku
"""

from datetime import datetime
import pytz
jp = pytz.timezone('Asia/Tokyo')
from bot.DataServer import BitflyData

def analyze(tohlc: dict):
    
    
    
    
    return

if __name__ == '__main__':
    server = BitflyData('bitfly', 'M15')
    server.loadFromCsv('./data/bitflyer_btcjpy_m15.csv')
    t = jp.localize(datetime(2020, 7, 5, 7, 0))
    data = server.dataFrom(t,  2 * 4 * 24)
    
    analyze(data)