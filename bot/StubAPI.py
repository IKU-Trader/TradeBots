# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 21:45:19 2022

@author: docs9
"""
from DataServer import DataServer, BitflyData

class StubAPI:
    def __init__(self, data_server: DataServer):
        self.server = data_server
    
    def initialData(self, size):
        if size > self.server.length:
            return None    
        self.current = size - 1
        return self.server.getData((0, size))
    
    def nextData(self):
        self.current += 1
        if self.current >= self.server.length:
            return None
        else:
            return self.server.getData((self.current, self.current + 1))

    def allData(self):
        return self.server.data
    
    
def test():
    server = BitflyData('bitfly', 'M15')
    server.loadFromCsv('./data/bitflyer_btcjpy_m15.csv') 
    api = StubAPI(server)
    data = api.initialData(500)
    if data is None:
        return
    d = api.nextData()        
        
if __name__ == '__main__':
    test()
    
    