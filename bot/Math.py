# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 21:36:39 2022

@author: docs9
"""

class Math:
    
    @classmethod
    def accumulate(cls, array):
        s = 0.0
        out = []
        for a in array:
            s += a
            out.append(s)
        return out

        