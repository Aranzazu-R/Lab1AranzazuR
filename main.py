#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 02:25:30 2022

@author: charlotte
"""

import data as dt
import functions as fun

pasiva = fun.inv_pasive(dt.data_mensual, dt.pesos, dt.cash, dt.k, dt.dates)
tab = fun.rend_pasiva(pasiva)