#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 02:25:30 2022

@author: charlotte
"""

import data as dt
import functions as fun
import visualizations as vs

cap_pasiva = fun.inv_pasive(dt.data_mensual, dt.pesos, dt.cash, dt.k, dt.dates)
df_pasiva = fun.rend_pasiva(cap_pasiva)

#plot = vs.returns_plot(df_pasiva)


