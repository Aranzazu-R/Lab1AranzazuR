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


# Pesos portafolio eficiente 
pesos_emv = fun.port_eficiente(dt.data_p, dt.Tickers)

prices_act = dt.data_mensual.loc[:,dt.data_mensual.columns.isin(pesos_emv.index.to_list())]
inv_activaSol = fun.inv_active(prices_act,pesos_emv,dt.closes_rend_dates,dt.k)

tabla_rend_activa = fun.tabla_rend(inv_activaSol[0])

df_titulos = inv_activaSol[1]