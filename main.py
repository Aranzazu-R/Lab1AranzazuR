#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 02:25:30 2022

@author: charlotte
"""

import data as dt
import functions as fun
import visualizations as vs

#%% INV PASIVA
cap_pasiva = fun.inv_pasive(dt.data_mensual.iloc[13:], dt.pesos, dt.cash, dt.k, dt.dates[13:])
df_pasiva = fun.rend_pasiva(cap_pasiva)

#%% INV ACTIVA

# Pesos portafolio eficiente 
pesos_emv = fun.port_eficiente(dt.data_p, dt.Tickers)

# Inversion activa
test_active = fun.inv_active1(dt.data_mensual.iloc[13,],pesos_emv, dt.rend_mensual,dt.k)
test_active2 = fun.inv_active2(dt.data_mensual,test_active[0],test_active[1],test_active[2],test_active[3],pesos_emv,dt.rend_mensual)
prices_act = dt.data_mensual.loc[:,dt.data_mensual.columns.isin(pesos_emv.index.to_list())]
#tablas 
df_activa = fun.rend_activa(test_active2[0])
df_operaciones = fun.operaciones(prices_act, test_active2[1], test_active2[2])


df_medidas = fun.medidas(df_activa,df_pasiva)

