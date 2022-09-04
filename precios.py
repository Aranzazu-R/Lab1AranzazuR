#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 23:38:55 2022

@author: charlotte
"""
import pandas as pd
import numpy as np
import yfinance as yf
import os
#from functions import change_char

# buscar los tickers que aparecen en todos los archivos (30-31)
# Leer todos los archivos y concatenarlos, contar losvalores unicos
# buscar aquellos tickers que esten 31 veces (o sea que estan en los 31 archivos)
# manualmente agregal el ".MX" y quitar los "*", "-", "."
# de los archivos necesito los tickers de los activos que se repiten en todas, las fechas y los pesos iniciales

# Para la pasiva filtrar los últimos precios
# Para el primer año de activa, utilizar precios diarios
# a partir del 2021, se empieza el rebalanceo mensual
#%% Data from file NAFTRAC 31/01/2022
file_data = pd.read_csv("files/NAFTRAC_20200131.csv", skiprows =2)

### Pesos iniciales
weights1 = file_data.iloc[:-1,0:4:3]

# Tickers a eliminar "KOFL", "KOFUBL", "USD", "BSMXB","NMKA"
weights1 = weights1[((weights1.Ticker != "KOFUBL") & (weights1.Ticker != "BSMXB"))]
weights1.sort_values(by = 'Ticker',inplace=True)

# A porcentajes
#weights1.loc['Peso (%)',:] = weights1["Peso (%)"]/100
weights1["Peso (%)"] = weights1["Peso (%)"]/ 100
pesos = weights1.iloc[:,1].to_numpy()

#%% Descargar precios

def change_char(x:'String to me modify'):
    try:
        x = ''.join(ch for ch in x if ch not in ['*'])
    except:
        pass
    return x

Tickers = weights1.Ticker.map(change_char).map(lambda x: x.replace('.','-')).map(lambda x: x+'.MX')


#Tickers = weights1.Ticker.to_list()
start = '2020-01-31'
end = '2022-07-29'

data = yf.download(tickers =Tickers[Tickers != "MXN.MX"].to_list(), start = start, end = end, interval = '1d')
data_p = data['Close'].dropna(axis=0)
#%%
# Rendimientos Logaritmicos
rend1 = np.log(data_p/data_p.shift(1)).dropna()

Retornos1 = rend1.mean()
Sigma1 = rend1.cov()

'''
#%% Section daily data Active Investment
data_act_d = yf.download(tickers=Tickers[Tickers != "MXN.MX"].to_list(), start='2020-01-31', end= '2022-01-27', interval='1d')
data_act_closes = data_act_d['Close'].dropna(axis=0)
data_rend_closes = data_act_closes.pct_change().dropna()
closes_rend_dates = data_rend_closes[np.roll(data_rend_closes.index.isin(dates),-1)]
### data corresponding to end of month 
data_M = data_act_closes.loc[dates,:]




#%% Section constants

rf = 0.0429/252
capital = 1000000
cash = capital*weights_initial.iloc[34,1]/100+capital*weights_initial.iloc[10,1]/100+capital*weights_initial.iloc[32,1]/100
'''