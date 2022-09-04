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
# a partir del 2021, se empieza el rebalanceo 

#%%
import glob
# importamos todos los archivos
csv_files = glob.glob('files/*.csv')
# concatenar todos los archivos
list_files = []
# Escribimos un loop que irá a través de cada uno de los nombres de archivo a través de globbing y el resultado final será la lista dataframes
for filename in csv_files:
    data = pd.read_csv(filename, skiprows =2)
    list_files.append(data)
 
df_files = pd.concat(list_files)

# buscar los Ticker que no se repiten 31 veces
val_u = df_files.groupby('Ticker').count().reset_index()
no_repetidos = val_u['Nombre'] != 31
repetidos = val_u['Nombre'] == 31

#Tickers que no se van a usar
t_no_repetidos = val_u[no_repetidos]
t_no_repetidos = t_no_repetidos.iloc[:,0]
t_no = t_no_repetidos.tolist()

# Tickers que si se van a usar
t_repetidos = val_u[repetidos]
t_repetidos = t_repetidos.iloc[:,0]
t = t_repetidos.tolist()

#%% Data (WEIGHTS) from file NAFTRAC 31/01/2022
file_data = pd.read_csv("files/NAFTRAC_20200131.csv", skiprows =2)

# Pesos iniciales
weights1 = file_data.iloc[:-1,0:4:3]

# Eliminar Tickers que no se repiten
for i in t_no:
    weights1 = weights1[(weights1.Ticker != i)]   

# Tickers extra a eliminar "KOFL", "KOFUBL", "USD", "BSMXB","NMKA"
weights1 = weights1[(weights1.Ticker != 'KOFL') & (weights1.Ticker != 'KOFUBL')]

weights1.sort_values(by = 'Ticker',inplace=True)

#%%
# Pasar pesos de porcentajes a decimales
#weights1.loc['Peso (%)',:] = weights1["Peso (%)"]/100
weights1["Peso (%)"] = weights1["Peso (%)"]/ 100
pesos = weights1.iloc[:,1].to_numpy()

#%% Descargar precios

# Se corrigen los nombres de manera manual para descargar los precios de Yahoo
# Los nombres fueron sacados de la lista llamada t, se elimino ademas MXN.MX
Tickers = ['AC.MX', 'ALFAA.MX', 'ALSEA.MX', 'AMXL.MX', 'ASURB.MX', 'BBAJIOO.MX', 'BIMBOA.MX', 'BOLSAA.MX', 'CEMEXCPO.MX', 'CUERVO.MX', 'ELEKTRA.MX', 'FEMSAUBD.MX', 'GAPB.MX', 'GCARSOA1.MX', 'GFINBURO.MX', 'GFNORTEO.MX', 'GMEXICOB.MX', 'GRUMAB.MX', 'KIMBERA.MX', 'LABB.MX', 'LIVEPOLC-1.MX', 'MEGACPO.MX', 'OMAB.MX', 'ORBIA.MX', 'PE&OLES.MX', 'PINFRA.MX', 'TLEVISACPO.MX', 'WALMEX.MX']

start = '2020-01-31'
end = '2022-07-29'
data = yf.download(tickers = Tickers, start = start, end = end, interval = '1d')
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