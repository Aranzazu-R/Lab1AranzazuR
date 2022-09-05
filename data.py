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
weights_in2 = file_data.iloc[:-1,0:4:3]

# Eliminar Tickers que no se repiten
for i in t_no:
    weights1 = weights1[(weights1.Ticker != i)]   

# Tickers extra a eliminar "KOFL", "KOFUBL", "USD", "BSMXB","NMKA"
weights1 = weights1[(weights1.Ticker != 'KOFL') & (weights1.Ticker != 'KOFUBL') & (weights1.Ticker != 'MXN')]

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
#%% Precios Iniciales INV PASIVA

precio_inicial = data_p.iloc[0]
#%% Rendimientos Logaritmicos
rend1 = np.log(data_p/data_p.shift(1)).dropna()

Retornos1 = rend1.mean()
Sigma1 = rend1.cov()


#%% Section daily data Active Investment
start = '2020-01-31'
end = '2022-07-29'
data_act_d = yf.download(tickers = Tickers, start = start, end= end, interval='1d')
data_act_closes = data_act_d['Close'].dropna(axis=0)
#%%
docs = os.listdir(os.getcwd()+"/files")

dates = [x[8:12] + "-"+ x[12:14] + "-" + x[14:16] for x in docs]
#%% se cambio la fecha manualmente de 2022-07-29 a 2022-07-28
data_rend_closes = data_act_closes.pct_change().dropna()
closes_rend_dates = data_rend_closes[np.roll(data_rend_closes.index.isin(dates),-1)]
### data corresponding to end of month 
data_M = data_act_closes.loc[dates,:]

#%% Constantes
k = 1000000 #capital de 1 millon
#CASH = "KOFL", "KOFUBL", "USD", "BSMXB","NMKA" (solo tienen ponderacion inicial "KOFUBL" y  "BSMXB")
cash = k*weights_in2.iloc[10,1]/100+k*weights_in2.iloc[34,1]/100


#%%
def inv_pasive(prices:"dataframe with prices", weights:"vector with weights", cash:"amount of cash", k:"capital inicial", dates:'vector de fechas'):
    """
    Function that calculates the pasive investment, take 4 inputs and returns a table with the result of the investment
    """
    posturas_d = weights*k #Posturas en dinero
    posturas_t = (posturas_d/prices.iloc[0,:]).round(0) #Posturas en titulos
    
    for i in dates:
        p_inicial = pd.DataFrame()
        p_inicial['posturas_d'] = prices[i] * posturas_t
    return p_inicial

inv_pasive(data_M, pesos, cash, k,dates)

#%%
    cash = cash-sum([comission(posturas_t[i],prices.iloc[0,i]) for i in range(len(posturas_t))]) # comisiones por movimientos
    amount = [(posturas_t.to_numpy()).dot(prices.iloc[i,:].to_numpy()) for i in range(len(prices))]
    out = pd.DataFrame(index = prices.index,columns= ['Capital'],data = amount)
    out['Capital'] = out.Capital#correction with the cash 
    return out

#def tabla_rend(data:"dataframe with capital"):
#    data['rend'] =data.Capital.pct_change().fillna(0).round(6)
#    data['rend_acum']= 100*((data.rend+1).cumprod()-1).round(6)
#    data['rend'] = 100*data.rend
#    return data



comission = lambda securitie,price: securitie*price*0.00125









