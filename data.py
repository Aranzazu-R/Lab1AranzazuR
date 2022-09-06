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

#%% Rendimientos Logaritmicos
rend1 = np.log(data_p/data_p.shift(1)).dropna()

Retornos1 = rend1.mean()
Sigma1 = rend1.cov()

# Section daily data Active Investment
data_act_d = yf.download(tickers = Tickers, start = start, end= end, interval='1d')
data_act_closes = data_act_d['Close'].dropna(axis=0)
#%%
docs = os.listdir(os.getcwd()+"/files")

dates = [x[8:12] + "-"+ x[12:14] + "-" + x[14:16] for x in docs]

dates.sort()
#%% se cambio la fecha manualmente de 2022-07-29 a 2022-07-28
dates[-1] = '2022-07-28'
# data correspondiente al final de cada mes 
data_mensual = data_p.loc[dates,:]
rend_closes = data_p.pct_change().dropna()
closes_rend_dates = rend_closes[np.roll(rend_closes.index.isin(dates),-1)]

#%% Constantes
k = 1000000 #capital de 1 millon
#CASH = "KOFL", "KOFUBL", "USD", "BSMXB","NMKA" (solo tienen ponderacion inicial "KOFUBL" y  "BSMXB")
cash = k*weights_in2.iloc[10,1]/100+k*weights_in2.iloc[34,1]/100




#%% INVERSION PASIVA

def inv_pasive(prices:"precios mensuales(df)", weights: "pesos iniciales(array)", cash: "cash", k: "capital inicial", dates: "fechas mensuales(lista)"):
    """
    Esta funcion calcula el capital restante por mes, despues de los movimientos en el precio de cada activo.
    """
    posturas_d = weights*k #Posturas en dinero
    posturas_t = (posturas_d/prices.iloc[0,:]).round(0) #Posturas en titulos
    p_inicial = pd.DataFrame()
    p_inicial['Titulos'] = posturas_t
    p_inicial['Posturas $'] = posturas_d
    comision = (p_inicial['Titulos']*0.00125*prices.iloc[0,:]) 
    c = comision.sum() #comisiones total
    r = cash - c #cash restante despues de comisiones
    cap_men = [(posturas_t.to_numpy()).dot(prices.loc[i,:].to_numpy()) for i in dates]
    cap_men[0] = cap_men[0]+r #sumar el cash restante al primer mes (despues de pagarlas)
    df_capital = pd.DataFrame(index = prices.index, columns = ['capital'],data = cap_men)
    return df_capital
pasiva = inv_pasive(data_mensual, pesos, cash, k, dates)

def rend_pasiva(capital:'capital por mes(df)'):
    '''
    Esta función completa el df realizado en la función inv_pasive, con rendimientos
    simples y acumulados.
    '''
    capital['rend'] = capital.pct_change().fillna(0).round(4)
    capital['rend_acum'] = 100*((capital.rend+1).cumprod()-1).round(4)
    capital['rend'] = 100*capital.rend
    return capital

tab = rend_pasiva(pasiva)

#%% INVERSION ACTIVA
# funcion para portafolio efciente maximizando sharpe
from scipy.optimize import minimize 

# Portafolio EMV
def port_eficiente(data:'data mensual',ind: "list of index"):
    rend_closes = data.pct_change().dropna() # Calcular los rendimientos
    # Resumen en base anual
    annual_ret_summary = pd.DataFrame({'Media': 252 * rend_closes.mean(), 'Vol': np.sqrt(252) * rend_closes.std()})
    corr = rend_closes.corr() #Matriz de correlacion 
    ## Construcción de parámetros
    # 1. Sigma: matriz de varianza-covarianza Sigma = S.dot(corr).dot(S)
    S = np.diag(annual_ret_summary.T.loc['Vol'].values)
    Sigma = S.dot(corr).dot(S)
    # 2. Eind: rendimientos esperados activos individuales
    Eind = annual_ret_summary.T.loc['Media'].values
    rf = 0.0775
    N = len(Eind) # Número de activos
    w0 = np.ones(N) / N# Dato inicial
    bnds = ((0, 1), ) * N # Cotas de las variables
    cons = {'type': 'eq', 'fun': lambda w: w.sum() - 1} # Restricciones
    varianza = lambda w, Sigma: w.T.dot(Sigma).dot(w)
    rendimiento = lambda w, r: w.dot(r)
    maxi = lambda w,Eind,rf,Sigma: -(rendimiento(w,Eind)-rf)/ varianza(w,Sigma)**0.5
    emv = minimize(fun=maxi,
                   x0=w0,
                   args=(Eind, rf, Sigma),
                   bounds=bnds,
                   constraints=cons)
    w_emv = emv.x
    df_pesos = pd.DataFrame(index =ind, columns = ['Peso %'], data = w_emv*100)
    df_pesos_emv = df_pesos.loc[df_pesos['Peso %']>0,:]
    return df_pesos_emv

# Pesos portafolio eficiente 
pesos_emv = port_eficiente(data_p, Tickers)


#%%
def inv_active(prices:"dataframe with prices", weights:"vector with weights",rendi_d:"dataframe with daily returns", k:"initial amount of money"):
    """
    Function that calculates the pasive investment, take 4 inputs and returns a table with the result of the investment
    """
    ##### calculations of the initial positions
    cash = 1000000 # cash to use in purchases
    com = 0 # commision quantity
    comission = lambda titles,price: titles*price*0.00125
    num_titles = np.zeros(len(weights)) #inicializar vector de titulos
    positions = rendi_d.T.loc[weights.index.to_list(),:].iloc[:,12].sort_values( ascending = False)#vector with daily returns sorted descending base on previous day 
    order = [rendi_d.T.loc[weights.index.to_list(),:].index.get_loc(positions.index[i]) for i in range(len(positions))]#list with index of securities based on daily returns 
    n_weights = weights.sort_index().iloc[:,0].to_numpy()/100 # vector with portfolio weights
    for posi in order:
        if (cash > 0) and (np.floor((k*n_weights[posi]/prices.iloc[13,posi]))*prices.iloc[13,posi] < cash):
            num_titles[posi]=(np.floor((k*n_weights[posi]/prices.iloc[13,posi])))
            cash = cash -1.00125*np.floor((k*n_weights[posi]/prices.iloc[13,posi]))*prices.iloc[13,posi]
            com += comission(np.floor((k*n_weights[posi]/prices.iloc[13,posi])),prices.iloc[13,posi])
        else: # case were there its no cash to cover all the positions 
            num_titles[posi]=(np.floor((0.99875*cash/prices.iloc[13,posi])))
            cash = cash -1.00125*np.floor((0.99875*cash/prices.iloc[13,posi]))*prices.iloc[13,posi]
            com += comission(np.floor((0.99875*cash/prices.iloc[13,posi])),prices.iloc[13,posi])
    
    ######## Calculations for the new portfolio weights
    returns = np.log(prices/prices.shift(1)).dropna() # dataframe with montly returns to determine the securitie to hold, seld or buy 
    wei = [num_titles]# list with positions 
    titles = [num_titles.sum()] # list with the total number of securities buy it per month
    commi = [com] # list with the total comission per month
    for i in range(12,len(returns)-1):
            sales = returns.columns[returns.iloc[i,:]< -1*0.05]# securities that must be sold 
            index_sales = [returns.columns.get_loc(stock) for stock in sales]# index of securities to sold
            new_weights = np.zeros(len(n_weights))
            for j in index_sales:
                new_weights[j] = -np.floor(num_titles[j]*0.025)
                cash +=  np.floor(num_titles[j]*0.025)*prices.iloc[i+2,j]
            purchase = returns.columns[returns.iloc[i,:] > 0.05] # securities that must be buy
            index_purchase = [returns.columns.get_loc(stock) for stock in purchase]# index of securities to buy
            positions = rendi_d.T.loc[weights.index.to_list(),:].iloc[:,i+1].sort_values( ascending = False)
            order = [rendi_d.T.loc[weights.index.to_list(),:].index.get_loc(positions.index[ors]) for ors in range(len(positions))]
            purchase_order = [w for w in order if w  in index_purchase]
            nn_titles = 0#number of titles bought
            com = 0 #payed comissions 
            for j in purchase_order:
                if (cash > 0) and (np.floor(num_titles[j]*0.025)*prices.iloc[i+2,j] < cash):
                      new_weights[j] = np.floor(num_titles[j]*0.025)
                      nn_titles += np.floor(num_titles[j]*0.025)
                      cash += -np.floor(num_titles[j]*0.025)*prices.iloc[i+2,j]-comission(np.floor(num_titles[j]*0.025),prices.iloc[i+2,j])
                      com += comission(np.floor(num_titles[j]*0.025),prices.iloc[i+2,j])
                else:
                      new_weights[j] = np.floor(cash/prices.iloc[i+2,j])
                      nn_titles += np.floor(cash/prices.iloc[i+2,j])
                      cash += -np.floor(cash/prices.iloc[i+2,j])*prices.iloc[i+2,j]-comission(np.floor(cash/prices.iloc[i+2,j]),prices.iloc[i+2,j])
                      com += comission(np.floor(cash/prices.iloc[i+2,j]),prices.iloc[i+2,j])
            titles.append(nn_titles)
            commi.append(com)#month commisions 
            num_titles = num_titles + new_weights#new positions
            wei.append(num_titles)
    out = pd.DataFrame(index = prices.index[13:], data = np.multiply(np.array(wei),np.array(prices[13:])).sum(axis=1), columns = ['Capital'])#dataframe with portfolio's value
    df_titulos = pd.DataFrame(index = prices.index[13:],data={'titulos_comprados':titles,'comision':commi})#dataframe with commisions and number of securities
    df_titulos['titulos_totales'] = df_titulos['titulos_comprados'].cumsum()
    df_titulos['comision_acum'] = df_titulos['comision'].cumsum()
    return out, df_titulos


#%%
prices_act = data_mensual.loc[:,data_mensual.columns.isin(pesos_emv.index.to_list())]
inv_activaSol = inv_active(prices_act,pesos_emv,closes_rend_dates,k)
#%%
def tabla_rend(data:"dataframe with capital"):
    data['rend'] =data.Capital.pct_change().fillna(0).round(6)
    data['rend_acum']= 100*((data.rend+1).cumprod()-1).round(6)
    data['rend'] = 100*data.rend
    return data


tabla_rend_activa = tabla_rend(inv_activaSol[0])

df_titulos = inv_activaSol[1]


