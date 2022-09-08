#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 02:24:53 2022

@author: charlotte
"""

import pandas as pd
import numpy as np
import data as dt
#%% INVERSION PASIVA

def inv_pasive(prices:"precios mensuales(df)", weights: "pesos iniciales(array)", cash: "cash(int)", k: "capital inicial", dates: "fechas mensuales(lista)"):
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

def rend_pasiva(capital:'capital por mes(df)'):
    '''
    Esta función completa el df realizado en la función inv_pasive, con rendimientos
    simples y acumulados.
    '''
    capital['rend'] = capital.pct_change().fillna(0).round(4)
    capital['rend_acum'] = 100*((capital.rend+1).cumprod()-1).round(4)
    capital['rend'] = 100*capital.rend
    return capital


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
    rf = 0.0429
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
    return df_pesos
#%%
def inv_active1(prices:'precios (solo del periodo de inversion)', weights: 'pesos',rend: 'rendimientos', k:'capital'):
    """
    Esta funcion regresara las condiciones iniciales del portafolio eficiente
    """
    cash = k
    com_acum = 0 # comisiones acumuladas
    p_com = 0.00125 # porcentaje de la comision
    comission = lambda titles,price: titles*price*p_com
    num_titles = np.zeros(len(weights)) #inicializar vector de titulos
    positions = rend.T.loc[weights.index.to_list(),:].iloc[:,13].sort_values( ascending = False) # rendimientos diarios
    order = [rend.T.loc[weights.index.to_list(),:].index.get_loc(positions.index[i]) for i in range(len(positions))] # titulos segun los rendimientos diarios  
    emv_weights = weights.sort_index().iloc[:,0].to_numpy()/100 # pesos del portafolio
    for i in order:
        if (cash > 0) and (np.floor((k*emv_weights[i]/prices[i]))*prices[i] < cash):
            num_titles[i]=(np.floor((k*emv_weights[i]/prices[i])))
            cash = cash+(-1-p_com)*np.floor((k*emv_weights[i]/prices[i]))*prices[i]
            com_acum += comission(np.floor((k*emv_weights[i]/prices[i])),prices[i])
        else: 
            num_titles[i]=(np.floor(((1-p_com)*cash/prices[i])))
            cash = cash+(-1-p_com)*np.floor(((1-p_com)*cash/prices[i]))*prices[i]
            com_acum += comission(np.floor(((1-p_com)*cash/prices[i])),prices[i])
    return num_titles, com_acum, emv_weights, cash

#%% REBALANCEO
def inv_active2(returns: 'retornos mensuales', prices:'precios mensuales', num_titles:'titulos iniciales por accion', com_acum:'comision despues del primer movimiento', emv_weights:'pesos eficientes iniciales ', cash:'efectivo restante', weights:'pesos emv', rend: 'rendimientos mensuales'):
    '''
    Esta funcion lleva a cabo el rebalanceo del protafolio a traves de los 18 periodos restantes y
    devuelve el capital, los titulos por adquirir y las comisiones por mes
    '''
    p_com = 0.00125 # porcentaje de la comision
    comission = lambda titles,price: titles*price*p_com
    com_total = [com_acum] # comision acumulada por mes
    pesos_in = [num_titles] # posiciones iniciales a comprar
    titulos = [num_titles.sum()] # posiciones totales
    for i in range(12,len(returns)-1):
            venta = returns.columns[returns.iloc[i,:]< -0.05] # si el precio bajo mas del 5% se venden
            tickers_venta = [returns.columns.get_loc(stock) for stock in venta] # tickers de las acciones a vender
            new_weights = np.zeros(len(emv_weights))
            for j in tickers_venta:
                new_weights[j] = -np.floor(num_titles[j]*0.025) # disminuir la posicion en 2.5%
                cash +=  np.floor(num_titles[j]*0.025)*prices.iloc[i+2,j]
            compra = returns.columns[returns.iloc[i,:] > 0.05] # si el precio incrementa en mas del 5%, se compran
            tickers_compra = [returns.columns.get_loc(stock) for stock in compra]# acciones a comprar
            positions = rend.T.loc[weights.index.to_list(),:].iloc[:,i+1].sort_values( ascending = False)
            order = [rend.T.loc[weights.index.to_list(),:].index.get_loc(positions.index[ors]) for ors in range(len(positions))]
            purchase_order = [x for x in order if x in tickers_compra]
            new_com = 0 # nuevas comisiones
            new_titles = 0 # cantidad de nuevos titulos
            for j in purchase_order:
                if (cash > 0) and (np.floor(num_titles[j]*0.025)*prices.iloc[i+2,j] < cash): # aumentar la posicion en 2.5% (si es que tengo el suficiente dinero)
                      new_weights[j] = np.floor(num_titles[j]*0.025)
                      new_titles += np.floor(num_titles[j]*0.025)
                      cash += -np.floor(num_titles[j]*0.025)*prices.iloc[i+2,j]-comission(np.floor(num_titles[j]*0.025),prices.iloc[i+2,j])
                      new_com += comission(np.floor(num_titles[j]*0.025),prices.iloc[i+2,j])
                else:
                      new_weights[j] = np.floor(cash/prices.iloc[i+2,j])
                      new_titles += np.floor(cash/prices.iloc[i+2,j])
                      cash += -np.floor(cash/prices.iloc[i+2,j])*prices.iloc[i+2,j]-comission(np.floor(cash/prices.iloc[i+2,j]),prices.iloc[i+2,j])
                      new_com += comission(np.floor(cash/prices.iloc[i+2,j]),prices.iloc[i+2,j])
            titulos.append(new_titles)
            com_total.append(new_com) #comisiones por mes 
            num_titles = num_titles + new_weights 
            pesos_in.append(num_titles)
    cap = pd.DataFrame(index = prices.index[13:], data = np.multiply(np.array(pesos_in),np.array(prices[13:])).sum(axis=1), columns = ['Capital'])
    return cap, titulos, com_total
#%% Data Frames
def rend_activa(capital:'capital por mes(df)'):
    '''
    Esta función completa el df realizado en la función inv_activa, con rendimientos
    simples y acumulados.
    '''
    capital['rend'] = capital.pct_change().fillna(0).round(4)
    capital['rend_acum'] = 100*((capital.rend+1).cumprod()-1).round(4)
    capital['rend'] = 100*capital.rend
    return capital

def operaciones(prices:'', titles:'', comission:''):
    '''
    Esta funcion devuelve el historico de operaciones de la inversion activa
    '''
    df_titulos = pd.DataFrame(index = prices.index[13:],data={'titulos_comprados':titles,'comision':comission})
    df_titulos['titulos_totales'] = df_titulos['titulos_comprados'].cumsum()
    df_titulos['comision_acum'] = df_titulos['comision'].cumsum()
    return df_titulos

def medidas(active: 'df rendimeinto inv activa', pasive: 'df rendimiento inv pasiva'):
    act1 = active.iloc[:,1]
    pas1 = pasive.iloc[:,1]
    act_acum = active.iloc[:,2]
    pas_acum = pasive.iloc[:,2]
    sharpe_act = act1.mean()/act1.std()
    sharpe_pas = pas1.mean()/pas1.std()
    prom_act = act1.mean()
    prom_pas = pas1.mean()
    prom_act_c = act_acum[-1]
    prom_pas_c = pas_acum[-1]
    tab = pd.DataFrame({'medida':['rend_m','rend_c','sharpe'], 'descripcion':['Rendimiento Promedio Mensual','Rendimiento mensual acumulado','Sharpe Ratio'],
                        'inv_activa':[prom_act,prom_act_c,sharpe_act],'inv_pasiva':[prom_pas,prom_pas_c,sharpe_pas]})
    return tab




