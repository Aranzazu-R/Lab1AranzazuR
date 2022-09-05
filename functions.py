#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 02:24:53 2022

@author: charlotte
"""

#%%INVERSION PASIVA
k = 1000000
#el 31 de enero de 2020 inicia la inversion con los precios iniciales (no olvidar considerar el cash)
#Posturas = Multiplicar cada peso (en decimales) por el capital 
#Titulos = dividir posturas / precios

import pandas as pd
import numpy as np


    
def inv_pasive(prices:"dataframe with prices", weights:"vector with weights", cash:"amount of cash", k:"capital inicial", dates:'vector de fechas'):
    """
    Function that calculates the pasive investment, take 4 inputs and returns a table with the result of the investment
    """
    p_inicial = []
    posturas_d = weights*k #Posturas en dinero
    posturas_t = (posturas_d/prices.iloc[0,:]).round(0) #Posturas en titulos
    
    for date in dates:
        p_inicial['posturas_d'] = prices['date'] * posturas_t
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
