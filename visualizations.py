#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 14:36:46 2022

@author: charlotte
"""

import matplotlib.pyplot as plt
import data as dt
import main as mn

#%% DATA
plt.plot(dt.data_mensual)
plt.show()
#%%
plt.plot(dt.weights1.iloc[1])
plt.show()
#%% PASIVA

cap_act = mn.test_active2[0]
fig, ax = plt.subplots()
ax.plot(cap_act.iloc[:,0], label ='Inv_Pasiva')
ax.plot(mn.cap_pasiva.iloc[:,0], label ='Inv_Pasiva')
plt.ylabel('Fechas')
plt.ylabel('Capital')
plt.title("Inv Activa vs Inv Pasiva")
plt.grid(True)
plt.show()
#%% 
plt.subplots(mn.pesos_emv)
plt.show()
#%%
fig, ax = plt.subplots()
ax.pie(mn.pesos_emv)
plt.show()