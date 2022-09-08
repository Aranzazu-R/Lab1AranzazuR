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

#%% PASIVA

cap_act = mn.test_active2[0]
plt.plot(cap_act.iloc[:,0])
plt.plot(mn.cap_pasiva.iloc[:,0], label ='Inv_Pasiva')
plt.ylabel('Fechas')
plt.ylabel('Capital')
plt.title("Inv Activa vs Inv Pasiva", 
          fontdict={'family': 'serif', 
                    'color' : 'black',
                    'weight': 'bold',
                    'size': 18})
plt.grid(True)
plt.show()
#%% 
plt.plot(mn.pesos_emv)
plt.show()