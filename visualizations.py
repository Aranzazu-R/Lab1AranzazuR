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



#%% PASIVA
plt.style.use('fivethirtyeight')
plt.ylabel('Capital')
plot_cap_pas = mn.cap_pasiva.iloc[:,0].plot(figsize=(10,8))

#%% ACTIVA
plt.style.use('fivethirtyeight')
cap_act = mn.test_active2[0]
plt.ylabel('Capital')
plot_cap_act = cap_act.iloc[:,0].plot(figsize=(10,8))

