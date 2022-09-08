#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 14:36:46 2022

@author: charlotte
"""

import matplotlib.pyplot as plt


def graph_tab(data:'datos a graficar en gráfica de tablas', ejex:'nombre de el eje x', ejey:'nombre de el eje y', titulo:'titulo de la grafica'):
    plt.style.use('ggplot')
    data.plot(kind='bar')
    plt.xlabel(ejex)
    plt.ylabel(ejey)
    plt.title(titulo)
    return plt.show()

def graph_lin(data:'datos a graficar', ejex:'nombre de el eje x', ejey:'nombre de el eje y', titulo:'titulo de la grafica'):
    plt.style.use('ggplot')
    data.plot()
    plt.xlabel(ejex)
    plt.ylabel(ejey)
    plt.title(titulo)
    return plt.show()

def graph_lin2(data1:'datos a graficar', data2:'datos a graficar', ejex:'nombre de el eje x', ejey:'nombre de el eje y', titulo:'titulo de la grafica'):
    plt.style.use('ggplot')
    data1.plot(label ='Inv_Pasiva')
    data2.plot(label ='Inv_Pasiva')
    plt.xlabel(ejex)
    plt.ylabel(ejey)
    plt.title(titulo)
    return plt.show()
