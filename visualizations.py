#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 14:36:46 2022

@author: charlotte
"""
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import data as dt

def returns_plot(table_returns):
    fig = px.line(table_returns.capital, title = "Amount in portfolio")
    fig.show()
    return fig

#rend_men_p = dt.rend_closes.pct_change().dropna()

a = dt.rend_closes