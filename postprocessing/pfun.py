import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import interp1d

def boxplot_error(data, error_column, mode):
    if mode == 'split':
        xvar = 'random state'
        xaxis_title = 'random state of the GroupShuffleSplit'
    elif mode == 'depth':
        xvar = 'decrease duration size'
        xaxis_title = 'decrease duration size [%]'
    elif mode == 'trial':
        xvar = 'decrease trials size'
        xaxis_title = 'decrease trials size [%]'

    error_column = error_column.lower()

    fig = go.Figure()
    groups = data['group'].unique()
    
    for group in groups:
        group_data = data[data['group'] == group]
        fig.add_trace(go.Box(y=group_data[error_column].values ,x=group_data[xvar], name=group, boxpoints='all', jitter=0.3, pointpos=-1.8, boxmean='sd'))
    fig.update_layout(title=f'{error_column} error boxplot', yaxis_title=f'{error_column} error', xaxis_title=xaxis_title)
    
    fig.show()

def plot_avg_values(data, error_column, mode): # split doesnt work since the all have the same decrease duration size or decrease trials size!!!
    if mode == 'split':
        xvar = 'random state'
        xaxis_title = 'random state of the GroupShuffleSplit'
    elif mode == 'depth':
        xvar = 'decrease duration size'
        xaxis_title = 'decrease duration size [%]'
    elif mode == 'trial':
        xvar = 'decrease trials size'
        xaxis_title = 'decrease trials size [%]'

    error_column = error_column.lower()

    fig = go.Figure()
    #groups = data['group'].unique()
    #for group in groups:
    #group_data = data[data['group'] == group]
    
    y=data[error_column].dropna().values
    mask = ~data[error_column].isna()
    mask_indices = np.where(mask)[0]
    new_mask = np.zeros_like(mask, dtype=bool)
    new_mask[mask_indices + 1] = True
    mask = new_mask
    x_values = data.loc[mask, xvar].values
    x=data[xvar].dropna().unique()

    #sorted_indices = np.argsort(x)
    x_sorted = x_values#[sorted_indices]
    y_sorted = y#[sorted_indices]

    # Fit a linear model (y = mx + b)
    m, b = np.polyfit(x_sorted, y_sorted, 1)  # 1st-degree polynomial (linear fit)
    # Fit a square model (y = ax^2 + bx + c)
    #a, d, c = np.polyfit(x_sorted, y_sorted, 2)  # 2nd-degree polynomial (square fit)

    # Generate fitted y-values
    y_fit = m * x_sorted + b  # Compute y values for fitted line
    #y_fit_square = a * x_sorted ** 2 + d * x_sorted + c  # Compute y values for fitted square

    fig.add_trace(go.Scatter(y=y,x=x_values, mode='markers', name=f'{error_column}', marker=dict(size=12)))
    fig.update_layout(title=f'{error_column} error plot', yaxis_title=f'{error_column} error', xaxis_title=xaxis_title)

    fig.add_trace(go.Scatter(y=y_fit, x=x_sorted, mode='lines', line=dict(color='gray', width=2), name='Linear fit'))
    #fig.add_trace(go.Scatter(y=y_fit_square, x=x_sorted, mode='lines', line=dict(color='red', width=2), name='Square fit'))
    fig.show()

def plot_NN(data, metric):
    metric= metric.lower()
    x=data['random state'].dropna().unique()
    if metric=='loss':
        fig=go.Figure()
        fig.add_trace(go.Scatter(y=data[f'avg val {metric}'].dropna().values, x=x, mode='lines+markers', name=f'validation {metric}', marker=dict(size=12)))
        fig.add_trace(go.Scatter(y=data[f'avg test {metric}'].dropna().values, x=x, mode='lines+markers', name=f'test {metric}', marker=dict(size=12)))
        fig.update_layout(title=f'{metric} plot', yaxis_title=f'{metric}', xaxis_title='random state')
        fig.show()
    elif metric=='r2':
        fig=go.Figure()
        fig.add_trace(go.Scatter(y=data[f'avg val {metric} score'].dropna().values, x=x, mode='lines+markers', name=f'validation {metric}', marker=dict(size=12)))
        fig.add_trace(go.Scatter(y=data[f'avg test {metric} score'].dropna().values, x=x, mode='lines+markers', name=f'test {metric}', marker=dict(size=12)))
        fig.update_layout(title=f'{metric} plot', yaxis_title=f'{metric}', xaxis_title='random state')
        fig.show()
    elif metric=='mae':
        fig=go.Figure()
        fig.add_trace(go.Scatter(y=data[f'avg val {metric}'].dropna().values, x=x, mode='lines+markers', name=f'validation {metric}', marker=dict(size=12)))
        fig.add_trace(go.Scatter(y=data[f'avg test {metric}'].dropna().values, x=x, mode='lines+markers', name=f'test {metric}', marker=dict(size=12)))
        fig.update_layout(title=f'{metric} plot', yaxis_title=f'{metric}', xaxis_title='random state')
        fig.show()
    elif metric=='rmse':
        fig=go.Figure()
        fig.add_trace(go.Scatter(y=data[f'avg val {metric}'].dropna().values, x=x, mode='lines+markers', name=f'validation {metric}', marker=dict(size=12)))
        fig.add_trace(go.Scatter(y=data[f'avg test {metric}'].dropna().values, x=x, mode='lines+markers', name=f'test {metric}', marker=dict(size=12)))
        fig.update_layout(title=f'{metric} plot', yaxis_title=f'{metric}', xaxis_title='random state')
        fig.show()

    