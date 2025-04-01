import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import interp1d
import re
from PIL import Image


def split_camel_case(text):
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', text)
    return ' '.join(words)

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

def plot_ntrials_depth(dataNN, dataSGPR, metric, vtb, nnsgprboth, mode, polydegree,save, target):
    metric= metric.lower()
    vtb= vtb.lower()
    nnsgprboth = nnsgprboth.lower()
    fig = go.Figure()
    if mode =='Number of Trials':
        error_column='decrease trials size'
    elif mode == 'Depth':
        error_column='decrease duration size'

    if nnsgprboth == 'nn':
        data=dataNN
    elif nnsgprboth == 'sgpr':
        data=dataSGPR

    dataNN = dataNN[dataNN['target'] == target]
    dataSGPR = dataSGPR[dataSGPR['target'] == target]
    
    if nnsgprboth != 'both':
        if vtb == 'validation' or vtb == 'test':
            mask = data[error_column].notna()
            x_values = data.loc[mask, error_column].values*100
            y_values = data.loc[mask, f'{vtb} {metric}'].values
            fig.add_trace(go.Scatter(y=y_values, x=x_values, mode='markers', name=f'{nnsgprboth.upper()}', marker=dict(size=8)))
            fig.update_layout(title=f'{vtb.capitalize()} {metric} plot', yaxis_title=f'Prediction Error ({vtb.capitalize()} {metric.upper()} [°])', xaxis_title=f'{mode.capitalize()} [%]')
            degree = polydegree
            A=np.vander(x_values, degree +1)
            coeffs, _, _, _ = np.linalg.lstsq(A, y_values, rcond=None)
            x = np.linspace(min(x_values), max(x_values), 100)
            y = np.polyval(coeffs, x)
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'x^{degree} Fit {nnsgprboth.upper()}', line=dict(color='red')))
        
        elif vtb == 'both':
            mask = data[error_column].notna()
            x_values = data.loc[mask, error_column].values*100
            y_values = data.loc[mask, f'validation {metric}'].values
            fig.add_trace(go.Scatter(y=y_values, x=x_values-0.5, mode='markers', name=f'Validation {metric.upper()}', marker=dict(size=8))
            )
            y_values = data.loc[mask, f'test {metric}'].values
            fig.add_trace(go.Scatter(y=y_values, x=x_values+0.5, mode='markers', name=f'Test {metric.upper()}', marker=dict(size=8))
            )
            fig.update_layout(title=f'Validation and Test {metric.upper()} plot for {nnsgprboth.upper()}', yaxis_title=f'{metric.upper()}', xaxis_title=f'{mode.capitalize()} [%]')
            degree = polydegree
            A=np.vander(x_values, degree +1)
            coeffs, _, _, _ = np.linalg.lstsq(A, data.loc[mask, f'validation {metric}'].values, rcond=None)
            x = np.linspace(min(x_values), max(x_values), 100)
            y = np.polyval(coeffs, x)
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'x^{degree} Fit for Validation', line=dict(color='red')))
            coeffs, _, _, _ = np.linalg.lstsq(A, data.loc[mask, f'test {metric}'].values, rcond=None)
            y = np.polyval(coeffs, x)
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'x^{degree} Fit for Test', line=dict(color='blue')))
    
    elif nnsgprboth == 'both':
        if vtb == 'validation' or vtb == 'test':
            masknn = dataNN[error_column].notna()
            masksgpr = dataSGPR[error_column].notna()
            x_values_nn = dataNN.loc[masknn, error_column].values*100
            y_values = dataNN.loc[masknn, f'{vtb} {metric}'].values
            fig.add_trace(go.Scatter(y=y_values, x=x_values_nn-0.5, mode='markers', name=f'DNN', marker=dict(size=8, color='red')))

            x_values_sgpr = dataSGPR.loc[masksgpr, error_column].values*100
            y_values = dataSGPR.loc[masksgpr, f'{vtb} {metric}'].values
            fig.add_trace(go.Scatter(y=y_values, x=x_values_sgpr+0.5, mode='markers', name=f'SGPR', marker=dict(size=8, color='blue')))
            fig.update_layout(title=f'NN and SGPR {vtb.capitalize()} {metric} plot', yaxis_title=f'Prediction Error ({vtb.capitalize()} {metric.upper()})', xaxis_title=f'{mode.capitalize()} [%]')
            
            degree = polydegree
            A=np.vander(x_values_nn, degree +1)
            coeffs, _, _, _ = np.linalg.lstsq(A, dataNN.loc[masknn, f'{vtb} {metric}'].values, rcond=None)
            x = np.linspace(min(x_values_nn), max(x_values_nn), 100)
            y = np.polyval(coeffs, x)
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'x^{degree} Fit DNN', line=dict(color='red')))
            
            A=np.vander(x_values_sgpr, degree +1)
            coeffs, _, _, _ = np.linalg.lstsq(A, dataSGPR.loc[masksgpr, f'{vtb} {metric}'].values, rcond=None)
            x = np.linspace(min(x_values_sgpr), max(x_values_sgpr), 100)
            y = np.polyval(coeffs, x)
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'x^{degree} Fit SGPR', line=dict(color='blue')))
        elif vtb == 'both':
            masknn = dataNN[error_column].notna()
            masksgpr = dataSGPR[error_column].notna()
            x_values_nn = dataNN.loc[masknn, error_column].values*100
            x_values_sgpr = dataSGPR.loc[masksgpr, error_column].values*100
            y_values_nn = dataNN.loc[masknn, f'validation {metric}'].values

            fig.add_trace(go.Scatter(y=y_values_nn, x=x_values_nn-0.01, mode='markers', name=f'NN Validation {metric.upper()}', marker=dict(size=8, color='red')))
            y_values_nn = dataNN.loc[mask, f'test {metric}'].values
            fig.add_trace(go.Scatter(y=y_values_nn, x=x_values_nn-0.005, mode='markers', name=f'NN Test {metric.upper()}', marker=dict(size=8, color='orange'))
            )
            y_values_sgpr = dataSGPR.loc[masksgpr, f'validation {metric}'].values
            fig.add_trace(go.Scatter(y=y_values_sgpr, x=x_values_sgpr+0.005, mode='markers', name=f'SGPR Validation {metric.upper()}', marker=dict(size=8, color='blue'))
            )
            y_values_sgpr = dataSGPR.loc[masksgpr, f'test {metric}'].values
            fig.add_trace(go.Scatter(y=y_values_sgpr, x=x_values_sgpr+0.01, mode='markers', name=f'SGPR Test {metric.upper()}', marker=dict(size=8, color='lightblue'))
            )
            fig.update_layout(title=f'Validation and Test {metric.capitalize()} for NN and SGPR', yaxis_title=f'{metric.upper()}', xaxis_title=f'{mode.capitalize()} [%]')
            '''degree = polydegree
            A=np.vander(x_values, degree +1)
            coeffs, _, _, _ = np.linalg.lstsq(A, dataNN.loc[mask, f'validation {metric}'].values, rcond=None)
            x = np.linspace(min(x_values), max(x_values), 100)
            y = np.polyval(coeffs, x)
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'x^{degree} fit for NN validation', line=dict(color='red')))
            coeffs, _, _, _ = np.linalg.lstsq(A, dataNN.loc[mask, f'test {metric}'].values, rcond=None)
            y = np.polyval(coeffs, x)
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'x^{degree} fit for NN test', line=dict(color='red')))'''

    fig.update_layout(legend=dict(x=1, y=1, xanchor="right", yanchor="top", font=dict(size=30), bordercolor="Black", borderwidth=1))
    fig.update_layout(xaxis=dict(dtick=10),yaxis=dict(range=(0,40), dtick=10)) #yaxis=dict(range=[0,40])
    fig.update_layout(
        title_font=dict(size=30),
        xaxis=dict(title_font=dict(size=30), tickfont=dict(size=30)),
        yaxis=dict(title_font=dict(size=30), tickfont=dict(size=30)),
    )
    if save:
            name=f'{metric}_vs_{mode}_for_{nnsgprboth}.html'
            fig.write_html(save + '/' + name)
    fig.show(config={'editable': True})

def plot_split(dataNNsplit, dataSGPRsplit, metric, vtb, nnsgprboth, meanad, save, target): #soll sowohl für NN als auch für SGPR funktionieren, einzeln oder zusammen, val und test 
    metric= metric.lower()
    vtb= vtb.lower()
    nnsgprboth = nnsgprboth.lower()

    dataNNsplit = dataNNsplit[dataNNsplit['target'] == target]
    dataSGPRsplit = dataSGPRsplit[dataSGPRsplit['target'] == target]

    x1=dataNNsplit['random state'].dropna()
    x2=dataSGPRsplit['random state'].dropna()
    if np.array_equal(x1,x2):
        x=x1
    else:
        raise ValueError('The random states are not the same')
    
    fig=go.Figure()
    
    if nnsgprboth == 'nn':
        data=dataNNsplit
    elif nnsgprboth == 'sgpr':
        data=dataSGPRsplit
    
    if nnsgprboth != 'both': #this section is for displaying the data of the NN or SGPR
        if vtb == 'validation' or vtb == 'test': # this section handles the validation or the test data in one plot
            mask = data[f'{vtb} {metric}'].notna() #this mask only filters for the single runs and excludes the cross validation summaries
            x_values = data.loc[mask, 'random state'].values
            y_values = data.loc[mask, f'{vtb} {metric}'].values
            fig.add_trace(go.Scatter(y=y_values, x=x_values, mode='markers', name=f'{vtb.capitalize()} {metric}', marker=dict(size=8)))
            if meanad:
                mean_data=np.mean(data[f'{vtb} {metric}'].dropna().values)
                mean_ad= np.mean(np.abs(data[f'{vtb} {metric}'].dropna().values - mean_data))
                upper_bound = np.full_like(x, mean_data + mean_ad)
                lower_bound = np.full_like(x, mean_data - mean_ad)
                fig.add_trace(go.Scatter(x=[min(x), max(x)],y=[mean_data, mean_data],mode="lines",line=dict(color="blue"), opacity=0.5, name=f"{nnsgprboth.upper()} average {vtb} {metric}"))
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x, x[::-1]]), 
                    y=np.concatenate([upper_bound, lower_bound[::-1]]), 
                    fill='toself',
                    fillcolor='rgba(0, 0, 255, 0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'Mean Absolute Deviation Range of {nnsgprboth.upper()} {vtb} {metric}'
                ))
            fig.update_layout(title=f'{vtb.capitalize()} {metric} plot', yaxis_title=f'{vtb.capitalize()} {metric}', xaxis_title='Random state of the "GroupShuffleSplit"')

        elif vtb == 'both': # this section handles both the validation and the test data in one plot
            mask = data[f'validation {metric}'].notna() #this mask only filters for the single runs and excludes the cross validation summaries
            x_values = data.loc[mask, 'random state'].values
            y_values = data.loc[mask, f'validation {metric}'].values
            y_values2 = data.loc[mask, f'test {metric}'].values
            
            fig.add_trace(go.Scatter(y=y_values, x=x_values-0.5, mode='markers', name=f'Validation {metric}', marker=dict(size=8)))
            fig.add_trace(go.Scatter(y=y_values2, x=x_values+0.5, mode='markers', name=f'Test {metric}', marker=dict(size=8)))
            if meanad:
                mean_data=np.mean(data[f'validation {metric}'].dropna().values)
                mean_ad= np.mean(np.abs(data[f'validation {metric}'].dropna().values - mean_data))
                upper_bound = np.full_like(x, mean_data + mean_ad)
                lower_bound = np.full_like(x, mean_data - mean_ad)
                fig.add_trace(go.Scatter(x=[min(x), max(x)],y=[mean_data, mean_data],mode="lines",line=dict(color="blue"), opacity=0.5, name=f"{nnsgprboth.upper()} average validation {metric}"))
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x, x[::-1]]), 
                    y=np.concatenate([upper_bound, lower_bound[::-1]]), 
                    fill='toself',
                    fillcolor='rgba(0, 0, 255, 0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'Mean Absolute Deviation Range of {nnsgprboth.upper()} validation {metric}'
                ))
                mean_data=np.mean(data[f'test {metric}'].dropna().values)
                mean_ad= np.mean(np.abs(data[f'test {metric}'].dropna().values - mean_data))
                upper_bound = np.full_like(x, mean_data + mean_ad)
                lower_bound = np.full_like(x, mean_data - mean_ad)
                fig.add_trace(go.Scatter(x=[min(x), max(x)],y=[mean_data, mean_data],mode="lines",line=dict(color="red"), opacity=0.5, name=f"{nnsgprboth.upper()} average test {metric}"))
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x, x[::-1]]), 
                    y=np.concatenate([upper_bound, lower_bound[::-1]]), 
                    fill='toself',
                    fillcolor='rgba(255, 0, 0, 0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'Mean Absolute Deviation Range of {nnsgprboth.upper()} test {metric}'
                ))
                fig.update_layout(title=f'Validation and test {metric} plot for {nnsgprboth.upper()}', yaxis_title=f'{metric.capitalize()}', xaxis_title='Random state of the "GroupShuffleSplit"')

        if save: #this section is for saving the plot to a interactive html file
            name=f'{metric}_vs_split_for_{nnsgprboth}.html'
            fig.write_html(save + '/' + name)

    elif nnsgprboth == 'both': #this section is for displaying plots for both NN and SGPR
        if vtb == 'validation' or vtb == 'test':
            mask = dataNNsplit[f'{vtb} {metric}'].notna()
            mask2 = dataSGPRsplit[f'{vtb} {metric}'].notna()
            x_values = dataNNsplit.loc[mask, 'random state'].values
            x_values2= dataSGPRsplit.loc[mask2, 'random state'].values
            y_values = dataNNsplit.loc[mask, f'{vtb} {metric}'].values
            y_values2 = dataSGPRsplit.loc[mask2, f'{vtb} {metric}'].values
            fig.add_trace(go.Scatter(y=y_values, x=x_values-0.5, mode='markers', name=f'NN {vtb} {metric}', marker=dict(size=12)))
            fig.add_trace(go.Scatter(y=y_values2, x=x_values2+0.5, mode='markers', name=f'SGPR {vtb} {metric}', marker=dict(size=12)))
            if meanad:
                mean_data=np.mean(dataNNsplit[f'{vtb} {metric}'].dropna().values)
                mean_ad= np.mean(np.abs(dataNNsplit[f'{vtb} {metric}'].dropna().values - mean_data))
                upper_bound = np.full_like(x, mean_data + mean_ad)
                lower_bound = np.full_like(x, mean_data - mean_ad)
                fig.add_trace(go.Scatter(x=[min(x), max(x)],y=[mean_data, mean_data],mode="lines",line=dict(color="blue"), opacity=0.5, name=f"NN Average {vtb} {metric}"))
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x, x[::-1]]), 
                    y=np.concatenate([upper_bound, lower_bound[::-1]]), 
                    fill='toself',
                    fillcolor='rgba(0, 0, 255, 0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'Mean Absolute Deviation Range of NN {vtb} {metric}'
                ))
                mean_data=np.mean(dataSGPRsplit[f'{vtb} {metric}'].dropna().values)
                mean_ad= np.mean(np.abs(dataSGPRsplit[f'{vtb} {metric}'].dropna().values - mean_data))
                upper_bound = np.full_like(x, mean_data + mean_ad)
                lower_bound = np.full_like(x, mean_data - mean_ad)
                fig.add_trace(go.Scatter(x=[min(x), max(x)],y=[mean_data, mean_data],mode="lines",line=dict(color="red"), opacity=0.5, name=f"SGPR Average {vtb} {metric}"))
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x, x[::-1]]), 
                    y=np.concatenate([upper_bound, lower_bound[::-1]]), 
                    fill='toself',
                    fillcolor='rgba(255, 0, 0, 0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'Mean Absolute Deviation Range of SGPR {vtb} {metric}'
                ))
            fig.update_layout(title=f'NN and SGPR {vtb.capitalize()} {metric} plot', yaxis_title=f'{vtb.capitalize()} {metric}', xaxis_title='Random state of the "GroupShuffleSplit"')
            
        elif vtb == 'both':
            mask = dataNNsplit[f'validation {metric}'].notna()
            mask2 = dataSGPRsplit[f'validation {metric}'].notna()
            x_values = dataNNsplit.loc[mask, 'random state'].values
            x_values2= dataSGPRsplit.loc[mask2, 'random state'].values
            y_values = dataNNsplit.loc[mask, f'validation {metric}'].values
            y_values2 = dataSGPRsplit.loc[mask2, f'validation {metric}'].values
            y_values3 = dataNNsplit.loc[mask, f'test {metric}'].values
            y_values4 = dataSGPRsplit.loc[mask2, f'test {metric}'].values

            fig.add_trace(go.Scatter(y=y_values, x=x_values-1, mode='markers', name=f'NN validation {metric}', marker=dict(size=8)))
            fig.add_trace(go.Scatter(y=y_values3, x=x_values-0.5, mode='markers', name=f'NN test {metric}', marker=dict(size=8)))
            fig.add_trace(go.Scatter(y=y_values2, x=x_values2+0.5, mode='markers', name=f'SGPR validation {metric}', marker=dict(size=8)))
            fig.add_trace(go.Scatter(y=y_values4, x=x_values2+1, mode='markers', name=f'SGPR test {metric}', marker=dict(size=8)))
                          
            fig.update_layout(title=f'Validation and Test {metric.capitalize()} for NN and SGPR', yaxis_title=f'{metric.capitalize()}', xaxis_title='Random state of the "GroupShuffleSplit"')
        
    fig.update_layout(legend=dict(x=1, y=1, xanchor="right", yanchor="top", font=dict(size=20), bordercolor="Black", borderwidth=1))
    fig.update_layout(xaxis=dict(range=[min(x)-2, max(x)+2], dtick=11))
    fig.update_layout(
        title_font=dict(size=30),
        xaxis=dict(title_font=dict(size=30), tickfont=dict(size=30)),
        yaxis=dict(title_font=dict(size=30), tickfont=dict(size=30)),
    )

    if save: #this section is for saving the plot to a interactive html file
            name=f'{metric}_vs_split_for_{nnsgprboth}.html'
            fig.write_html(save + '/' + name, config={"editable": True})

    fig.show(config={'editable': True})
    
def plot_comparison_nnspgr(nndata, sgprdata, metric, vtb, save, targets):
    fig = go.Figure()
    x_sgpr = 'SGPR'
    x_nn = 'NN'

    allnndata = nndata
    allsgprdata = sgprdata

    colors=['purple','green','orange']

    for target, color in zip (targets, colors):
        targetfilternn = allnndata['target'] == target
        targetfiltersgpr = allsgprdata['target'] == target
        nndata = allnndata.loc[targetfilternn]
        sgprdata = allsgprdata.loc[targetfiltersgpr]
        targetname = split_camel_case(target)

        if target == 'WristAngle':
            x_nn = 'NN W'
            x_sgpr = 'SGPR W'
        elif target == 'ElbowAngle':
            x_nn = 'NN E'
            x_sgpr = 'SGPR E'
        elif target == 'ShoulderAngleZ':
            x_nn = 'NN S'
            x_sgpr = 'SGPR S'
        #x_nn = f'NN {targetname}'
        #x_sgpr = f'SGPR {targetname}'

        if vtb != 'both':
            masknn = nndata[f'{vtb} {metric}'].notna()
            masksgpr = sgprdata[f'{vtb} {metric}'].notna()

            y_sgpr = sgprdata.loc[masksgpr, f'{vtb} {metric}'].values
            y_nn = nndata.loc[masknn, f'{vtb} {metric}'].values

            fig.add_trace(go.Box(y=y_sgpr, x=[x_sgpr]*len(y_sgpr), name=f'SGPR {targetname} {vtb.capitalize()} {metric.upper()}', boxpoints='outliers', jitter=0.3, pointpos=0,marker=dict(size=10),showlegend=False, marker_color=color,line=dict(width=4)))
            fig.add_trace(go.Box(y=y_nn, x=[x_nn]*len(y_nn), name=f'NN {targetname} {vtb.capitalize()} {metric.upper()}', boxpoints='outliers', jitter=0.3, pointpos=0,marker=dict(size=10),showlegend=False, marker_color=color,line=dict(width=4)))
            
        
        elif vtb == 'both':
            masksgpr = sgprdata[f'validation {metric}'].notna()
            y_sgpr = sgprdata.loc[masksgpr, f'validation {metric}'].values
            fig.add_trace(go.Box(y=y_sgpr, x=[x_sgpr+' Validation']*len(y_sgpr), name=f'SGPR {targetname} Validation {metric.upper()}', boxpoints='all', jitter=0.3, pointpos=0, boxmean='sd',showlegend=False))

            masksgpr = sgprdata[f'test {metric}'].notna()
            y_sgpr = sgprdata.loc[masksgpr, f'test {metric}'].values
            fig.add_trace(go.Box(y=y_sgpr, x=[x_sgpr+' Test']*len(y_sgpr), name=f'SGPR {targetname} Test {metric.upper()}', boxpoints='all', jitter=0.3, pointpos=0, boxmean='sd',showlegend=False))

            masknn = nndata[f'validation {metric}'].notna()
            y_nn = nndata.loc[masknn, f'validation {metric}'].values
            fig.add_trace(go.Box(y=y_nn, x=[x_nn+' Validation']*len(y_nn), name=f'NN {targetname} Validation {metric.upper()}', boxpoints='all', jitter=0.3, pointpos=0, boxmean='sd',showlegend=False))

            masknn = nndata[f'test {metric}'].notna()
            y_nn = nndata.loc[masknn, f'test {metric}'].values
            fig.add_trace(go.Box(y=y_nn, x=[x_nn+' Test']*len(y_nn), name=f'NN {targetname} Test {metric.upper()}', boxpoints='all', jitter=0.3, pointpos=0, boxmean='sd',showlegend=False))

    #fig.update_layout(legend=dict(x=1, y=1, xanchor="right", yanchor="top", font=dict(size=20), bordercolor="Black", borderwidth=1))
    fig.update_layout(
        boxgroupgap=0.1,
        boxgap=0,
        title_font=dict(size=30),
        xaxis=dict(title_font=dict(size=30), tickfont=dict(size=20)),
        yaxis=dict(title_font=dict(size=30), tickfont=dict(size=20)),
    )

    if vtb !='both':
        fig.update_layout(title=f'{vtb.capitalize()} {metric} comparison', yaxis_title=f'{vtb.capitalize()} {metric.upper()}', font=dict(size=30))
    if vtb =='both':
        fig.update_layout(title=f'Validation and Test {metric} comparison', yaxis_title=f'{metric.upper()}', font=dict(size=30))
    if save: #this section is for saving the plot to a interactive html file
            if vtb == 'both':
                vtb = 'ValidationTest'
            name=f'{vtb}_{metric}_NNvsSGPR.html'
            fig.write_html(save + '/' + name, config={"editable": True})

    #if targets is list:
    image = Image.open('arm_3.png')
    fig.add_layout_image(
    dict(
        source=image,
        xref="x",
        yref="y",
        x= -0.6,
        y=20,
        sizex=6.5,
        sizey=20,
        sizing="stretch",
        opacity=0.25,
        layer="below",)
    )
    fig.update_layout(template="plotly_white")
    fig.update_layout(yaxis=dict(range=[0, 20], dtick=5))
    #fig.update_layout(height=1000)

    fig.show(config={'editable': True})
    