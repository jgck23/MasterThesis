import pandas as pd
import plotly.graph_objects as go
import preprocessing.Studie_Sebastian.seb_xsens as seb_xsens
import colorcet as cc

folder_path_Xsens = ['/Users/jacob/Documents/Microsoft Visual Studio Code Projects/Masterarbeit/Data/Xsens Data/Studie_Sebastian/Loewe29']

data, xs_frames, header = seb_xsens.main(folder_path_Xsens)
data=data.values

column=2
name='Loewe29'

drilling_height=data[:, 3]
colors=cc.glasbey

count=0
fig=go.Figure()
for i, color in zip(range(len(xs_frames)),colors):
    x=data[count:count+xs_frames[i],column]
    y=drilling_height[count:count+xs_frames[i]]
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color=color), name=f'Trial {i}'))
    count += xs_frames[i]

fig.update_layout(title=f'{name} {header[column]} vs Drilling Height', xaxis_title=f'{header[column]} Angle', yaxis_title='Drilling Height')
fig.write_html(f'{name}_{header[column]}_vs_Drilling_Height.html')
fig.show()


