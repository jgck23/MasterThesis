import json

import plotly.graph_objects as go

# Load the JSON file
with open('avsp.json', 'r') as file:
    data = json.load(file)

# Create a Plotly figure from the JSON data
fig = go.Figure(data=data['data'], layout=data['layout'])
fig.update_layout(showlegend=False)
fig.update_layout(
        title_font=dict(size=30),
        xaxis=dict(title_font=dict(size=30), tickfont=dict(size=30)),
        yaxis=dict(title_font=dict(size=30), tickfont=dict(size=30)),
    )

# Show the figure
fig.show(config={'editable': True})