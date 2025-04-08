import json
import plotly.graph_objects as go

# Load the JSON file
with open('avsp.json', 'r') as file:
    data = json.load(file)

# Create your original figure from the JSON data.
fig = go.Figure(data=data['data'], layout=data['layout'])

# Define the range and compute y-values for the filled area between y = x + 5.79 and y = x - 5.79.
x_values = list(range(41, 131))
y1_values = [x + 5.79 for x in x_values]
y2_values = [x - 5.79 for x in x_values]

# Define your background traces.
background_traces = [
    go.Scatter(
        x=x_values, 
        y=y1_values, 
        mode='lines', 
        line=dict(color='rgba(0,0,0,0)', width=0),
        showlegend=False,
    ),
    go.Scatter(
        x=x_values, 
        y=y2_values, 
        mode='lines', 
        line=dict(color='rgba(0,0,0,0)', width=0),
        fill='tonexty', 
        fillcolor='rgba(173, 216, 230, 0.7)', 
        showlegend=False,
    )
]

# Define the range and compute y-values for the filled area between y = 90 and y = 70.
y3_values = [90] * len(x_values)
y4_values = [70] * len(x_values)

# Define the new background trace for the light orange area.
orange_background_trace = [
    go.Scatter(
        x=x_values,
        y=y3_values,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)', width=0),
        showlegend=False,
    ),
    go.Scatter(
        x=x_values,
        y=y4_values,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)', width=0),
        fill='tonexty',
        fillgradient=dict(
                type="vertical",
                colorscale=[[0.0, 'rgba(255, 165, 0, 0.3)'], [0.4, 'rgba(255, 255, 255, 0.0)'],[0.6, 'rgba(255, 255, 255, 0.0)'], [1.0, 'rgba(255, 165, 0, 0.3)']],
        ),
        #fillcolor='rgba(211, 211, 211, 0.35)',  # Light grey with 0.15 opacity
        showlegend=False,
    )
]

# Add the new orange background trace to the background traces.
background_traces.extend(orange_background_trace)

# Define the range and compute y-values for the filled area between x = 70 and x = 90.
x_red_values = list(range(70, 91))
y5_values = [max(y1_values)] * len(x_red_values)  # Top boundary (same as y1_values max)
y6_values = [min(y2_values)] * len(x_red_values)  # Bottom boundary (same as y2_values min)

# Define the new background trace for the light red area.
red_background_trace = [
    go.Scatter(
        x=x_red_values,
        y=y5_values,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)', width=0),
        showlegend=False,
    ),
    go.Scatter(
        x=x_red_values,
        y=y6_values,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)', width=0),
        fill='tonexty',
        fillgradient=dict(
                type="horizontal",
                colorscale=[[0.0, 'rgba(255, 99, 71, 0.3)'], [0.4, 'rgba(255, 255, 255, 0.0)'],[0.6, 'rgba(255, 255, 255, 0.0)'], [1.0, 'rgba(255, 99, 71, 0.3)']],
        ),
        #fillcolor='rgba(255, 99, 71, 0.15)',  # Light red with 0.2 opacity
        showlegend=False,
    )
]

# Add the new red background trace to the background traces.
background_traces.extend(red_background_trace)

# Combine the new background traces with the original ones,
# ensuring the background traces come first.
combined_traces = background_traces + list(fig.data)

# Create a new figure with the combined traces and the existing layout.
fig_updated = go.Figure(data=combined_traces, layout=fig.layout)

# Optionally, update additional layout settings as before.
fig_updated.update_layout(showlegend=False)
fig_updated.update_layout(
    title_font=dict(size=30),
    xaxis=dict(title_font=dict(size=30), tickfont=dict(size=30), dtick=10, range=[40, 131]),
    yaxis=dict(title_font=dict(size=30), tickfont=dict(size=30), dtick=10, range=[40, 131]),
)
fig_updated.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig_updated.update_xaxes(showgrid=True, gridcolor='lightgrey')
fig_updated.update_yaxes(showgrid=True, gridcolor='lightgrey', zerolinecolor='lightgrey')

# Show the new figure
fig_updated.show(config={'editable': True})