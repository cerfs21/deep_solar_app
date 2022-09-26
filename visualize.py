# visualize v1.4:
#   English comments

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Plot a bar graph to compare installed base and model prediction
def make_figure_from_prediction(installed, target):
    x=["Base installée","Prédiction"]
    y=[installed,target]
    fig = go.Figure([go.Bar(x=x, y=y , text=y, marker_color=['green', 'yellow'])])
    fig.update_layout(
        font_color='black',
        xaxis_tickfont_size=16,
        yaxis=dict(
            title='Surface en m²',
            titlefont_size=16,
            tickfont_size=14),
    )
    return fig
