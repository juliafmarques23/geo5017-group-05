from plotly import express as px
from plotly import graph_objects as go
import pandas as pd

def plot_trajectory_simple(positions):
    '''Plots 3D line graph'''
    for position in positions:
        pos_x = position[0]
        pos_y = position[1]
        pos_z = position[2]
    fig = px.line_3d(x=pos_x, y=pos_y, z=pos_z)
    fig.show()

def plot_trajectory(positions):
    '''Plots 3D line graph with markers on the computed positions'''
    #create lists with all the x, y, z values
    pos_x = []
    pos_y = []
    pos_z = []
    for position in positions:
        pos_x.append(position[0])
        pos_y.append(position[1])
        pos_z.append(position[2])
    #plot graph with points and lines
    fig = go.Figure(data=go.Scatter3d(
        x=pos_x, y=pos_y, z=pos_z,
        marker=dict(size=3,color='darkblue', symbol='x'),
        line=dict(color='darkblue', width=2)))
    fig.show()

#given drone positions
positions = [(2, 0, 1), (1.08, 1.68, 2.38),
                 (-0.83, 1.82, 2.49), (-1.97, 0.28, 2.15),
                 (-1.31, -1.51, 2.59), (0.57, -1.91, 4.32)]

plot_trajectory(positions)
