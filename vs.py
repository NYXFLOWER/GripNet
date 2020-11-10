import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import torch

import numpy as np

ccc = {'GCN': '0,150,80',
       'GAT': '230, 81, 78',
       'RGCN': '149, 121, 212',
       'GripNet': '0, 0, 0',
       'ComplEx': '230, 194, 112',
       'DistMult': '20, 234, 245',
       'RotatE': '182, 48, 191',
       'TransE': '20, 121, 245',
       'GripNet-l': '0, 0, 0'}
path = ['out_baseline/pose-0-baselines/0-ComplEx-34-record.pt',
        'out_baseline/pose-0-baselines/0-DistMult-84-record.pt',
        'out_baseline/pose-0-baselines/0-RGCN-76-record.pt',
        'out_baseline/pose-0-baselines/0-RotatE-32-record.pt',
        'out_baseline/pose-0-baselines/0-TransE-135-record.pt',
        'out_gripnet/pose-nneg-0/0-[32, 16, 16]-[16, 32]-[48, 32]-record.pt']
path2 = ['out_baseline/pose-2-baselines/2-ComplEx-26-record.pt',
         'out_baseline/pose-2-baselines/2-DistMult-66-record.pt',
         'out_baseline/pose-2-baselines/4-RGCN-58-record.pt',
         'out_baseline/pose-2-baselines/2-RotatE-24-record.pt',
         'out_baseline/pose-2-baselines/2-TransE-106-record.pt',
         'out_gripnet/pose-nneg-2/2-[32, 16, 16]-[16, 32]-[48, 32]-record.pt']
m = ['ComplEx', 'DistMult', 'RGCN', 'RotatE', 'TransE', 'GripNet-l']
x = list(range(100))

# ######################## figure 1 ########################
fig = make_subplots(rows=1, cols=1, start_cell="bottom-left")
eva = 2     # ap@50
for i in range(6):
    n = m[i]
    record = torch.load(path[i])['test_out']
    y = [record[i][eva] for i in range(100)]
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        line=dict(color='rgb({})'.format(ccc[n])),
        mode='lines',
        name=n
    ), row=1, col=1)

fig.update_layout(height=380, width=450)
fig.update_xaxes(tickvals=list(range(0, 120, 20)), range=[0, 100])
fig.update_yaxes(range=[0.45, 0.95])
fig.write_image("f1.png")

# ######################## figure 1+Pose2 ########################
fig = make_subplots(rows=1, cols=1, start_cell="bottom-left")
xx = [8, 16, 24, 32]
m = ['TransE', 'RGCN', 'DistMult', 'GripNet-l']
auroc = [[0.591, 0.652, 0.669, 0.674],
         [0.888, 0.880, 0.888, 0.846],
         [0.498, 0.503, 0.517, 0.800],
         [0.921, 0.918, 0.917, 0.918]]
for i in range(4):
    n = m[i]
    y = auroc[i]
    fig.add_trace(go.Scatter(
        x=xx,
        y=y,
        line=dict(color='rgb({})'.format(ccc[n])),
        mode='lines',
        name=n
    ), row=1, col=1)

m = ['RotatE', 'ComplEx']
x = [8, 16, 24]
auroc = [[0.696, 0.796, 0.851], [0.501, 0.488, 0.469]]
for i in range(2):
    n = m[i]
    y = auroc[i]
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        line=dict(color='rgb({})'.format(ccc[n])),
        mode='lines',
        name=n
    ), row=1, col=1)

fig.update_layout(height=380, width=450)
fig.update_xaxes(tickvals=xx, range=[6, 34])
fig.update_yaxes(range=[0.45, 0.95])
fig.write_image("f1.png")

# ######################## figure 1+Pose2 ########################
fig = make_subplots(rows=1, cols=1, start_cell="bottom-left")
xx = [8, 16, 24, 32]
m = ['TransE', 'RGCN', 'DistMult', 'GripNet-l']
auroc = [[2.81, 3.95, 5.41, 9.64],
         [19.92, 23.35, 26.93, 26.63],
         [4.16, 7.74, 11.31, 14.89],
         [21.18, 23.91, 0.917, 0.918]]
for i in range(4):
    n = m[i]
    y = auroc[i]
    fig.add_trace(go.Scatter(
        x=xx,
        y=y,
        line=dict(color='rgb({})'.format(ccc[n])),
        mode='lines',
        name=n
    ), row=1, col=1)

m = ['RotatE', 'ComplEx']
x = [8, 16, 24]
auroc = [[0.696, 0.796, 0.851], [0.501, 0.488, 0.469]]
for i in range(2):
    n = m[i]
    y = auroc[i]
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        line=dict(color='rgb({})'.format(ccc[n])),
        mode='lines',
        name=n
    ), row=1, col=1)

fig.update_layout(height=380, width=450)
fig.update_xaxes(tickvals=xx, range=[6, 34])
fig.update_yaxes(range=[0.45, 0.95])
fig.write_image("mem.png")

# ######################## figure 1 == pose-0 ########################
fig = make_subplots(rows=1, cols=1, start_cell="bottom-left")
x = [8, 16, 24, 32]
auroc = [[0.4998, 0.4939, 0.5656, 0.6107],    # ComplEx
         [0.5003, 0.4967, 0.5304, 0.6159],    # DistMult
         [0.9066, 0.9067, 0.9065, 0.9055],    # RGCN
         [0.8102, 0.8704, 0.8870, 0.8907],    # RotatE
         [0.6881, 0.7526, 0.7615, 0.7759],    # TransE
         [0.9189, 0.9193, 0.9217, 0.9198]]    # GripNet

for i in range(6):
    n = m[i]
    y = auroc[i]
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        line=dict(color='rgb({})'.format(ccc[n])),
        mode='lines',
        name=n
    ), row=1, col=1)

fig.update_layout(height=380, width=450)
fig.update_xaxes(tickvals=x, range=[6, 34])
fig.update_yaxes(range=[0.45, 0.95])
fig.write_image("f1.png")

# ######################## figure 2 ########################
fig = make_subplots(rows=1, cols=1, start_cell="bottom-left")
m = ['ComplEx', 'DistMult', 'RGCN', 'RotatE', 'TransE', 'GripNet-l']
t = [29.2, 29, 47.5, 29.7, 30, 40]
t2 = [30.4, 29.7, 60.3, 30.6, 30.8, 40.3]
eva = 1     # ap@50
for i in range(6):
    n = m[i]
    xx = np.array(x) * t[i]
    record = torch.load(path2[i])['test_out']
    y = [record[i][eva] for i in range(100) if xx[i] < 3001]
    fig.add_trace(go.Scatter(
        x=xx.tolist(),
        y=y,
        line=dict(color='rgb({})'.format(ccc[n])),
        mode='lines',
        name=n
    ), row=1, col=1)

fig.update_layout(height=380, width=450)
fig.update_yaxes(range=[0.45, 0.95])
fig.write_image("f22.png")


# ######################## figure 3 ########################
row_head = ['GCN', 'GAT', 'RGCN', 'GripNet']
col_head = [1, 2, 3, 4]
mean = np.array([[0.3094, 0.4328, 0.5081, 0.5632],
                 [0.3408, 0.3652, 0.4644, 0.5057],
                 [0.3002, 0.4542, 0.4979, 0.5643],
                 [0.3001, 0.4641, 0.5638, 0.5924]])
error = np.array([[0.0090, 0.0098, 0.0108, 0.0092],
                  [0.0042, 0.0049, 0.0039, 0.0042],
                  [0.0042, 0.0048, 0.0062, 0.0058],
                  [0.0016, 0.0018, 0.0021, 0.0024]])
upper = mean + error
lower = mean - error
fig = make_subplots(rows=1, cols=1, start_cell="bottom-left")
for i in range(4):
    n, u, l, m = row_head[i], upper[i, :].tolist(), lower[i, :].tolist(), mean[i, :].tolist()
    fig.add_trace(go.Scatter(
        x=col_head,
        y=m,
        line=dict(color='rgb({})'.format(ccc[n])),
        mode='lines',
        name=n
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=col_head+col_head[::-1],  # x, then x reversed
        y=u+l[::-1],  # upper, then lower reversed
        fill='toself',
        fillcolor='rgba({},0.2)'.format(ccc[n]),
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ), row=1, col=1)
fig.update_layout(height=380, width=450)
fig.write_image("f3.png")

# ######################## figure 4 ########################
col_head = [14989, 354961, 457504, 1100400]
fig = make_subplots(rows=1, cols=1, start_cell="bottom-left")
for i in range(4):
    n, u, l, m = row_head[i], upper[i, :].tolist(), lower[i, :].tolist(), mean[i, :].tolist()
    fig.add_trace(go.Scatter(
        x=col_head,
        y=m,
        line=dict(color='rgb({})'.format(ccc[n])),
        mode='lines',
        name=n
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=col_head+col_head[::-1],  # x, then x reversed
        y=u+l[::-1],  # upper, then lower reversed
        fill='toself',
        fillcolor='rgba({},0.2)'.format(ccc[n]),
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ), row=1, col=1)

fig.update_layout(height=380, width=450,
                  xaxis={'type': 'log'})
fig.show()
fig.write_image("f4.png")