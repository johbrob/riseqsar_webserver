import pandas as pd
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from plotly.graph_objs.histogram import XBins
import json
import math


def create_plot(preds):
    s = pd.Series(preds)
    s.name = 'Predictions'

    data = make_subplots(rows=3, cols=1, row_heights=[0.25, 0.25, 0.5], shared_xaxes=True)

    bin_width = 0.025
    # here you can choose your rounding method, I've chosen math.ceil
    nbins = math.ceil((s.max() - s.min()) / bin_width)
    fig1 = px.histogram(s, x="Predictions", marginal='rug', nbins=nbins, histnorm='probability')
    fig2 = px.histogram(s, x="Predictions", marginal='violin', nbins=nbins, histnorm='probability')
    hist_trace = go.Histogram(histnorm="probability",
                              x=s,
                              xbins = XBins(start=0, end=1, size=bin_width),
                              showlegend=False)
    hist_trace.marker = fig1.data[0].marker

    ## violin plot components
    data.add_trace(fig2.data[1], row=1, col=1)

    ## rug plot components
    data.add_trace(fig1.data[1], row=2, col=1)

    ## histogram component
    # data.add_trace(fig1.data[0], row=3, col=1)
    data.add_trace(hist_trace, row=3, col=1)

    # data['layout']['xaxis']['title'] = 'Violin Plot'
    # data['layout']['xaxis2']['title'] = 'Rugplot'
    data['layout']['xaxis3']['title'] = 'Prediction Probability'
    data['layout']['yaxis3']['title'] = 'Density'
    data.update_layout(xaxis=dict(range=[ 0, 1]))
    data.update_layout(xaxis2=dict(range=[0, 1]))
    data.update_layout(xaxis3=dict(range=[0, 1], showgrid=True), yaxis3=dict(range=[0, 1]))


    #data.update_layout(title=f"SMILES: {'CCCC'}<br>Endpoint: {'bbbp'}<br>Model: {'Logistic Regression 20022-202-220-1010'}<br>")

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

if __name__ == '__main__':
    create_plot([0.90, 0.901, 0.902, 0.903])