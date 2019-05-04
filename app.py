# Import Libraries
###################################################################

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly as py
py.offline.init_notebook_mode()
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
import matplotlib.pyplot as plt
import matplotlib
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
import seaborn as sns 
import numpy as np
import pandas as pd
import numpy as np
import random as rnd
import re
import io
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score , average_precision_score
from sklearn.metrics import precision_score, precision_recall_curve
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS

# from mpl_toolkits.basemap import Basemap
from matplotlib import animation, rc
from IPython.display import HTML

import warnings
warnings.filterwarnings('ignore')

import base64
from IPython.display import HTML, display
import warnings
warnings.filterwarnings('ignore')
from scipy.misc import imread
import codecs
from subprocess import check_output
import folium 
from folium import plugins
from folium.plugins import HeatMap

import nltk

nltk.download('punkt')

nltk.download('stopwords')
import altair as alt

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

import cufflinks as cf

alt.renderers.enable('notebook')
# Read data
###################################################################

terror = pd.read_csv('globalterrorismdb_0718dist.csv', encoding='ISO-8859-1')

terror.columns

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
terror_filter = terror[terror['targtype1_txt'] == "Private Citizens & Property"]
terror_count = terror_filter.groupby(['country_txt'])['targtype1_txt'].count()
countries = pd.DataFrame({'country':terror_count.index,'number':terror_count.values })

yearly_killed = terror.groupby(['iyear'])['nkill'].sum().reset_index()
yearly_wounded = terror.groupby(['iyear'])['nwound'].sum().reset_index()

trace = go.Bar(
	x = yearly_killed['iyear'],
    y = yearly_killed['nkill'],
    name = 'Killed',
    marker = dict(
        color = 'red'
    )
	)

trace1 = go.Bar(
    x = yearly_wounded['iyear'],
    y = yearly_wounded['nwound'],
    name = 'Wounded',
    marker = dict(
        color = 'orange',
        opacity = 0.5
    )
)

terror_df = pd.read_csv('globalterrorismdb_0718dist.csv', encoding='ISO-8859-1')

terror_df['casualities'] = terror_df['nkill'] + terror_df['nwound']

terror_bubble_df = terror_df[(terror_df['gname'] != 'Unknown') & (terror_df['casualities'] > 50)]

terror_bubble_df = terror_bubble_df.sort_values(['region_txt', 'country_txt'])

terror_bubble_df.isnull().sum()

terror_bubble_df = terror_bubble_df.drop(['latitude','longitude','summary','motive','target1'],axis=1)

terror_bubble_df = terror_bubble_df.dropna(subset=['city'])

terror_bubble_df.isnull().sum()

hover_text = []
for index, row in terror_bubble_df.iterrows():
    hover_text.append(('City: {city}<br>'+
                      'gname: {group}<br>'+
                      'casualities: {casualities}<br>'+
                      'iyear: {year}').format(city=row['city'],
                                            group=row['gname'],
                                            casualities=row['casualities'],
                                            year=row['iyear']))
terror_bubble_df['text'] = hover_text

trace_0 = go.Scatter(
    x=terror_bubble_df['iyear'][terror_bubble_df['country_txt'] == 'Iraq'],
    y=terror_bubble_df['casualities'][terror_bubble_df['country_txt'] == 'Iraq'],
    mode='markers',
    name='Iraq',
    text=terror_bubble_df['text'][terror_bubble_df['country_txt'] == 'Iraq'],
    marker=dict(
        symbol='circle',
        sizemode='area',
        size=terror_bubble_df['casualities'][terror_bubble_df['country_txt'] == 'Iraq'],
        line=dict(
            width=2
        ),
    )
)
trace_1 = go.Scatter(
    x=terror_bubble_df['iyear'][terror_bubble_df['country_txt'] == 'Pakistan'],
    y=terror_bubble_df['casualities'][terror_bubble_df['country_txt'] == 'Pakistan'],
    mode='markers',
    name='Pakistan',
    text=terror_bubble_df['text'][terror_bubble_df['country_txt'] == 'Pakistan'],
    marker=dict(
        symbol='circle',
        sizemode='area',
        size=terror_bubble_df['casualities'][terror_bubble_df['country_txt'] == 'Pakistan'],
        line=dict(
            width=2
        ),
    )
)
trace_2 = go.Scatter(
    x=terror_bubble_df['iyear'][terror_bubble_df['country_txt'] == 'Afghanistan'],
    y=terror_bubble_df['casualities'][terror_bubble_df['country_txt'] == 'Afghanistan'],
    mode='markers',
    name='Afghanistan',
    text=terror_bubble_df['text'][terror_bubble_df['country_txt'] == 'Afghanistan'],
    marker=dict(
        symbol='circle',
        sizemode='area',
        size=terror_bubble_df['casualities'][terror_bubble_df['country_txt'] == 'Afghanistan'],
        line=dict(
            width=2
        ),
    )
)
trace_3 = go.Scatter(
    x=terror_bubble_df['iyear'][terror_bubble_df['country_txt'] == 'India'],
    y=terror_bubble_df['casualities'][terror_bubble_df['country_txt'] == 'India'],
    mode='markers',
    name='India',
    text=terror_bubble_df['text'][terror_bubble_df['country_txt'] == 'India'],
    marker=dict(
        symbol='circle',
        sizemode='area',
        size=terror_bubble_df['casualities'][terror_bubble_df['country_txt'] == 'India'],
        line=dict(
            width=2
        ),
    )
)

terror['nwound'] = terror['nwound'].fillna(0).astype(int)
terror['nkill'] = terror['nkill'].fillna(0).astype(int)
terror['casualities'] = terror['nkill'] + terror['nwound']

terror = terror.sort_values(by='casualities',ascending=False)[:30]


map1=terror.pivot_table(index='country_txt',columns='iyear',values='casualities')
map1.fillna(0,inplace=True)


app.layout = html.Div(children=[
    html.H1(children='Global Terrorism'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='graph1',
     figure={'data': 
    [dict(type='choropleth',
    locations=countries['country'],
    locationmode='country names',
    z=countries['number'],
    text=countries['country'],
    colorscale='Jet',
    reversescale=True,
    marker=dict(line=dict(color='rgb(180,180,180)')),
    colorbar = {'title': 'No of incidents'},
	)],
    'layout': (go.Layout(dict(
    title='Countries with most terror attacks',
    geo=dict(showframe=False, showcoastlines=True, projection=dict(type='mercator')),
    width=1000,
    height=1000,))),
	
      }
    ),

	dcc.Graph(
        id='graph2',
        figure={
        'data':[trace,trace1],
		'layout': (go.Layout(
		title = 'Yearly Casualities',
		xaxis = dict(
		title = 'Year'
    ),
    barmode = 'stack'
  )),
	
      }
    ),
	
	dcc.Graph(
        id='graph3',
        figure={
        'data': [trace_0, trace_1, trace_2, trace_3],
		'layout': (go.Layout(
         title = 'Worst Terrorism Countries',
         xaxis = dict(
             title = 'Year',
             #type = 'log',
             range = [1976,2016],
             tickmode = 'auto',
             nticks = 30,
             showline = True,
             showgrid = False
             ),
         paper_bgcolor='rgb(243, 243, 243)',
         plot_bgcolor='rgb(243, 243, 243)',
         )),
	
      }
   ),
	
 ], )


if __name__ == '__main__':
    app.run_server(debug=True)