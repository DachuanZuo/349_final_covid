# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '..')

from utils import data
import os
import sklearn
import numpy as np
from sklearn.neighbors import (
    KNeighborsClassifier,
    DistanceMetric
)
import json
from shapely.geometry import Point
import matplotlib.pyplot as plt
import geopandas as gpd
import geoplot as gplt
import mapclassify
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def kmeans_usa_states():
    from sklearn.cluster import KMeans
        # ------------ HYPERPARAMETERS -------------
    BASE_PATH = './COVID-19/csse_covid_19_data/'
    MIN_CASES = 1000
    # ------------------------------------------
    
    confirmed = os.path.join(
        BASE_PATH, 
        'csse_covid_19_time_series',
        'time_series_covid19_confirmed_US.csv')
    confirmed = data.load_csv_data(confirmed)
    state_feature = {}
    states = list(set(confirmed["Province_State"]))
    # print (confirmed.iloc[232,11:confirmed.shape[0]-11])
    
    for idx in range(confirmed.shape[0]):
        if confirmed["Province_State"][idx] in list(state_feature.keys()):
            state_feature[confirmed["Province_State"][idx]] += confirmed.iloc[idx,11:confirmed.shape[0]-11]
        else:
            state_feature[confirmed["Province_State"][idx]] = confirmed.iloc[idx,11:confirmed.shape[0]-11]
    features = np.asarray(list(state_feature.values()))
    targets = np.asarray(list(state_feature.keys()))
            
    contiguous_usa = gpd.read_file(gplt.datasets.get_path('contiguous_usa'))
    gplt.polyplot(contiguous_usa)
     
    contiguous_usa["class"] = np.full(contiguous_usa.shape[0], -1)            
     
    
    kmeans = KMeans(n_clusters=4, max_iter=1000)
    kmeans.fit(features, targets)
    state_classes = kmeans.predict(features)
     
    state_classes = dict((targets[idx], state_classes[idx]) for idx in range(targets.shape[0]))
    for idx in range(contiguous_usa.shape[0]):
        if contiguous_usa["state"][idx] in targets:
            contiguous_usa["class"][idx] = state_classes[contiguous_usa["state"][idx]]
    df_means = contiguous_usa[["state","class"]]    
             
             
             
    c = mcolors.ColorConverter().to_rgb
    rvb = make_colormap(
    [ c('orange'), c('violet'), 0.33, c('violet'), 0.50, c('blue'), 0.66, c('red'),c('green')])
     
    scheme = mapclassify.NaturalBreaks(list(df_means["class"]))
    gplt.choropleth(
    contiguous_usa, hue=list(df_means["class"]), scheme=scheme, 
      figsize=(8, 4),legend=False,cmap = 'binary'
    )
           
  
     


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


if __name__ == "__main__":
 
    kmeans_usa_states()
     