"""
Experiment summary
------------------
Treat each province/state in a country cases over time
as a vector, do a simple K-Nearest Neighbor between 
countries. What country has the most similar trajectory
to a given country?
"""

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




def kmeans_covid_pattern(means):
           # ------------ HYPERPARAMETERS -------------
       BASE_PATH = './COVID-19/csse_covid_19_data/'
       N_NEIGHBORS = 5
       MIN_CASES = 1000
       NORMALIZE = True
       # ------------------------------------------
       
       confirmed = os.path.join(
           BASE_PATH, 
           'csse_covid_19_time_series',
           'time_series_covid19_confirmed_global.csv')
       confirmed = data.load_csv_data(confirmed)
       features = []
       targets = []
       
       for val in np.unique(confirmed["Country/Region"]):
           df = data.filter_by_attribute(
               confirmed, "Country/Region", val)
           cases, labels = data.get_cases_chronologically(df)
           features.append(cases)
           targets.append(labels)
       
       features = np.concatenate(features, axis=0)
       targets = np.concatenate(targets, axis=0)
        
       Country_Region = list(set(confirmed["Country/Region"]))
       
       country_features = {}
       for t in range(features.shape[0]):
           checked = []
           if confirmed["Country/Region"][t] in checked:
               country_features[confirmed["Country/Region"][t]] += features[t]
           else:
               country_features[confirmed["Country/Region"][t]] = features[t]
   
       country_features_values = np.asarray(list(country_features.values()))    
       country_targets = sorted(list(set(confirmed["Country/Region"])))
       from sklearn.cluster import KMeans
       kmeans = KMeans(n_clusters=means, max_iter=1000)
       kmeans.fit(country_features_values, country_targets)
       kmeans_predictions = kmeans.predict(country_features_values)
       

       world = gpd.read_file(
           gpd.datasets.get_path('naturalearth_lowres')
       )
       boroughs = gpd.read_file(
           gplt.datasets.get_path('nyc_boroughs')
       )
       collisions = gpd.read_file(
           gplt.datasets.get_path('nyc_injurious_collisions')
       )
       
       
       fig,ax = plt.subplots()

   
       world["class"] = np.full(world.shape[0],-1)
       for idx in world.index:
           if world["name"][idx] in list(country_features.keys()):
               world["class"][idx] = kmeans_predictions[idx]
       
               
       df_means = world[["name","class"]]
       world, country_means, country_index, df_means = kmeans_covid_pattern(5)

       scheme = mapclassify.NaturalBreaks(list(df_means["class"]))
       gplt.choropleth(
        world, hue=list(df_means["class"]), scheme=scheme, 
          figsize=(8, 4),legend=False,cmap = "binary"
       )
        
       
       # return world, country_means, country_means_index, df_means

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
    kmeans_covid_pattern(4)
#     c = mcolors.ColorConverter().to_rgb
#     rvb = make_colormap(
#     [c("white"), c('red'), c('violet'), 0.33, c('violet'), 0.50, c('blue'), 0.66, c('green'),c('orange')])
    
#     world, country_means, country_index, df_means = kmeans_covid_pattern(5)

#     scheme = mapclassify.NaturalBreaks(list(df_means["class"]))
#     gplt.choropleth(
#     world, hue=list(df_means["class"]), scheme=scheme, 
#       figsize=(8, 4),legend=False,cmap = "binary"
# )
    
    
    
    # country_features_sub = dict((country, feature) for country, feature in country_features.items() if country in country_4_means[0])
    # country_features_values_sub = np.asarray(list(country_features_sub.values()))    
    # country_targets_sub = sorted(list(country_features_sub.keys()))
    # kmeans_sub = KMeans(n_clusters=10, max_iter=300)
    # kmeans_sub.fit(country_features_values_sub, country_targets_sub)
    # kmeans_predictions_sub = kmeans_sub.predict(country_features_values_sub)
    # country_means_sub = {}
    # country_means_index_sub = {}
    # dict_means_sub = {}
    # for p in range(len(kmeans_predictions_sub)):
    #     dict_means_sub[country_targets_sub[p]] = kmeans_predictions_sub[p]
    #     if kmeans_predictions_sub[p] in list(country_means_sub.keys()):
    #         country_means_sub[kmeans_predictions_sub[p]].append(country_targets_sub[p])
    #         country_means_index_sub[kmeans_predictions_sub[p]].append(p)
    #     else:
    #         country_means_sub[kmeans_predictions_sub[p]] = [country_targets_sub[p]]
    #         country_means_index_sub[kmeans_predictions_sub[p]] = [p]
    
    
    # # ------------ HYPERPARAMETERS -------------
    # BASE_PATH = './COVID-19/csse_covid_19_data/'
    # N_NEIGHBORS = 5
    # MIN_CASES = 1000
    # NORMALIZE = True
    # # ------------------------------------------
    
    # confirmed = os.path.join(
    #     BASE_PATH, 
    #     'csse_covid_19_time_series',
    #     'time_series_covid19_confirmed_global.csv')
    # confirmed = data.load_csv_data(confirmed)
    # features = []
    # targets = []
    
    # for val in np.unique(confirmed["Country/Region"]):
    #     df = data.filter_by_attribute(
    #         confirmed, "Country/Region", val)
    #     cases, labels = data.get_cases_chronologically(df)
    #     features.append(cases)
    #     targets.append(labels)
    
    # features = np.concatenate(features, axis=0)
    # targets = np.concatenate(targets, axis=0)
    # predictions = {}
    
    # for _dist in ['minkowski', 'manhattan']:
    #     for val in np.unique(confirmed["Country/Region"]):
    #         # test data
    #         df = data.filter_by_attribute(
    #             confirmed, "Country/Region", val)
    #         cases, labels = data.get_cases_chronologically(df)
    
    #         # filter the rest of the data to get rid of the country we are
    #         # trying to predict
    #         mask = targets[:, 1] != val
    #         tr_features = features[mask]
    #         tr_targets = targets[mask][:, 1]
    
    #         above_min_cases = tr_features.sum(axis=-1) > MIN_CASES
    #         tr_features = tr_features[above_min_cases]
    #         if NORMALIZE:
    #             tr_features = tr_features / tr_features.sum(axis=-1, keepdims=True)
    #         tr_targets = tr_targets[above_min_cases]
    
    #         # train knn
    #         knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, metric=_dist)
    #         knn.fit(tr_features, tr_targets)
    
    #         # predict
    #         cases = cases.sum(axis=0, keepdims=True)
    #         # nearest country to this one based on trajectory
    #         label = knn.predict(cases)
            
    #         if val not in predictions:
    #             predictions[val] = {}
    #         predictions[val][_dist] = label.tolist()
    
    # with open('./exp/results/knn_raw.json', 'w') as f:
    #     json.dump(predictions, f, indent=4)
        
    # res = json.load(open('./exp/results/knn_raw.json'))
    
     
    # Country_Region = list(set(confirmed["Country/Region"]))
    
    # country_features = {}
    # for t in range(features.shape[0]):
    #     checked = []
    #     if confirmed["Country/Region"][t] in checked:
    #         country_features[confirmed["Country/Region"][t]] += features[t]
    #     else:
    #         country_features[confirmed["Country/Region"][t]] = features[t]

    # country_features_values = np.asarray(list(country_features.values()))    
    # country_targets = sorted(list(set(confirmed["Country/Region"])))
    # from sklearn.cluster import KMeans
    # kmeans = KMeans(n_clusters=3, max_iter=300)
    # kmeans.fit(country_features_values, country_targets)
    # kmeans_predictions = kmeans.predict(country_features_values)
    # print (sorted(list(set(kmeans_predictions))))
    
    # # country_30_means = {}
    # # country_30_means_index = {}
    # # dict_30_means = {}
    # # for p in range(len(kmeans_predictions)):
    # #     dict_30_means[country_targets[p]] = kmeans_predictions[p]
    # #     if kmeans_predictions[p] in list(country_30_means.keys()):
    # #         country_30_means[kmeans_predictions[p]].append(country_targets[p])
    # #         country_30_means_index[kmeans_predictions[p]].append(p)
    # #     else:
    # #         country_30_means[kmeans_predictions[p]] = [country_targets[p]]
    # #         country_30_means_index[kmeans_predictions[p]] = [p]
    
    # # country_5_means = {}
    # # country_5_means_index = {}
    # # dict_5_means = {}
    # # for p in range(len(kmeans_predictions)):
    # #     dict_5_means[country_targets[p]] = kmeans_predictions[p]
    # #     if kmeans_predictions[p] in list(country_5_means.keys()):
    # #         country_5_means[kmeans_predictions[p]].append(country_targets[p])
    # #         country_5_means_index[kmeans_predictions[p]].append(p)
    # #     else:
    # #         country_5_means[kmeans_predictions[p]] = [country_targets[p]]
    # #         country_5_means_index[kmeans_predictions[p]] = [p]
    
    # country_3_means = {}
    # country_3_means_index = {}
    # dict_3_means = {}
    # for p in range(len(kmeans_predictions)):
    #     dict_3_means[country_targets[p]] = kmeans_predictions[p]
    #     if kmeans_predictions[p] in list(country_3_means.keys()):
    #         country_3_means[kmeans_predictions[p]].append(country_targets[p])
    #         country_3_means_index[kmeans_predictions[p]].append(p)
    #     else:
    #         country_3_means[kmeans_predictions[p]] = [country_targets[p]]
    #         country_3_means_index[kmeans_predictions[p]] = [p]


    # world = gpd.read_file(
    #     gpd.datasets.get_path('naturalearth_lowres')
    # )
    # boroughs = gpd.read_file(
    #     gplt.datasets.get_path('nyc_boroughs')
    # )
    # collisions = gpd.read_file(
    #     gplt.datasets.get_path('nyc_injurious_collisions')
    # )
    # gplt.polyplot(world, figsize=(8, 4))
    # country_loc = {}
    # for i in range((len(confirmed))):
    #     if confirmed.loc[i]["Country/Region"] in list(country_loc.keys()):
    #         country_loc[confirmed.loc[i]["Country/Region"]].append([confirmed.loc[i]["Long"],confirmed.loc[i]["Lat"]])
    #     else:
    #         country_loc[confirmed.loc[i]["Country/Region"]] = [[confirmed.loc[i]["Long"],confirmed.loc[i]["Lat"]]]
    
    # for country, loc in country_loc.items():
    #     country_loc[country] = np.average(np.asarray(loc),axis = 0)

    
    # # import geopandas as gpd

    # # world.plot()
    
    
    # country_points = [Point(loc) for loc in list(country_loc.values())]
    # country_points_df = gpd.GeoDataFrame(geometry = country_points)
    # fig,ax = plt.subplots()
    # world.plot(ax=ax,alpha=0.4)
    # country_points_df.plot(ax=ax)


    # world["class"] = np.full(world.shape[0],-1)
    # for idx in world.index:
    #     if world["name"][idx] in list(country_features.keys()):
    #         world["class"][idx] = dict_3_means[world["name"][idx]]
    
    # for idx in range(world.shape[0]):
    #     if world.loc[idx]["name"] in list(country_features.keys()):
    #         world.loc[idx]["class"] = dict_3_means[world.loc[idx]["name"]]
    

    
    
    