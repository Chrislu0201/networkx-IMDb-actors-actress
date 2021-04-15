#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:55:02 2019

@author: Minghan Li, Tingyi Lu, Xinran Qian
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

#import all datasets
names = pd.read_table("name.basics.tsv",'\t')
ratings = pd.read_table("title.ratings.tsv",'\t')
tbasics = pd.read_table("title.basics.tsv",'\t')
tprincipals = pd.read_table("title.principals.tsv",'\t')

mask1 = tbasics['startYear'] >= 2000
mask2 = tbasics['titleType'] == 'movie'
movie = tbasics[mask1 & mask2]  # only consider movies
mask1 = ratings['averageRating'] >= 7.8
mask2 = ratings['numVotes'] >= 100
masked_ratings = ratings[mask1 & mask2]  # only consider ratings above 7.8 with more than 100 votes
movies_with_rating = pd.merge(masked_ratings, movie, how='inner', on=['tconst'])  
# how = 'inner' will find the intersection of two dataframes

mask1 = tprincipals['category'] == 'actor'
mask2 = tprincipals['category'] == 'actress'
actors = tprincipals[mask1 | mask2]
# entire_cast stores all actors and actresses for movies with rating
entire_cast = pd.merge(actors, movies_with_rating, how='inner', on=['tconst'])

# create a list for the list of actors/actresses
actors_list = list(set(entire_cast['nconst']))

# create a ratings matrix that stores the film rating for each combination
ratingsMatrix = pd.DataFrame(np.zeros((len(actors_list),len(actors_list))), columns = actors_list, index= actors_list )
# create another connections matrix that stores the number of connection for each combination
connectionsMatrix = pd.DataFrame(np.zeros((len(actors_list),len(actors_list))), columns = actors_list, index= actors_list )

# use a for-loop to go through all movies and update the two matrices accordingly
for t in movies_with_rating['tconst']:
    mask = entire_cast['tconst']==t
    temp = entire_cast[mask]
    temp_list = []
    for s in temp['nconst']:
        temp_list.append(s)
    index = 0
    for i in temp_list:
        for j in temp_list[index+1:]:
            # use i and j to pinpoint the two space in the matrix
            # then update
            score = set(temp['averageRating']).pop()
            # num = ratingsMatrix.at[i, j] + score
            ratingsMatrix.at[i, j] +=score
            ratingsMatrix.at[j, i] +=score
            connectionsMatrix.at[i, j] +=1.0
            connectionsMatrix.at[j, i] +=1.0
        index +=1
        
#create a weighted matrix that combines both ratings matrix and connections matrix
weightedMatrix = ratingsMatrix.divide(connectionsMatrix, fill_value=None)
weightedMatrix = weightedMatrix.fillna(0)

np_weightedMatrix = np.array(weightedMatrix)

#create a DiGraph using networkx
G = nx.DiGraph()
#add actors' names as nodes
G.add_nodes_from(actors_list)
#iterate through all entries in the weighted matrix and only add edges when there is a connection between two actos
for i in range(np_weightedMatrix.shape[0]):
    for j in range(np_weightedMatrix.shape[1]):
        if np_weightedMatrix[i][j] > 0:
            #add edges
            G.add_edge(actors_list[i],actors_list[j],weight=np_weightedMatrix[i][j])

#match actors' identifiers with their names
entire_cast_names = pd.merge(entire_cast, names, on=['nconst'])
namecode = list(entire_cast_names['nconst'])
nameactual = list(entire_cast_names['primaryName'])
namedic = dict(zip(namecode, nameactual))

#relabel all nodes with actors' actual names instead of their identifiers
mapping = dict()
for i in G.nodes():
    if i in namedic.keys():
        mapping[i] = namedic[i]
G = nx.relabel_nodes(G, mapping)

#use page rank to rank the actors by importance
pr = nx.pagerank(G, alpha = 0.85)

prv = pr.values()
prv = [prv[i]*10000 for i in range(len(prv))]
#sort the rank values to obtain 20th and 70th rank values
prv.sort()
filter_pagerank_value = prv[-70]
topten_pagerank_value = prv[-20]

updated_nodes = list(G.nodes())
#remove three names containing special characters that cannot be displayed
remove = ['Toshir\xc3\xb4 Mifune','Rudolf Hrus\xc3\xadnsk\xc3\xbd','Gian Maria Volont\xc3\xa8'] 
for actor in updated_nodes:
    if actor in remove:
        G.remove_node(actor)
        continue
        #remove actors who have lower rank values than the 70th actor, so that we only have the top 70 actors
    if pr[actor]*10000 < filter_pagerank_value:
        G.remove_node(actor)

color_map = []
inner_layout = []
outer_layout = []
size_map = []
#initialize the two shells and colors of nodes and edges
for actor in G.nodes():
    if pr[actor]*10000 >= topten_pagerank_value:
        color_map.append('#A1D6E2')
        inner_layout.append(actor)
    else:
        color_map.append('#BCBABE')
        outer_layout.append(actor)
    size_map.append(pr[actor]*5000000)
    
plt.figure(figsize=(15,15))
#initialize shell layout
pos = nx.shell_layout(G, [inner_layout,outer_layout])
#draw the final graph with shell layout
nx.draw(G, with_labels = True, width=4, font_size=12, node_size = size_map, 
        node_color = color_map, edge_color = '#f4cc70',pos=pos)

plt.show()

#create the horizontal bar plot for the top 20 most important actors
objects = inner_layout
y_pos = np.arange(len(objects))
pr_sorted = sorted(pr.iteritems(), key=lambda (k,v): (v,k), reverse = True)
pr_sorted1 = pr_sorted[0:19]

performance = []
objects = []

for i in pr_sorted1:
    performance.append(i[1])
    objects.append(i[0])

plt.barh(y_pos, performance, align='center', alpha=0.5)
plt.yticks(y_pos, objects)
plt.xlabel('Rank value')
plt.title('Top 20 actors\' rank value')
 
plt.show()
