import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat
import networkx as nx
from tqdm import tqdm_notebook

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as plt
from itertools import count
from operator import itemgetter
from networkx.drawing.nx_agraph import graphviz_layout
import pylab
import Graph_Sampling
from Snowball import Snowball
trust_data = np.loadtxt('datasets/Epinions/trust_data.txt', dtype=np.int32)
df=pd.DataFrame(trust_data, columns=['member1', 'member2', 'ss'])
df= df.iloc[:, :-1]
# df = df[0 : 1000]

pd.set_option('precision',10)
G = nx.from_pandas_edgelist(df, 'member1', 'member2', create_using = nx.Graph())
object=Snowball()
sample = object.snowball(G, 100, 60) # graph, number of nodes to sample , k set
print("Number of nodes sampled=", len(sample.nodes()))
print("Number of edges sampled=", len(sample.edges()))
print(nx.average_clustering(sample))

nodes = sample.nodes()
degree = sample.degree()
colors = [degree[n] for n in nodes]

pos = nx.kamada_kawai_layout(sample)
cmap = plt.cm.viridis_r
cmap = plt.cm.Greys

vmin = min(colors)
vmax = max(colors)

fig = plt.figure(figsize = (15,9), dpi=100)

nx.draw(sample, pos, alpha = 0.8, nodelist = nodes, node_color ='w', node_size = 10, with_labels= False, font_size = 6, width = 0.5, cmap = cmap, edge_color ='yellow')
fig.set_facecolor('#0B243B')

plt.show()
y_value = []

degree =  nx.degree_histogram(sample)
x = range(len(degree))
y = [z / float(sum(degree)) for z in degree]
plt.loglog(x,y,color="blue",linewidth=2)
plt.show()
x_value = []

for i in nx.degree_histogram(G):
    y_value.append(float(i) / sum(nx.degree_histogram(G)))

for j in range(len(nx.degree_histogram(G))):
    x_value.append(j)
x_y_value = {}

for i in range(len(x_value)):
    for j in range(len(y_value)):
        if (i == j and y_value[j] != 0):
            x_y_value[x_value[i]] = y_value[j]
plt.xlabel('K')
plt.ylabel("P(K)")

plt.hist(x_y_value.keys(),x_y_value.values(),c = "blue")
plt.show()