import pandas
from pylab import rcParams
import seaborn as sb
import matplotlib.pyplot as plb
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import StandardScaler
import numpy as np


import sklearn
from sklearn.cluster import DBSCAN
from collections import Counter

import matplotlib.pyplot as plt



#from matplotlib import  inline

import numpy


def MyDBSCAN(D, eps, MinPts):


    labels = [0] * len(D)

    C = 0



    # For each point j in the Dataset D...
    #it check for minots
    # ('j' is the index of the datapoint, rather than the datapoint itself.)
    for j in range(0, len(D)):

        # Only points that have not already been claimed can be picked as new
        # seed points.
        # If the point's label is not 0, continue to the next point.
        if not (labels[j] == 0):
            continue

        # Find all of j's neighboring points.
        NeighborPts = CalDistQuery(D, j, eps)

        #checks condition for Minpts
        if len(NeighborPts) >= MinPts:
            C += 1
            growCluster(D, labels, j, NeighborPts, C, eps, MinPts)

        #labels is put as -1 which mean it is an anamoly
        else:
            labels[j] = -1


    return labels


def growCluster(D, labels, P, NeighborPts, C, eps, MinPts):



    labels[P] = C

    i = 0
#check for neightbours add it to lisst
    while i < len(NeighborPts):


        Pn = NeighborPts[i]

        if labels[Pn] == 0:
            labels[Pn] = C

            PnNeighborPts = CalDistQuery(D, Pn, eps)

            if len(PnNeighborPts) >= MinPts:
                NeighborPts = NeighborPts + PnNeighborPts


        elif labels[Pn] == -1:
            labels[Pn] = C



        i += 1


#calculates euclidean distance between the given data points

def CalDistQuery(D, P, eps):

    neighbors = []

    Pn=0
    while Pn < len(D):



        if numpy.linalg.norm(D[P] - D[Pn]) < eps:
            neighbors.append(Pn)

        Pn=Pn+1

    return neighbors

rcParams['figure.figsize'] = 5,4



ts = TimeSeries(key='YZVD5E27768X5Z0J', output_format='pandas')
#data, meta_data = ts.get_intraday(symbol='MSFT',interval='1min', outputsize='full')
df,meta_data2=ts.get_daily(symbol='MSFT', outputsize='full')
df['date'] = df.index.astype(str)
df.columns = ['open', 'high','low','close','volume','date']
df.drop(["date"], axis = 1, inplace = True)

data = df.iloc[:,0:4]
#data = df.ix[10,:]
data=data[2:1000]
#print(data)
target = df.iloc[:,4]

#data = StandardScaler().fit(data)
#data = StandardScaler().transform(data)
data=StandardScaler().fit_transform(data)
labels2=MyDBSCAN(data,0.8,50)
print("amey")
print(labels2)
k=2000;
print("Stock Ananmolies are :")
for p in range(len(labels2)):
    if labels2[p]==-1:
        print(df.iloc[p+k])



model = DBSCAN(eps=0.8,min_samples=50).fit(data)

core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
core_samples_mask[model.core_sample_indices_] = True

print(model)

outliers_df=pd.DataFrame(data)

print(Counter(model.labels_))


#print(outliers_df[model.labels_==-1])

#fig=plt.figure()

#ax=fig.add_axes([.1,.1,1,1])
labels=model.labels_
n_clusters_ = len(set(labels))
n_clusters_=2
#n_clusters_2 = len(set(labels)) - (1 if -1 in labels else 0)

print("number of clusters")
#print(n_clusters_2)
#colors=model.labels_

unique_labels = set(labels)
print(unique_labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
print(unique_labels)

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [1, 0, 0, 0]

    class_member_mask = (labels == k)

    xy = data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='b', markersize=14)

    xy = data[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='b', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()


#print(data['close','volume'])


#ax.scatter(data[:,2],data[data:,1],c=colors,s=120)
