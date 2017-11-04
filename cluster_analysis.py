import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import RandomizedPCA
from sklearn.cluster import KMeans
import math


# Parameters for plotting

PLOT_TYPE_TEXT = False    # to see indices
PLOT_VECTORS = True       # to see original features in P.C.-Space


matplotlib.style.use('ggplot') # styling to make look pretty
c = ['red', 'green', 'blue', 'orange', 'yellow', 'brown']

def drawVectors(transformed_features, components_, columns, plt):
  num_columns = len(columns)

  # This function will project  original feature (columns) onto principal component feature space, to
  # visualize how important each one was in the multi-dimensional scaling
  
  # Scaling the principal components by the max value in 
  # the transformed set belonging to that component
  xvector = components_[0] * max(transformed_features[:,0])
  yvector = components_[1] * max(transformed_features[:,1])
  
  # Visualize projections
  # Sorting each column by its length. These are original columns, not the principal components

  important_features = { columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2) for i in range(num_columns) }
  important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)
  print ("Projected Features by importance:\n", important_features)

  ax = plt.axes()

  # Using an arrow to project each original feature as a labeled vector on your principal component axes
  
  for i in range(num_columns):  
    plt.arrow(0, 0, xvector[i], yvector[i], color='b', width=0.0005, head_width=0.02, alpha=0.75, zorder=600000)
    plt.text(xvector[i]*1.2, yvector[i]*1.2, list(columns)[i], color='b', alpha=0.75, zorder=600000)
  return ax
    
# Doing PCA for dimensionality reduction
def doPCA(data, dimensions=2):
  model = RandomizedPCA(n_components=dimensions)
  model.fit(data)
  return model

# Applying KMeans on data and passing the number of clusters in the parameter

def doKMeans(data, clusters=0):
  model = KMeans(clusters)
  model.fit(data) 
  return model.cluster_centers_, model.labels_



# Loading up the dataset. It may or may not have nans in it. Making
# sure to catch them and destroy them, by setting them to '0'.

df = pd.read_csv('Wholesale-customers-data.csv')
df = df.fillna(0)


# Getting rid of the 'Channel' and 'Region' columns, since it is being assumed 
# as if this were a single location wholesaler, rather than a national / 
# international one.

# Dropping  'Channel' and 'Region'
df = df.drop(labels = ['Channel', 'Region'], axis=1)

# Plotting before proceeding in order to get idea of data
df.plot.hist(alpha = 0.4)


# Having checked out data, it is noticed there's a pretty big gap
# between the top customers in each feature category and the rest. Some feature
# scaling algos won't get rid of outliers, so it's a good idea to handle that
# manually.

# Removing top 5 and bottom 5 samples for each column:
drop = {}
for col in df.columns:
  # Bottom 5
  sort = df.sort_values(by=col, ascending=True)
  if len(sort) > 5: sort=sort[:5]
  for index in sort.index: drop[index] = True # Just store the index once

  # Top 5
  sort = df.sort_values(by=col, ascending=False)
  if len(sort) > 5: sort=sort[:5]
  for index in sort.index: drop[index] = True # Just store the index once


# Drop rows by index. We do this all at once in case there is a
# collision. This way, we don't end up dropping more rows than we have
# to, if there is a single row that satisfies the drop for multiple columns.
# Since there are 6 rows, if we end up dropping < 5*6*2 = 60 rows, that means
# there indeed were collisions.
print ("Dropping {0} Outliers...".format(len(drop)))
df.drop(inplace=True, labels=drop.keys(), axis=0)
print (df.describe())


# NORMALIZATION: Normalization divides each item by the average overall amount of spending
#                new feature is = the contribution of overall spending going into
#                that particular item: $spent on feature / $overall spent by sample
#
# MINMAX:        When dealing with all the same units, this will produce a near face-value amount
#                a single outlier, can cause all data to get squashed up in lower percentages.
#                [(sampleFeatureValue-min) / (max-min)] * (max-min) + min
#                Where min and max are for the overall feature values for all samples.


T = df



# Sometimes PCA  is performed before doing KMeans, so that KMeans only
# operates on the most meaningful features. In my case, there are so few features
# that doing PCA ahead of time isn't really necessary, and can do KMeans in
# feature space.


# Doing KMeans

n_clusters = 6
centroids, labels = doKMeans(T, n_clusters)


# Printing centroids

print(centroids)

# Performing PCA after to visualize the results. Projecting the centroids and 
# the samples into the new 2D feature space for visualization purposes.

display_pca = doPCA(T)
T = display_pca.transform(T)
CC = display_pca.transform(centroids)


# Visualizing all the samples and coloring accordingly

fig = plt.figure()
ax = fig.add_subplot(111)
if PLOT_TYPE_TEXT:
  # Plotting the index of the sample
  for i in range(len(T)): ax.text(T[i,0], T[i,1], df.index[i], color=c[labels[i]], alpha=0.75, zorder=600000)
  ax.set_xlim(min(T[:,0])*1.2, max(T[:,0])*1.2)
  ax.set_ylim(min(T[:,1])*1.2, max(T[:,1])*1.2)
else:
  # Plotting a regular scatter plot
  sample_colors = [ c[labels[i]] for i in range(len(T)) ]
  ax.scatter(T[:, 0], T[:, 1], c=sample_colors, marker='o', alpha=0.2)


# Plotting the Centroids as X's, and labelling them

ax.scatter(CC[:, 0], CC[:, 1], marker='x', s=169, linewidths=3, zorder=1000, c=c)
for i in range(len(centroids)): ax.text(CC[i, 0], CC[i, 1], str(i), zorder=500010, fontsize=18, color=c[i])


# Displaying feature vectors

if PLOT_VECTORS: drawVectors(T, display_pca.components_, df.columns, plt)


# Adding the cluster label back into the dataframe and displaying it

df['label'] = pd.Series(labels, index=df.index)
print (df)
# Plotting the plot
plt.show()