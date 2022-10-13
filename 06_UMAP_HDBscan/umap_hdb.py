import pandas as pd
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# load and read a data
iris = load_iris()
df_iris_with_species = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris_with_species['species'] = pd.Series(iris.target).map(dict(zip(range(3), iris.target_names)))
df_data = iris.data

# plot 16 images
# sns.pairplot(pd.DataFrame(df_iris_with_species), hue='species')
# plt.savefig('images/16graphs.png')

# scale a data
scaler = StandardScaler()
df_data = scaler.fit_transform(df_data)

# show 9 umap using images
# f, axarr = plt.subplots(3, 3)
# for i in range(3):
#     for k in range(3):
#         reducer = umap.UMAP(n_neighbors=50, min_dist=0.001)
#         embedding = reducer.fit_transform(df_data)
#         clusterer = hdbscan.HDBSCAN()
#         clusterer.fit(embedding)
#
#         axarr[i, k].scatter(embedding[:, 0], embedding[:, 1],
#                             s=3,
#                             c=[x for x in clusterer.labels_],
#                             cmap=plt.get_cmap('rainbow', 3))
#
# plt.show()

# use UMAP
reducer = umap.UMAP(n_neighbors=50, min_dist=0.001)
embedding = reducer.fit_transform(df_data)

# plotting a data before clustering
plt.scatter(embedding[:, 0], embedding[:, 1], s=3, c=[sns.color_palette()[x] for x in iris.target])
plt.title('UMAP projection of the IRIS dataset', fontsize=12)
plt.savefig('images/umap_fitted.png')

# clustering
clusterer = hdbscan.HDBSCAN()
clusterer.fit(embedding)

# plotting a data
plt.scatter(embedding[:, 0], embedding[:, 1],
            s=3,
            c=[x for x in clusterer.labels_],
            cmap=plt.get_cmap('rainbow', 3))
# cbar = plt.colorbar()
# cbar.set_label('Types of Iris')
# cbar.set_ticks([0.5, 1.5, 2.5])
# cbar.set_ticklabels(['Setosa', 'Versicolor', 'Virginica'])
plt.title('UMAP projection of the IRIS dataset with classes', fontsize=12)
plt.grid()
plt.grid()
plt.savefig('images/umap_plus_clustered.png')
