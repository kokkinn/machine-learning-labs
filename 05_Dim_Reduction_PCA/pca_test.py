import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from tensorflow.keras.datasets import mnist

# # define a data
# pca = PCA(784)
#
# (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# X_test = X_test.reshape(X_test.shape[0], 784)


# # define cumulative ratio data frame
# df_digits_DimRed = pd.DataFrame(pca.fit_transform(X_test))
# print(df_digits_DimRed)
# ser_range = pd.Series(range(1, 785), name='N_dim')
# df_digits_CumVarRat = pd.concat([ser_range, pd.Series(np.cumsum(pca.explained_variance_ratio_), name='CumVarRat')],
#                                 axis=1)

# create CSV file
# df_digits_CumVarRat.to_csv('csv/Cumulative_Variance_Ratio.csv', index=False)

# # plot  a graph
# plt.scatter(df_digits_CumVarRat['N_dim'], df_digits_CumVarRat['CumVarRat'], c='green', s=2)
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')
# plt.grid(True)
# plt.show()

# # define a memory usage dataframe
# df_memory_usage = pd.DataFrame(columns=['Num_dim', 'Memory_used'])
# X_test = X_test[0:800]
# # print(X_test.ndim, X_test.shape)
#
# for i in range(1, 785):
#     pca = PCA(i)
#     df_memory_usage = df_memory_usage.append({'Num_dim': i,
#                                               'Memory_used': pca.fit_transform(X_test).size * pca.fit_transform(
#                                                   X_test).itemsize},
#                                              ignore_index=True)
# # # to csv
# df_memory_usage.to_csv('csv/memory_usage.csv', index=False)
#
# # plot  a graph
# plt.scatter(df_memory_usage['Num_dim'], df_memory_usage['Memory_used'], c='green', s=2)
# plt.xlabel('Number of dimensions')
# plt.ylabel('Memory used in bytes')
# plt.grid(True)
# plt.show()
