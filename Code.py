import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.decomposition import PCA
pca = PCA()
fn = r'C:\Users\DELL I5558\Desktop\Python\electricity_price_and_demand_20170926.csv'
my_data = genfromtxt(fn, delimiter=',')
pca.fit(my_data)
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xticks(features)
plt.ylabel('variance')
plt.xlabel('PCA feature')
plt.show()
