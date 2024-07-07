# clustering.py
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data
data = pd.read_csv('preprocessed_data.csv')

# Implement K-Means clustering
def kmeans_clustering(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    data['Cluster'] = kmeans.fit_predict(data[['Return', 'Volatility']])
    return data, kmeans

data, kmeans = kmeans_clustering(data)

# Plot clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Return', y='Volatility', hue='Cluster', data=data, palette='viridis')
plt.title('K-Means Clustering')
plt.show()
