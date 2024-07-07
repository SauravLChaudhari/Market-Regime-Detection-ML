# clustering.py
from sklearn.cluster import DBSCAN

# Implement DBSCAN clustering
def dbscan_clustering(data, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    data['Cluster'] = dbscan.fit_predict(data[['Return', 'Volatility']])
    return data, dbscan

data, dbscan = dbscan_clustering(data)

# Plot clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Return', y='Volatility', hue='Cluster', data=data, palette='viridis')
plt.title('DBSCAN Clustering')
plt.show()
