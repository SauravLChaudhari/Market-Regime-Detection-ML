# visualization.py
import pandas as pd
import matplotlib.pyplot as plt

# Load data with cluster labels
data = pd.read_csv('preprocessed_data.csv')

# Plot regime shifts
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='Close Price')
for cluster in data['Cluster'].unique():
    cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(cluster_data.index, cluster_data['Close'], label=f'Cluster {cluster}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Market Regime Shifts')
plt.legend()
plt.show()
