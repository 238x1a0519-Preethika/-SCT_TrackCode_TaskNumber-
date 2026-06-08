# Mall Customer Segmentation using K-Means Clustering

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Create a sample customer dataset
data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Annual_Income': [15, 16, 17, 18, 19, 50, 55, 60, 65, 70],
    'Spending_Score': [39, 81, 6, 77, 40, 50, 55, 52, 60, 65]
}

# Convert to a Pandas DataFrame
df = pd.DataFrame(data)

# 2. Select features for clustering
X = df[['Annual_Income', 'Spending_Score']]

# 3. Apply K-Means Clustering
# n_init is explicitly set to avoid future deprecation warnings
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

# 4. Display customer groups
print("Customer Segmentation Results")
print("-----------------------------")
print(df.to_string(index=False))

# 5. Plot the clusters and their centroids
plt.figure(figsize=(8, 6))

# Scatter plot for the customer data points, colored by their assigned cluster
scatter = plt.scatter(
    df['Annual_Income'],
    df['Spending_Score'],
    c=df['Cluster'],
    cmap='viridis',
    s=100,
    edgecolor='k',
    alpha=0.8,
    label='Customers'
)

# Plot cluster centroids
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    s=250,
    c='red',
    marker='X',
    edgecolor='black',
    label='Centroids'
)

# Customize plot aesthetics
plt.title('Customer Segments using K-Means Clustering', fontsize=14, pad=15)
plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='best')
plt.tight_layout()

# Show the visualization
plt.show()