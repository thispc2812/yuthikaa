import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
file_path = '/Users/admin/Desktop/admin/Yuthika/online_retail.csv'  # Update your file path
data = pd.read_csv(file_path, encoding='latin1')

# Prepare the data
# Use 'Quantity' and 'UnitPrice' as features, and 'Country' as the target for LDA
features = data[['Quantity', 'UnitPrice']].dropna()
target = data['Country'][features.index]

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

### 1. Principal Component Analysis (PCA)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_scaled)
explained_variance = pca.explained_variance_ratio_

# Plot PCA
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7, edgecolor='k')
plt.title('PCA Results: First Two Components')
plt.xlabel(f'Principal Component 1 ({explained_variance[0]*100:.2f}% Variance)')
plt.ylabel(f'Principal Component 2 ({explained_variance[1]*100:.2f}% Variance)')
plt.grid(True)
plt.show()

### 2. Linear Discriminant Analysis (LDA)
# Encode the target variable (Country)
encoder = LabelEncoder()
target_encoded = encoder.fit_transform(target)

# Split the data for LDA
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_encoded, test_size=0.3, random_state=42)

# Apply LDA
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)

# Evaluate the LDA model
print("LDA Accuracy:", accuracy_score(y_test, y_pred))
print("\nLDA Classification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# Transform the entire dataset for LDA visualization
lda_transformed = lda.transform(features_scaled)

# Plot LDA
plt.figure(figsize=(10, 8))
scatter_lda = plt.scatter(lda_transformed[:, 0], lda_transformed[:, 1], c=target_encoded, cmap='viridis', alpha=0.7)
plt.colorbar(scatter_lda, label='Target (Encoded Countries)')
plt.title('LDA Projection: First Two Components')
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.grid(True)
plt.show()

### 3. Cluster Analysis
# Determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(features_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True)
plt.show()

# Apply K-Means with the chosen number of clusters (e.g., k=3)
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(features_scaled)

# Add cluster labels to the original data
data['Cluster'] = clusters

# Plot Clusters in PCA space
plt.figure(figsize=(10, 8))
scatter_cluster = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis', alpha=0.7)
plt.colorbar(scatter_cluster, label='Cluster')
plt.title('Cluster Analysis: PCA Projection')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.show()
