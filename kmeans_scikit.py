import logging
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kmeans_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# ---------------- Load Data ----------------
iris = load_iris()
X = iris.data
logger.info("Loaded Iris dataset with shape %s", X.shape)

# ---------------- Standardize ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
logger.info("Data standardized.")

# ---------------- K-Means Clustering ----------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
logger.info("K-Means model trained with 3 clusters.")

# ---------------- Plotting ----------------
plt.figure(figsize=(6, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='X', s=200, label='Centroids')
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.title('K-Means Clustering - Iris Dataset')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------- Logging Results ----------------
logger.info("Cluster centers:\n%s", kmeans.cluster_centers_)
