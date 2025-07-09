import logging
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pca_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# ---------------- Load Data ----------------
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
logger.info("Loaded Iris dataset with %d samples", X.shape[0])

# ---------------- Standardize ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
logger.info("Data standardized.")

# ---------------- PCA Transformation ----------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
logger.info("Reduced to 2 principal components.")
logger.info("Explained Variance Ratio: %s", pca.explained_variance_ratio_)

# ---------------- Plotting ----------------
plt.figure(figsize=(6, 5))
colors = ['red', 'green', 'blue']
for i, color, label in zip([0, 1, 2], colors, target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, label=label)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - Iris Dataset')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
