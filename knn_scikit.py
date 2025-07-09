import logging
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# ------------------------ Logging Setup ------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("knn_log.txt"),   # log to file
        logging.StreamHandler()               # also show in console
    ]
)
logger = logging.getLogger()

# ------------------------ Load Data ------------------------
iris = load_iris()
X, y = iris.data, iris.target
logger.info("Loaded Iris dataset with %d samples", len(X))

# ------------------------ Train/Test Split ------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logger.info("Split data into training (%d) and testing (%d) samples", len(X_train), len(X_test))

# ------------------------ Feature Scaling ------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
logger.info("Features standardized")

# ------------------------ Train KNN ------------------------
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
logger.info("Trained KNN with k=3")

# ------------------------ Predict and Evaluate ------------------------
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
logger.info("Model Accuracy: %.2f%%", accuracy * 100)

logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))
