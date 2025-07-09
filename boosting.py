import logging
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("boosting_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# ---------------- Load Data ----------------
iris = load_iris()
X, y = iris.data, iris.target
logger.info("Loaded Iris dataset with %d samples", X.shape[0])

# ---------------- Train/Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
logger.info("Split data: %d train / %d test", len(X_train), len(X_test))

# ---------------- Boosting ----------------
boosting = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=1
)
boosting.fit(X_train, y_train)
logger.info("Trained AdaBoost with 50 weak learners (depth=1)")

# ---------------- Prediction ----------------
y_pred = boosting.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logger.info("Boosting Accuracy: %.2f%%", accuracy * 100)
