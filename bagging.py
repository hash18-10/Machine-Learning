import logging
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bagging_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Load data
X, y = load_iris(return_X_y=True)
logger.info("Loaded Iris dataset with %d samples", len(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
logger.info("Split data into training (%d) and testing (%d) samples", len(X_train), len(X_test))

# Bagging with Decision Trees
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,
    random_state=1
)
logger.info("Initialized BaggingClassifier with 10 DecisionTree estimators")

bagging.fit(X_train, y_train)
logger.info("Trained BaggingClassifier")

y_pred = bagging.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logger.info("Bagging Accuracy: %.2f%%", accuracy * 100)

