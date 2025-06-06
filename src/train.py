# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# import joblib

# iris = load_iris()
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
# clf = RandomForestClassifier()
# clf.fit(X_train, y_train)
# joblib.dump(clf, 'model/model.pkl')
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load data
df = pd.read_csv('data/iris.csv')
X = df.drop('target', axis=1)
y = df['target']

# Train model
clf = RandomForestClassifier()
clf.fit(X, y)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/model.pkl")
print("✅ Model trained and saved to model/model.pkl")
