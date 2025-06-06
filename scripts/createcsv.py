from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Save to CSV
iris_df.to_csv('iris.csv', index=False)

print("iris.csv has been saved!")
