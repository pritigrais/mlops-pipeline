import joblib
import pandas as pd

# Sample input (can also be from CLI or CSV)
sample_input = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],
                            columns=['sepal length (cm)', 'sepal width (cm)',
                                     'petal length (cm)', 'petal width (cm)'])

# Load model
model = joblib.load("model/model.pkl")

# Predict
prediction = model.predict(sample_input)
print(f"🔍 Predicted class: {prediction[0]}")
