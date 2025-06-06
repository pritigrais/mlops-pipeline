import os
import joblib
import pandas as pd

def test_model_exists():
    assert os.path.exists("model/model.pkl"), "Model file not found!"

def test_prediction_shape():
    model = joblib.load("model/model.pkl")
    sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],
                          columns=['sepal length (cm)', 'sepal width (cm)',
                                   'petal length (cm)', 'petal width (cm)'])
    pred = model.predict(sample)
    assert pred.shape == (1,)
