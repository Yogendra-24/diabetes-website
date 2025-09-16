from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np

# Dummy training data (real project me tum apna diabetes dataset use karoge)
X = np.array([[25, 22, 5.5, 120], [45, 30, 8.0, 160]])
y = np.array([0, 1])

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "model_pipeline.joblib")
print("Model saved successfully!")
