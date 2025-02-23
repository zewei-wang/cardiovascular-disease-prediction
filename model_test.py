import joblib
import numpy as np

if __name__ == '__main__':
    # Load the saved model
    model = joblib.load("KNeighbors_HeartFailure_Model.joblib")

    # Example: Make a prediction on new data (replace with actual features)
    new_sample = np.array([[1.01, 0.00, 1.00, -0.03, 0.00, 1.66, 0.00, 0.30, 2.00]])  # Example input features
    prediction = model.predict(new_sample)

    # Get predicted probability (optional)
    prediction_proba = model.predict_proba(new_sample)

    print("Predicted Class:", prediction[0])  # 0 = No Heart Disease, 1 = Heart Disease
    print("Prediction Probability:", prediction_proba)
