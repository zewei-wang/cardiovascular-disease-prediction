import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_input(user_data):
    # Define label encoders
    le_sex = LabelEncoder()
    le_cp = LabelEncoder()
    le_exang = LabelEncoder()
    le_slope = LabelEncoder()
    
    # Fit label encoders with possible categorical values
    le_sex.fit(["M", "F"])
    le_cp.fit(["TA", "ATA", "NAP", "ASY"])
    le_exang.fit(["Y", "N"])
    le_slope.fit(["Up", "Flat", "Down"])
    
    # Transform categorical values
    user_data["Sex"] = le_sex.transform([user_data["Sex"]])[0]
    user_data["ChestPainType"] = le_cp.transform([user_data["ChestPainType"]])[0]
    user_data["ExerciseAngina"] = le_exang.transform([user_data["ExerciseAngina"]])[0]
    user_data["ST_Slope"] = le_slope.transform([user_data["ST_Slope"]])[0]
    
    # Keep only required features (align with model input)
    selected_features = ["Age", "Sex", "ChestPainType", "Cholesterol", "FastingBS", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"]
    df = pd.DataFrame([user_data])[selected_features]
    
    # Apply standardization
    numerical_features = ["Age", "Cholesterol", "MaxHR", "Oldpeak"]
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    return df.values

def get_user_input():
    feature_prompts = {
        "Age": "Age (years)",
        "Sex": "Sex (M/F)",
        "ChestPainType": "Chest Pain Type (TA/ATA/NAP/ASY)",
        "Cholesterol": "Serum Cholesterol (mg/dl)",
        "FastingBS": "Fasting Blood Sugar (1 if > 120 mg/dl, else 0)",
        "MaxHR": "Maximum Heart Rate Achieved", 
        "ExerciseAngina": "Exercise-Induced Angina (Y/N)",
        "Oldpeak": "Oldpeak (ST Depression)",
        "ST_Slope": "Slope of Peak Exercise ST Segment (Up/Flat/Down)"
    }
    
    user_data = {}
    print("\n=== Heart Failure Risk Prediction ===")
    for feature, prompt in feature_prompts.items():
        while True:
            try:
                user_input = input(f"{prompt}: ")
                if feature in ["Age", "Cholesterol", "MaxHR", "Oldpeak"]:
                    user_data[feature] = float(user_input)
                elif feature == "FastingBS":
                    user_data[feature] = int(user_input)
                else:
                    user_data[feature] = user_input  # Categorical
                break
            except ValueError:
                print(f"Invalid input for {feature}. Please enter a valid value.")
    
    return preprocess_input(user_data)

if __name__ == '__main__':
    # Load the saved model
    model = joblib.load("KNeighbors_HeartFailure_Model.joblib")
    
    # Get user input
    new_sample = get_user_input()
    
    # Make a prediction
    prediction = model.predict(new_sample)
    prediction_proba = model.predict_proba(new_sample)
    
    # Display the results
    diagnosis = "Heart Disease" if prediction[0] == 1 else "No Heart Disease"
    print("\nPredicted Diagnosis:", diagnosis)
    print("Prediction Probability:", max(prediction_proba[0]) * 100, "%")
