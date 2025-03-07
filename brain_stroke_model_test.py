import pandas as pd
import joblib


def test_model(gender, age, hypertension, heart_disease, ever_married,
               work_type, Residence_type, avg_glucose_level, bmi,
               smoking_status):
    """
    Test a single data record by encoding the input parameters and predicting the stroke outcome.

    Parameters:
      - gender: (str) 'Female' or other string (if not 'Female', treated as male)
      - age: (float or int)
      - hypertension: (int) 0 or 1
      - heart_disease: (int) 0 or 1
      - ever_married: (str) 'Yes' or other string (if not 'Yes', treated as No)
      - work_type: (str) e.g., 'Govt_job', 'Private', 'Self-employed', 'children'
      - Residence_type: (str) e.g., 'Rural', 'Urban'
      - avg_glucose_level: (float)
      - bmi: (float)
      - smoking_status: (str) e.g., 'Unknown', 'formerly smoked', 'never smoked', 'smokes'

    Returns:
      The predicted stroke outcome (0 or 1).
    """
    # 1. Create a dictionary for the input data.
    data = {
        'gender': [1 if gender == 'Female' else 0],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [1 if ever_married == 'Yes' else 0],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        # Temporary columns for one-hot encoding:
        'work_type': [work_type],
        'Residence_type': [Residence_type],
        'smoking_status': [smoking_status]
    }

    # 2. Create a DataFrame
    df = pd.DataFrame(data)

    # 3. One-hot encode the multi-category columns
    df_encoded = pd.get_dummies(df, columns=['work_type', 'Residence_type', 'smoking_status'])

    # 4. Define the expected feature columns (note: target 'stroke' is removed)
    expected_columns = [
        'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
        'avg_glucose_level', 'bmi',
        'work_type_Govt_job', 'work_type_Private', 'work_type_Self-employed', 'work_type_children',
        'Residence_type_Rural', 'Residence_type_Urban',
        'smoking_status_Unknown', 'smoking_status_formerly smoked',
        'smoking_status_never smoked', 'smoking_status_smokes'
    ]

    # 5. For any expected column missing from df_encoded, add it with default value 0.
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # 6. Reorder the columns to match the expected order.
    df_encoded = df_encoded.reindex(columns=expected_columns)

    # 7. Load the trained model (ensure the file path is correct)
    model = joblib.load("brain_stroke_model.pkl")


    print(f"Input Record:\n{df}\n")
    print(f"Encoded Input:\n{df_encoded}\n")

    prediction = model.predict(df_encoded)
    prediction_proba = model.predict_proba(df_encoded)

    # Display the results
    diagnosis = "Stroke" if prediction[0] == 1 else "No Stroke"
    print("\nPredicted Diagnosis:", diagnosis)
    print("Prediction Probability:", max(prediction_proba[0]) * 100, "%")

    return prediction


# Example usage:
if __name__ == "__main__":
    # Provide a sample record (adjust values as needed)
    test_model(
        gender="Female",
        age=50,
        hypertension=0,
        heart_disease=0,
        ever_married="Yes",
        work_type="Private",
        Residence_type="Urban",
        avg_glucose_level=228.69,
        bmi=36.6,
        smoking_status="formerly smoked"
    )
