from flask import Blueprint, render_template, request
import joblib
import pandas as pd  # Import pandas

views = Blueprint(__name__, "views")

@views.route("/")
def index():
    return render_template("index.html")

@views.route("/form")
def home():
    return render_template("form.html")

@views.route("/predict", methods=["POST"])
def predict():
    try:
        # Load the model
        model = joblib.load('static/model/diabetes_model.pkl')
    except FileNotFoundError:
        return "Model file not found. Please ensure the model is trained and saved correctly.", 500

    data = request.form
    try:
        # Recategorize smoking_history just like during training
        raw_smoking = data['smoking_history'].lower()
        if raw_smoking in ['never', 'no_info']:
            smoking = 'non-smoker'
        elif raw_smoking == 'current':
            smoking = 'current'
        else:
            smoking = 'past_smoker'

        # Construct input DataFrame
        input_df = pd.DataFrame([{
            'age': float(data['age']),
            'hypertension': int(data.get('hypertension', '0') == 'on'),
            'heart_disease': int(data.get('heart_disease', '0') == 'on'),
            'bmi': float(data['bmi']),
            'HbA1c_level': float(data['HbA1c_level']),
            'blood_glucose_level': float(data['blood_glucose_level']),
            'gender': data['gender'].lower(),
            'smoking_history': smoking
        }])

        # Predict
        prediction = model.predict(input_df)[0]
        if prediction == 0:
            message = "Hooray!, You are not at risk of Diabetes in the Future"
        else:
            message = "You are at risk of Diabetes in the Future"

        firstname = data.get('firstname', 'Unknown')
        lastname = data.get('lastname', 'Unknown')

        return render_template("result.html", prediction=message, firstname=firstname, lastname=lastname)


    except Exception as e:
        print("Prediction error:", e)
        return f"An error occurred during prediction: {e}", 500

