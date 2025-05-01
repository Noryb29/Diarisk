import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as imbPipeline
import joblib

# Load data
def train_model():
    df = pd.read_csv("./static/csv/Diabetes_Prediction.csv")
    df = df.drop_duplicates()
    df = df[df['gender'] != 'Other']

    def recategorize_smoking(smoking_status):
        if smoking_status in ['never', 'no_Info']:
            return 'non-smoker'
        elif smoking_status == 'current':
            return 'current'
        elif smoking_status in ['ever', 'former', 'not current']:
            return 'past_smoker'
    df['smoking_history'] = df['smoking_history'].apply(recategorize_smoking)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level','hypertension','heart_disease']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['gender','smoking_history'])  # updated here
        ])
    
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']

    over = SMOTE(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.5)
    clf = imbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('over', over),
        ('under', under),
        ('classifier', RandomForestClassifier())
    ])
    param_grid = {
        'classifier__n_estimators': [100],
        'classifier__max_depth': [None],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1]
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    grid_search = GridSearchCV(clf, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(grid_search.best_estimator_, 'static/model/diabetes_model.pkl')

    return grid_search.best_estimator_

# Load model once
model = train_model()

def predict_diabetes(input_dict):
    import numpy as np
    input_df = pd.DataFrame([input_dict])
    return model.predict(input_df)[0]
