import warnings
warnings.filterwarnings('ignore')

# --- Libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- ML Tools ---
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as imbPipeline

# --- Load and clean data ---
df = pd.read_csv("./static/csv/Diabetes_Prediction.csv")
df = df.drop_duplicates()
df = df[df['gender'] != 'Other']

# --- Re-categorize smoking history ---
def recategorize_smoking(smoking_status):
    if smoking_status in ['never', 'No Info']:
        return 'non-smoker'
    elif smoking_status == 'current':
        return 'current'
    else:
        return 'past_smoker'

df['smoking_history'] = df['smoking_history'].apply(recategorize_smoking)

# --- One-hot encoding ---
df = pd.get_dummies(df, columns=['gender', 'smoking_history'], drop_first=True)

# --- Visualize class distribution ---
sns.countplot(x='diabetes', data=df)
plt.title('Diabetes Class Distribution')
plt.xlabel('Diabetes')
plt.ylabel('Count')
plt.show()

# --- Correlation matrix ---
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title("Correlation Matrix Heatmap")
plt.show()

# --- Split data ---
X = df.drop('diabetes', axis=1)
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Resampling and preprocessing ---
numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'hypertension', 'heart_disease']

preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), numerical_cols)],
    remainder='passthrough'
)

over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)

# --- Pipeline with Random Forest ---
rf_pipeline = imbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('over', over),
    ('under', under),
    ('classifier', RandomForestClassifier(random_state=42))
])

# --- Grid search ---
param_grid = {
    'classifier__n_estimators': [100],
    'classifier__max_depth': [None],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1]
}

grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# --- Evaluate Random Forest model ---
y_pred = grid_search.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# --- Confusion matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# --- Feature importances ---
feature_names = numerical_cols + list(X.columns.difference(numerical_cols))
importances = grid_search.best_estimator_.named_steps['classifier'].feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# --- Age group analysis ---
df['age_group'] = pd.cut(df['age'], bins=[0, 20, 35, 50, 65, 80, 100],
                         labels=['0-20', '21-35', '36-50', '51-65', '66-80', '81+'])

plt.figure(figsize=(10, 6))
sns.countplot(x='age_group', hue='diabetes', data=df)
plt.title('Diabetes Distribution Across Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(title='Diabetes')
plt.show()

# --- Model comparison ---
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

basic_results = []
for name, model in models.items():
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    basic_results.append((name, round(acc * 100, 2), round(f1 * 100, 2)))

basic_results_df = pd.DataFrame(basic_results, columns=["Model", "Accuracy (%)", "F1 Score (%)"])
basic_results_df.sort_values(by="F1 Score (%)", ascending=False, inplace=True)
basic_results_df.reset_index(drop=True, inplace=True)

print(basic_results_df)
