import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def train_and_predict(filepath, selected_model=None):
    df = pd.read_csv(filepath)

    # 👉 Step 1: Drop non-numeric columns from input features
    X = df.drop(columns=['student_id', 'name', 'gender', 'Risk_Level', 'Prediction'], errors='ignore')

    # 👉 Step 2: Use one of the numeric columns as the raw target
    raw_target = df[['math_marks', 'science_marks', 'english_marks']].mean(axis=1)

    # 👉 Step 3: Convert to risk labels (0=Risk, 1=Average, 2=Advanced)
    labels = []
    for score in raw_target:
        if score <= raw_target.quantile(0.33):
            labels.append(0)
        elif score <= raw_target.quantile(0.66):
            labels.append(1)
        else:
            labels.append(2)
    y = labels

    # 👉 Step 4: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 👉 Step 5: Dynamically adjust k for KNN to avoid crash
    k_neighbors = min(3, len(X_train))

    # 👉 Step 6: Define models
    models = {
        'SVM': SVC(),
        'DecisionTree': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier(n_neighbors=k_neighbors)
    }

    results = {}

    # 👉 Step 7: Train selected model(s)
    if selected_model and selected_model in models:
        model_items = {selected_model: models[selected_model]}
    else:
        model_items = models

    for name, model in model_items.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'model': model,
            'accuracy': accuracy
        }

    # 👉 Step 8: Determine best model (highest accuracy among all trained models)
    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    best_model = results[best_model_name]['model']

    # 👉 Step 9: Predict on full dataset using the best model
    full_predictions = best_model.predict(X)

    # 👉 Step 10: Map predictions to human-readable risk levels
    risk_labels = []
    for pred in full_predictions:
        if pred == 0:
            risk_labels.append('Risk')
        elif pred == 1:
            risk_labels.append('Average')
        else:
            risk_labels.append('Advanced')

    # 👉 Step 11: Add predictions to original DataFrame
    df['Prediction'] = full_predictions
    df['Risk_Level'] = risk_labels

    # 👉 Step 12: Return df, best model name, best accuracy, and all model results
    return df, best_model_name, results[best_model_name]['accuracy'], results
