import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def train_and_predict(filepath, selected_model=None):
    df = pd.read_csv(filepath)

    # Drop non-numeric columns from input features
    X = df.drop(columns=['student_id', 'name', 'gender', 'Risk_Level', 'Prediction'], errors='ignore')

    # Compute raw target as average of marks
    raw_target = df[['math_marks', 'science_marks', 'english_marks']].mean(axis=1)

    # Convert to risk labels (0=Risk, 1=Average, 2=Advanced)
    labels = []
    for score in raw_target:
        if score <= raw_target.quantile(0.33):
            labels.append(0)
        elif score <= raw_target.quantile(0.66):
            labels.append(1)
        else:
            labels.append(2)
    y = labels

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Adjust K for KNN
    k_neighbors = min(3, len(X_train))

    # Define models
    models = {
        'SVM': SVC(),
        'DecisionTree': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier(n_neighbors=k_neighbors)
    }

    results = {}

    # Train only selected model if provided
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

    # Use the selected model for full dataset prediction
    if selected_model and selected_model in results:
        best_model_name = selected_model
    else:
        best_model_name = max(results, key=lambda k: results[k]['accuracy'])

    best_model = results[best_model_name]['model']

    full_predictions = best_model.predict(X)

    # Map predictions to human-readable labels
    risk_labels = ['Risk' if p==0 else 'Average' if p==1 else 'Advanced' for p in full_predictions]

    df['Prediction'] = full_predictions
    df['Risk_Level'] = risk_labels

    # Return dataframe, selected model name, accuracy, all model results
    return df, best_model_name, results[best_model_name]['accuracy'], results
