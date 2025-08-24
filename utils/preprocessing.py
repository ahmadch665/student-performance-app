import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def preprocess_csv(filepath):
    df = pd.read_csv(filepath)

    # Drop duplicates and fill missing values
    df.drop_duplicates(inplace=True)
    df.fillna(method='ffill', inplace=True)

    # 👉 Preserve original values for readability
    original_df = df.copy()

    # 👉 Encode object columns (excluding name/gender)
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        if col not in ['name', 'gender']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    # 👉 Identify columns to skip normalization (marks + attendance)
    exclude_cols = [col for col in df.columns if col.endswith('_marks')] + ['attendance_percent', 'semester']

    # 👉 Apply normalization to remaining numeric columns
    columns_to_normalize = df.select_dtypes(include=['int64', 'float64']).columns.difference(exclude_cols)
    scaler = MinMaxScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    # 👉 Restore original mark columns and others (optional safety)
    for col in exclude_cols + ['student_id', 'name', 'gender', 'age']:
        if col in original_df.columns:
            df[col] = original_df[col]

    return df