# src/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(csv_path: str):
    df = pd.read_csv(csv_path)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Feature Scaling (especially Amount and Time)
    scaler = StandardScaler()
    X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
