# src/model.py

import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier
from preprocessing import load_and_clean_data, split_data

def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    print(f"\n Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n {model_name} Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"\n {model_name} Classification Report:")
    print(classification_report(y_test, y_pred))

    f1 = f1_score(y_test, y_pred)
    return model, f1

if __name__ == "__main__":
    print(" Loading and preprocessing data...")
    X, y = load_and_clean_data("../data/creditcard.csv")
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Models to compare
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    best_model = None
    best_f1 = 0
    best_model_name = ""

    for name, model in models.items():
        trained_model, f1 = train_and_evaluate_model(model, name, X_train, X_test, y_train, y_test)
        if f1 > best_f1:
            best_f1 = f1
            best_model = trained_model
            best_model_name = name

    # Save best model
    os.makedirs("models", exist_ok=True)
    best_model_path = f"models/{best_model_name.lower()}.pkl"
    joblib.dump(best_model, best_model_path)
    print(f"\n Best model: {best_model_name} with F1 score = {best_f1:.4f}")
    print(f" Model saved to: {best_model_path}")
