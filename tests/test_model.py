# tests/test_model.py

from src.model import train_and_evaluate_model
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

def test_train_and_evaluate_model_returns_model_and_score():
    # Create a small synthetic classification dataset for unit testing purposes
    X, y = make_classification(n_samples=100, n_features=5,
                               n_classes=2, weights=[0.9, 0.1],
                               random_state=42)

    # Split manually
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    trained_model, f1 = train_and_evaluate_model(model, "LogisticRegression", X_train, X_test, y_train, y_test)

    assert trained_model is not None
    assert 0.0 <= f1 <= 1.0
