from src.model import train_and_evaluate_model
from src.preprocessing import load_and_clean_data, split_data
from sklearn.linear_model import LogisticRegression

def test_train_and_evaluate_model_returns_model_and_score():
    X, y = load_and_clean_data("data/creditcard.csv")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000, random_state=42)
    trained_model, f1 = train_and_evaluate_model(model, "LogisticRegression", X_train, X_test, y_train, y_test)

    assert trained_model is not None
    assert 0 <= f1 <= 1
