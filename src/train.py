import argparse
import os

import mlflow
import mlflow.xgboost
import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from xgboost import XGBClassifier


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of trees",
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=5,
        help="Maximum depth of trees",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # MLflow setup
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment("xgboost-github-actions-demo")

    # Random dataset
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": 0.05,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
    }

    with mlflow.start_run():

        # log params
        mlflow.log_params(params)

        model = XGBClassifier(**params)

        model.fit(X_train, y_train)

        # predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)

        # log model
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
        )

        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    main()