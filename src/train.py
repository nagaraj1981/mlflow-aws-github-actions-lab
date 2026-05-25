import os
import argparse
import mlflow
import mlflow.xgboost
import numpy as np

from xgboost import XGBClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost model and log to MLflow")

    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of trees for XGBoost"
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=5,
        help="Maximum depth of each tree"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        raise ValueError(
            "MLFLOW_TRACKING_URI is not set. "
            "Please add it as a GitHub secret or export it locally."
        )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("xgboost-github-actions-demo")

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_classes=2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42
    }

    with mlflow.start_run(run_name="xgboost-github-actions-run"):
        model = XGBClassifier(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_params(params)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.xgboost.log_model(
            xgb_model=model,
            name="xgboost-model"
        )

        print("Model training completed successfully")
        print(f"MLflow Tracking URI: {tracking_uri}")
        print(f"n_estimators: {args.n_estimators}")
        print(f"max_depth: {args.max_depth}")
        print(f"accuracy: {accuracy}")
        print(f"precision: {precision}")
        print(f"recall: {recall}")
        print(f"f1_score: {f1}")


if __name__ == "__main__":
    main()
