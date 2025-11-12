import argparse
import datetime
import os
from joblib import dump
import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    timestamp = args.timestamp
    print(f"Timestamp received from GitHub Actions: {timestamp}")

    # === 1️⃣ Load dataset ===
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # === 2️⃣ Train model ===
    forest = RandomForestClassifier(random_state=42)
    forest.fit(X_train, y_train)

    # === 3️⃣ Evaluate on test set ===
    y_pred = forest.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # === 4️⃣ Log with MLflow ===
    mlflow.set_tracking_uri("./mlruns")
    experiment_name = f"breast_cancer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="RandomForest_BreastCancer"):
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("algorithm", "RandomForestClassifier")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

    # === 5️⃣ Save artifacts ===
    os.makedirs("models", exist_ok=True)
    model_filename = f"models/model_{timestamp}_rf_model.joblib"
    dump(forest, model_filename)
    print(f"Model saved → {model_filename}")

    # Save test split for evaluation script
    os.makedirs("data", exist_ok=True)
    from joblib import dump as save_obj
    save_obj((X_test, y_test), "data/test_split.joblib")

    print("✅ Training complete.")