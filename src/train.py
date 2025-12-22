import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from lightgbm import LGBMClassifier
import joblib
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

def prepare_data(X, y):
    feature_columns = [
        "dist_sma_7",
        "dist_sma_20",
        "dist_sma_200",
        "bb_pct_b",
        "rsi",
        "volume_ratio",
        "close_open_pct",
        "high_low_pct",
        "macd_diff"
    ]

    # Keep only needed columns
    X = X[feature_columns]

    # Drop rows with NaNs (and align y)
    X = X.dropna()
    y = y.loc[X.index]

    print("Data prepared successfully")
    print(X.info())
    print(y.info())

    return X, y, feature_columns

def train_model(x_train,y_train,x_val,y_val):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',
            solver='lbfgs',
            random_state=42,
            class_weight='balanced'

        ),
        "Random Forest":RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            max_depth=-1,
            num_leaves=64,
            min_child_samples=10,
            learning_rate=0.05,
            random_state=42,
            class_weight='balanced'

        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            random_state=42,
            eval_metric='mlogloss',
            n_jobs=-1
        ),
        "CatBoost": CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            random_state=42,
            verbose=False,
            auto_class_weights="Balanced"
        )




    }
    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        score = model.score(x_val, y_val)
        results[name] = {
            "model": model,
            "score": score}
        print(f"{name} Validation Score: {score:.4f}")

    #selecting best model
    best_model_name = max(results, key=lambda k: results[k]["score"])
    return results[best_model_name]["model"], best_model_name
if __name__ == "__main__":
    import os

    # Load data
    X = pd.read_csv("data/processed/X_clean.csv")
    y = pd.read_csv("data/processed/y_clean.csv").iloc[:, 0]

    X, y, feature_columns = prepare_data(X, y)

    # Time-based split
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)

    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]

    X_val = X.iloc[train_size:train_size + val_size]
    y_val = y.iloc[train_size:train_size + val_size]

    X_test = X.iloc[train_size + val_size:]
    y_test = y.iloc[train_size + val_size:]

    # Scale ONLY for Logistic Regression (optional for now)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train models (tree models use raw data)
    model, model_name = train_model(
        X_train,
        y_train,
        X_val,
        y_val
    )

    print(f"\nBest model selected: {model_name}")

    # Ensure model directory exists
    MODEL_DIR = "../models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save artifacts
    joblib.dump(model, f"{MODEL_DIR}/classifier.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(feature_columns, f"{MODEL_DIR}/feature_columns.pkl")


    # Train model
    #model, model_name = train_model(X_train_scaled, y_train, X_val_scaled, y_val)
    #save
    # Save
  #joblib.dump(model, "../models/classifier.pkl")
    #joblib.dump(scaler, "../models/scaler.pkl")
    #joblib.dump(feature_columns, "../models/feature_columns.pkl")


