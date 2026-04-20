import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

from config import TABLES, MODELS, FIGURES
from preprocess import load_and_preprocess


def save_actual_vs_pred(y_true, y_pred, out_path, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Actual H")
    plt.ylabel("Predicted H")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def train_xgboost_model(df=None, feature_cols=None, target_col="H", test_size=0.2, random_state=42):

    if df is None:
        df = load_and_preprocess()

    if feature_cols is None:
        feature_cols = [
            "delta_pct",
            "LogNb_Ti_over_V_Zr",
            "Nb_over_Nb+Zr",
            "V_Zr"
        ]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    model_df = df[feature_cols + [target_col]].dropna(subset=[target_col]).copy()

    X = model_df[feature_cols]
    y = model_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=random_state
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")

    results = pd.DataFrame([{
        "Model": "XGBoost",
        "Features": ", ".join(feature_cols),
        "Train_R2": r2_score(y_train, y_train_pred),
        "Test_R2": r2_score(y_test, y_test_pred),
        "Train_RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "Test_RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "Train_MAE": mean_absolute_error(y_train, y_train_pred),
        "Test_MAE": mean_absolute_error(y_test, y_test_pred),
        "CV_R2_mean": cv_scores.mean(),
        "CV_R2_std": cv_scores.std(),
        "Rows_used": len(model_df)
    }])

    results.to_csv(TABLES / "xgboost_results.csv", index=False)
    joblib.dump(pipeline, MODELS / "xgboost_model.joblib")

    train_pred_df = pd.DataFrame({
        "Actual_H": y_train.values,
        "Predicted_H": y_train_pred
    })
    train_pred_df.to_csv(TABLES / "xgboost_train_predictions.csv", index=False)

    test_pred_df = pd.DataFrame({
        "Actual_H": y_test.values,
        "Predicted_H": y_test_pred
    })
    test_pred_df.to_csv(TABLES / "xgboost_test_predictions.csv", index=False)

    save_actual_vs_pred(
        y_train.values, y_train_pred,
        FIGURES / "xgboost_actual_vs_pred_train.png",
        "XGBoost Train: Actual vs Predicted"
    )
    save_actual_vs_pred(
        y_test.values, y_test_pred,
        FIGURES / "xgboost_actual_vs_pred_test.png",
        "XGBoost Test: Actual vs Predicted"
    )

    print(results)
    print(f"Saved model to {MODELS / 'xgboost_model.joblib'}")
    return pipeline, X_train, X_test, y_train, y_test, feature_cols


if __name__ == "__main__":
    train_xgboost_model()
