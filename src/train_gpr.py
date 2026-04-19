import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from config import TABLES, MODELS
from preprocess import load_and_preprocess


def train_gpr_model(feature_cols=None, target_col="H", test_size=0.2, random_state=42):
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

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(len(feature_cols))) + WhiteKernel()

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            random_state=random_state,
            n_restarts_optimizer=5
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")

    results = pd.DataFrame([{
        "Model": "GPR",
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

    results.to_csv(TABLES / "gpr_results.csv", index=False)
    joblib.dump(pipeline, MODELS / "gpr_model.joblib")

    pred_df = pd.DataFrame({
        "y_test": y_test.values,
        "y_pred": y_test_pred
    })
    pred_df.to_csv(TABLES / "gpr_test_predictions.csv", index=False)

    print(results)
    print(f"Saved model to {MODELS / 'gpr_model.joblib'}")
    return pipeline


if __name__ == "__main__":
    train_gpr_model()
