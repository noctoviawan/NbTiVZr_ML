import os
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def main():
    file_path = "data/raw/NTVZ_LIB_ALL.csv"
    out_dir = "outputs/physics_driven_model"
    os.makedirs(out_dir, exist_ok=True)

    features = [
        "delta_pct",
        "LogNb_Ti_over_V_Zr",
        "Nb_over_Nb+Zr",
        "V_Zr"
    ]
    target = "H"

    df = pd.read_csv(file_path)
    df = df.apply(pd.to_numeric, errors="coerce")

    data = df[features + [target]].dropna().reset_index(drop=True)
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models = {
        "Linear": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),
        "Lasso": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.01, max_iter=10000))
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=300,
                max_depth=4,
                random_state=42
            ))
        ]),
        "GPR": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", GaussianProcessRegressor(
                kernel=C(1.0, constant_value_bounds="fixed")
                       * RBF(length_scale=np.ones(len(features)), length_scale_bounds="fixed")
                       + WhiteKernel(noise_level=1e-2, noise_level_bounds="fixed"),
                alpha=1e-6,
                optimizer=None,
                normalize_y=True,
                random_state=42
            ))
        ])
    }

    holdout_rows = []
    prediction_frames = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        holdout_rows.append({
            "model": model_name,
            "train_r2": r2_score(y_train, pred_train),
            "test_r2": r2_score(y_test, pred_test),
            "train_rmse": rmse(y_train, pred_train),
            "test_rmse": rmse(y_test, pred_test),
            "train_mae": mean_absolute_error(y_train, pred_train),
            "test_mae": mean_absolute_error(y_test, pred_test),
        })

        pred_df = pd.DataFrame({
            "model": model_name,
            "split": ["train"] * len(y_train) + ["test"] * len(y_test),
            "actual_H": list(y_train.values) + list(y_test.values),
            "predicted_H": list(pred_train) + list(pred_test)
        })
        prediction_frames.append(pred_df)

        plt.figure(figsize=(6, 6))
        plt.scatter(y_train, pred_train, label="Train")
        plt.scatter(y_test, pred_test, label="Test")
        lo = min(y.min(), pred_train.min(), pred_test.min())
        hi = max(y.max(), pred_train.max(), pred_test.max())
        plt.plot([lo, hi], [lo, hi], linestyle="--")
        plt.xlabel("Actual H")
        plt.ylabel("Predicted H")
        plt.title(f"{model_name}: Actual vs Predicted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{model_name}_actual_vs_predicted.png"), dpi=300)
        plt.close()

    holdout_df = pd.DataFrame(holdout_rows).sort_values("test_r2", ascending=False)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    holdout_df.to_csv(os.path.join(out_dir, "holdout_metrics.csv"), index=False)
    predictions_df.to_csv(os.path.join(out_dir, "train_test_predictions.csv"), index=False)

    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_rows = []

    for model_name, model in models.items():
        cv_result = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring={
                "r2": "r2",
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error"
            },
            return_train_score=True,
            n_jobs=1
        )

        cv_rows.append({
            "model": model_name,
            "cv_train_r2_mean": cv_result["train_r2"].mean(),
            "cv_train_r2_std": cv_result["train_r2"].std(),
            "cv_test_r2_mean": cv_result["test_r2"].mean(),
            "cv_test_r2_std": cv_result["test_r2"].std(),
            "cv_train_rmse_mean": (-cv_result["train_rmse"]).mean(),
            "cv_test_rmse_mean": (-cv_result["test_rmse"]).mean(),
            "cv_train_mae_mean": (-cv_result["train_mae"]).mean(),
            "cv_test_mae_mean": (-cv_result["test_mae"]).mean(),
        })

        oof_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=1)

        plt.figure(figsize=(6, 6))
        plt.scatter(y, oof_pred)
        lo = min(y.min(), oof_pred.min())
        hi = max(y.max(), oof_pred.max())
        plt.plot([lo, hi], [lo, hi], linestyle="--")
        plt.xlabel("Actual H")
        plt.ylabel("10-fold CV Predicted H")
        plt.title(f"{model_name}: 10-fold CV OOF")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{model_name}_cv10_oof.png"), dpi=300)
        plt.close()

    cv_df = pd.DataFrame(cv_rows).sort_values("cv_test_r2_mean", ascending=False)
    cv_df.to_csv(os.path.join(out_dir, "cv10_metrics.csv"), index=False)

    pearson_df = data[features + [target]].corr(method="pearson")
    pearson_df.to_csv(os.path.join(out_dir, "pearson_matrix.csv"))

    plt.figure(figsize=(7, 6))
    im = plt.imshow(pearson_df.values, aspect="auto")
    plt.colorbar(im, label="Pearson r")
    plt.xticks(range(len(pearson_df.columns)), pearson_df.columns, rotation=90)
    plt.yticks(range(len(pearson_df.index)), pearson_df.index)
    for i in range(pearson_df.shape[0]):
        for j in range(pearson_df.shape[1]):
            val = pearson_df.iloc[i, j]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    plt.title("Pearson Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pearson_matrix.png"), dpi=300)
    plt.close()

    gpr_model = models["GPR"]
    gpr_model.fit(X_train, y_train)
    pred_train, std_train = gpr_model.predict(X_train, return_std=True)
    pred_test, std_test = gpr_model.predict(X_test, return_std=True)

    gpr_uncertainty_df = pd.DataFrame({
        "split": ["train"] * len(y_train) + ["test"] * len(y_test),
        "actual_H": list(y_train.values) + list(y_test.values),
        "predicted_H": list(pred_train) + list(pred_test),
        "uncertainty": list(std_train) + list(std_test)
    })
    gpr_uncertainty_df.to_csv(os.path.join(out_dir, "gpr_uncertainty_predictions.csv"), index=False)

    plt.figure(figsize=(6, 5))
    plt.scatter(std_test, np.abs(y_test.values - pred_test))
    plt.xlabel("Predicted uncertainty")
    plt.ylabel("Absolute error")
    plt.title("GPR Test Uncertainty vs Error")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gpr_uncertainty_vs_error.png"), dpi=300)
    plt.close()

    Xs = data[features].copy()
    medians = Xs.median()

    for f1, f2 in itertools.combinations(features, 2):
        x1 = np.linspace(Xs[f1].min(), Xs[f1].max(), 60)
        x2 = np.linspace(Xs[f2].min(), Xs[f2].max(), 60)
        xx1, xx2 = np.meshgrid(x1, x2)

        grid = pd.DataFrame({f: np.repeat(medians[f], xx1.size) for f in features})
        grid[f1] = xx1.ravel()
        grid[f2] = xx2.ravel()

        pred_grid, std_grid = gpr_model.predict(grid, return_std=True)
        zz = pred_grid.reshape(xx1.shape)
        uu = std_grid.reshape(xx1.shape)

        plt.figure(figsize=(7, 6))
        cs = plt.contourf(xx1, xx2, zz, levels=20)
        plt.colorbar(cs, label="Predicted H")
        plt.scatter(Xs[f1], Xs[f2], s=15)
        plt.xlabel(f1)
        plt.ylabel(f2)
        plt.title(f"GPR Surface: {f1} vs {f2}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"GPR_surface_{f1}_vs_{f2}.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(7, 6))
        cs = plt.contourf(xx1, xx2, uu, levels=20)
        plt.colorbar(cs, label="Prediction uncertainty")
        plt.scatter(Xs[f1], Xs[f2], s=15)
        plt.xlabel(f1)
        plt.ylabel(f2)
        plt.title(f"GPR Uncertainty: {f1} vs {f2}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"GPR_uncertainty_{f1}_vs_{f2}.png"), dpi=300)
        plt.close()

    excel_path = os.path.join(out_dir, "physics_driven_model_summary.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        holdout_df.to_excel(writer, sheet_name="holdout_metrics", index=False)
        cv_df.to_excel(writer, sheet_name="cv10_metrics", index=False)
        predictions_df.to_excel(writer, sheet_name="train_test_predictions", index=False)
        pearson_df.to_excel(writer, sheet_name="pearson_matrix")
        gpr_uncertainty_df.to_excel(writer, sheet_name="gpr_uncertainty", index=False)

    print("Done.")
    print("Saved outputs to:", out_dir)


if __name__ == "__main__":
    main()
