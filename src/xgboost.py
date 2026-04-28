# =========================
# Install packages
# =========================
!pip install xgboost shap -q

# =========================
# Import
# =========================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# =========================
# Load data
# =========================
df = pd.read_csv("NTVZ_LIB_ALL.csv")

FEATURES = [
    "delta_pct",
    "LogNb_Ti_over_V_Zr",
    "Nb_over_Nb+Zr",
    "V_Zr"
]

TARGET = "H"

data = df[FEATURES + [TARGET]].replace([np.inf, -np.inf], np.nan)
data = data.dropna(subset=[TARGET])

X = data[FEATURES]
y = data[TARGET]

# =========================
# Output folder
# =========================
outdir = "outputs_xgboost_physics"
os.makedirs(outdir, exist_ok=True)

# =========================
# Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

# =========================
# Model
# =========================
imputer = SimpleImputer(strategy="median")

X_train_imp = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=FEATURES
)

X_test_imp = pd.DataFrame(
    imputer.transform(X_test),
    columns=FEATURES
)

X_all_imp = pd.DataFrame(
    imputer.transform(X),
    columns=FEATURES
)

model = XGBRegressor(
    n_estimators=150,
    max_depth=2,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.10,
    reg_lambda=5.0,
    min_child_weight=3,
    objective="reg:squarederror",
    random_state=42
)

model.fit(X_train_imp, y_train)

# =========================
# Prediction
# =========================
y_train_pred = model.predict(X_train_imp)
y_test_pred = model.predict(X_test_imp)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

print("Train R2:", train_r2)
print("Test R2:", test_r2)
print("Test RMSE:", test_rmse)
print("Test MAE:", test_mae)

# =========================
# Actual vs Predicted plot
# =========================
plt.figure(figsize=(7, 6))

plt.scatter(y_train, y_train_pred, label="Train", s=45, alpha=0.85)
plt.scatter(y_test, y_test_pred, label="Test", s=55, alpha=0.90)

min_val = min(y.min(), y_train_pred.min(), y_test_pred.min()) - 0.2
max_val = max(y.max(), y_train_pred.max(), y_test_pred.max()) + 0.2

plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.xlabel("Actual H")
plt.ylabel("Predicted H")
plt.title("Physics-Guided XGBoost: Actual vs Predicted")
plt.legend()

plt.text(
    0.04, 0.96,
    f"Train R² = {train_r2:.3f}\nTest R² = {test_r2:.3f}\nTest RMSE = {test_rmse:.3f}",
    transform=plt.gca().transAxes,
    va="top",
    bbox=dict(boxstyle="round", alpha=0.15)
)

plt.tight_layout()
plt.savefig(f"{outdir}/xgboost_actual_vs_predicted.png", dpi=300)
plt.show()

# =========================
# SHAP analysis
# =========================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_all_imp)

# SHAP summary plot
plt.figure()
shap.summary_plot(shap_values, X_all_imp, show=False)
plt.tight_layout()
plt.savefig(f"{outdir}/shap_summary.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================
# SHAP dependence plots
# =========================
for feature in FEATURES:
    shap.dependence_plot(
        feature,
        shap_values,
        X_all_imp,
        show=False
    )
    plt.title(f"SHAP Dependence: {feature}")
    plt.tight_layout()
    plt.savefig(f"{outdir}/shap_dependence_{feature}.png", dpi=300, bbox_inches="tight")
    plt.show()

# =========================
# Save prediction result
# =========================
pred_df = pd.DataFrame({
    "Actual_H": y_test.values,
    "Predicted_H": y_test_pred,
    "Residual": y_test.values - y_test_pred
})

pred_df.to_csv(f"{outdir}/xgboost_test_predictions.csv", index=False)

print("Saved all outputs to:", outdir)
