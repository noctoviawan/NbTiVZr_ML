from train_xgboost import train_xgboost_model
from train_gpr import train_gpr_model
from shap_analysis import run_shap_analysis

FEATURES = [
    "delta_pct",
    "LogNb_Ti_over_V_Zr",
    "Nb_over_Nb+Zr",
    "V_Zr"
]

if __name__ == "__main__":
    train_xgboost_model(feature_cols=FEATURES)
    train_gpr_model(feature_cols=FEATURES)
    run_shap_analysis(feature_cols=FEATURES)
