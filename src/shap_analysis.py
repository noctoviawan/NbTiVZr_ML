import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd

from config import MODELS, SHAP_DIR
from preprocess import load_and_preprocess


def save_figure(path, dpi=300):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def run_shap_analysis(df=None, feature_cols=None, target_col="H"):
    if df is None:
        df = load_and_preprocess()

    if feature_cols is None:
        feature_cols = [
            "delta_pct",
            "LogNb_Ti_over_V_Zr",
            "Nb_over_Nb+Zr",
            "V_Zr"
        ]

    model_df = df[feature_cols + [target_col]].dropna(subset=[target_col]).copy()
    X = model_df[feature_cols]

    pipeline = joblib.load(MODELS / "xgboost_model.joblib")
    imputer = pipeline.named_steps["imputer"]
    model = pipeline.named_steps["model"]

    X_imp = imputer.transform(X)
    X_imp_df = pd.DataFrame(X_imp, columns=feature_cols)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_imp_df)

    plt.figure()
    shap.summary_plot(shap_values, X_imp_df, plot_type="bar", show=False)
    save_figure(SHAP_DIR / "shap_summary_bar.png")

    plt.figure()
    shap.summary_plot(shap_values, X_imp_df, show=False)
    save_figure(SHAP_DIR / "shap_summary_beeswarm.png")

    for feat in feature_cols:
        plt.figure()
        shap.dependence_plot(feat, shap_values, X_imp_df, show=False)
        save_figure(SHAP_DIR / f"shap_dependence_{feat}.png")

    print(f"SHAP outputs saved to: {SHAP_DIR}")


if __name__ == "__main__":
    run_shap_analysis()
