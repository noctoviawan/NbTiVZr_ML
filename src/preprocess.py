import pandas as pd
import numpy as np
from config import INPUT_FILE, DATA_PROCESSED


def safe_log10(x):
    x = pd.to_numeric(x, errors="coerce")
    return np.log10(x.where(x > 0))


def load_and_preprocess():
    df = pd.read_csv(INPUT_FILE)

    df.columns = [c.strip() for c in df.columns]

    required = ["Nb", "Ti", "V", "Zr", "H"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass

    def ratio(a, b):
        return df[a] / df[b].replace(0, np.nan)

    df["Nb_Ti"] = ratio("Nb", "Ti")
    df["V_Zr"] = ratio("V", "Zr")
    df["Ti_V"] = ratio("Ti", "V")
    df["Nb_Zr"] = ratio("Nb", "Zr")
    df["Ti_Zr"] = ratio("Ti", "Zr")
    df["Nb_V"] = ratio("Nb", "V")
    df["Nb+Ti_over_V+Zr"] = (df["Nb"] + df["Ti"]) / (df["V"] + df["Zr"]).replace(0, np.nan)
    df["Nb_Ti_over_V_Zr"] = df["Nb_Ti"] / df["V_Zr"].replace(0, np.nan)
    df["Nb_over_Nb+Zr"] = df["Nb"] / (df["Nb"] + df["Zr"]).replace(0, np.nan)
    df["Zr_over_Nb+Zr"] = df["Zr"] / (df["Nb"] + df["Zr"]).replace(0, np.nan)
    df["Ti_V_over_Nb_Zr"] = df["Ti_V"] / df["Nb_Zr"].replace(0, np.nan)
    df["LogNb_Ti_over_V_Zr"] = safe_log10(df["Nb_Ti_over_V_Zr"])

    optional_cols = ["PDAS", "delta_pct", "VEC", "Smix (J/mol·K)", "Hmix", "Er"]
    for c in optional_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    out_file = DATA_PROCESSED / "cleaned_engineered_data.csv"
    df.to_csv(out_file, index=False)

    print(f"Saved processed data to: {out_file}")
    return df


if __name__ == "__main__":
    load_and_preprocess()
