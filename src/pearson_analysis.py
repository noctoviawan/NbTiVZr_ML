import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# FILE PATH
# =========================
file_path = "/content/NbTiVZr_ML/data/raw/NTVZ_LIB_ALL.csv"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(file_path)

# =========================
# OUTPUT FOLDER
# =========================
out_dir = "/content/NbTiVZr_ML/outputs/pearson"
os.makedirs(out_dir, exist_ok=True)

# =========================
# TARGET
# =========================
target = "H"   # change to "Hardness" if needed

# =========================
# CANDIDATE FEATURES
# Edit this list to match your real column names
# =========================
candidate_features = [
    # elemental composition
    "Nb", "Ti", "V", "Zr",

    # important ratios
    "Nb+Ti_over_V+Zr",
    "V_Zr",
    "Nb_Ti",
    "Ti_V",
    "Nb_Zr",
    "Ti_Zr",
    "Nb_V",
    "Nb_Ti_over_V_Zr",
    "Ti_V_over_Nb_Zr",
    "Ti_Zr_over_Nb_V",
    "Nb_over_Nb+Zr",
    "Zr_over_Nb+Zr",

    # log ratios
    "LogNb_Ti_over_V_Zr",
    "logNb_over_V_Zr",
    "Log_Nb_Ti",
    "Log_Ti_V",
    "Log_Nb_Zr",

    # materials / physics descriptors
    "PDAS",
    "VEC",
    "delta_pct",
    "Smix_J_over_molK",
    "Mixing_Enthalpy",
    "Estimated_Density",
    "Er"
]

# =========================
# KEEP ONLY EXISTING COLUMNS
# =========================
existing_features = [c for c in candidate_features if c in df.columns]

missing_features = [c for c in candidate_features if c not in df.columns]

print("Existing features used:")
print(existing_features)
print("\nMissing features skipped:")
print(missing_features)

if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found. Available columns:\n{list(df.columns)}")

# =========================
# BUILD DATASET FOR PEARSON
# =========================
cols = existing_features + [target]
data = df[cols].copy()

# keep numeric only
data = data.apply(pd.to_numeric, errors="coerce")

# optional: drop rows with missing target
data = data.dropna(subset=[target])

# =========================
# FULL PEARSON MATRIX
# =========================
corr = data.corr(method="pearson")

# save full matrix
corr.to_csv(os.path.join(out_dir, "pearson_full_matrix.csv"))

# =========================
# HARDNESS-ONLY CORRELATION TABLE
# =========================
hardness_corr = corr[[target]].drop(index=target).sort_values(by=target, ascending=False)
hardness_corr.to_csv(os.path.join(out_dir, "pearson_vs_hardness.csv"))

print("\nPearson correlation vs hardness:")
print(hardness_corr)

# =========================
# TRIANGULAR HEATMAP
# =========================
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr,
    mask=mask,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    annot=True,
    fmt=".2f",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8, "label": "Pearson correlation"}
)

plt.title("Pearson Correlation Analysis: Composition, Ratios, Descriptors, and Hardness", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "pearson_heatmap_all_features.png"), dpi=300, bbox_inches="tight")
plt.show()

# =========================
# OPTIONAL: BAR PLOT OF CORRELATION WITH HARDNESS
# =========================
plt.figure(figsize=(8, max(6, len(hardness_corr) * 0.35)))
hardness_corr.sort_values(by=target, ascending=True).plot(
    kind="barh",
    legend=False,
    figsize=(8, max(6, len(hardness_corr) * 0.35))
)
plt.xlabel("Pearson correlation with Hardness")
plt.ylabel("Feature")
plt.title("Pearson Correlation of Features with Hardness")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "pearson_vs_hardness_barplot.png"), dpi=300, bbox_inches="tight")
plt.show()
