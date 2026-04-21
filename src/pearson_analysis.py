import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# =========================
# Load data
# =========================
df = pd.read_csv("/content/NbTiVZr_ML/data/raw/NTVZ_LIB_ALL.csv")  # <-- update path if needed

# =========================
# Select columns
# (UPDATE if your column names differ)
# =========================
elements = ["Nb", "Ti", "V", "Zr"]   # your system
target = "H"  # hardness column (change to "Hardness" if needed)

cols = elements + [target]
data = df[cols].copy()

# =========================
# Pearson correlation
# =========================
corr = data.corr(method="pearson")

# =========================
# Mask upper triangle
# =========================
mask = np.triu(np.ones_like(corr, dtype=bool))

# =========================
# Create output folder
# =========================
os.makedirs("outputs/figures", exist_ok=True)

# =========================
# Plot
# =========================
plt.figure(figsize=(8,6))

sns.heatmap(
    corr,
    mask=mask,
    cmap="coolwarm",
    vmin=-1, vmax=1,
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"label": "Pearson Correlation"}
)

plt.title("Pearson Correlation between Composition and Hardness", fontsize=14)
plt.tight_layout()

# =========================
# Save figure
# =========================
plt.savefig("outputs/figures/pearson_elements_hardness.png", dpi=300)
plt.show()
