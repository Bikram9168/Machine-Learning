import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine

wine = load_wine()

df = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

df["target"] = wine.target

features = wine.feature_names

# Create correlation matrix
corr = df[features].corr().copy()

# Remove diagonal self-correlations
np.fill_diagonal(corr.values, np.nan)

# Find strongest positive correlation
max_corr = corr.stack().idxmax()
max_value = corr.stack().max()

print("Strongest positive correlation:")
print(max_corr)

print("Correlation value:", max_value)

# Generate heatmap
plt.figure(figsize=(12, 9))

sns.heatmap(
    df[features].corr(),   # use original correlation matrix for heatmap
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Wine Dataset")

plt.tight_layout()
plt.show()
