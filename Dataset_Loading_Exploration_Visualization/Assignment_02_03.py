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
corr = df[features].corr()
corr_for_max = corr.mask(
    np.eye(corr.shape[0], dtype=bool)
)

max_corr = corr_for_max.stack().idxmax()
max_value = corr_for_max.stack().max()

print("Strongest positive correlation:")
print(max_corr)
print("Correlation value:", max_value)
plt.figure(figsize=(12, 9))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Wine Dataset")
plt.tight_layout()
plt.show()