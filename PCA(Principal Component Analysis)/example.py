
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load Data
data = load_breast_cancer

X = data.data
y = data.target

# 2. Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Original Dimensions:", X.shape)
print("Reduced Dimensions:", X_pca.shape)

print("\n--- Explained Variance Ratio ---")
print(pca.explained_variance_ratio_)

total_variance = sum(pca.explained_variance_ratio_) * 100

print(
    f"Total Variance Explained by 2 Components: "
    f"{total_variance:.2f}%"
)

# 5. Visualization
plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap="viridis",
    edgecolors="k"
)

# Axis Labels
plt.xlabel(
    f"Principal Component 1 "
    f"({pca.explained_variance_ratio_[0] * 100:.1f}%)"
)

plt.ylabel(
    f"Principal Component 2 "
    f"({pca.explained_variance_ratio_[1] * 100:.1f}%)"
)

plt.title("PCA Projection of Wine Dataset")

# 6. Add Legend
handles, _ = scatter.legend_elements()

plt.legend(
    handles,
    data.target_names,
    title="Classes"
)

# 7. Grid and Display
plt.grid(True, alpha=0.5)

plt.tight_layout()
plt.show()
