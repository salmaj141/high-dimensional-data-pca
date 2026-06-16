import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 0. Create outputs folder if it doesn't exist
# ==========================================
output_dir = "outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ============================
# 1. Load the data
# ============================
df = pd.read_csv("wines.csv")

# First column - class
labels = df.iloc[:, 0]

# Extract features
X = df.iloc[:, 1:].values        # shape: (n_samples, 13)

# ============================
# 2. Plot Feature 1 vs Feature 2
# ============================
feature1 = X[:, 0]   # first feature
feature2 = X[:, 1]   # second feature

plt.figure(figsize=(7, 5))

# Define colors for each class
colors = {1: 'red', 2: 'green', 3: 'blue'}

# Plot each class separately for clearer visualization
for cls in sorted(labels.unique()):
    idx = (labels == cls)
    plt.scatter(feature1[idx],
                feature2[idx],
                color=colors[cls],
                label=f'Class {cls}',
                alpha=0.7)

plt.xlabel("Alcohol")
plt.ylabel("Malic.Acid")
plt.title("Wine Dataset: Feature 1 vs Feature 2")
plt.legend(title="Grower (class)")
plt.grid(True)
plt.tight_layout()
# Saved inside outputs folder
plt.savefig(os.path.join(output_dir, "feature1_vs_feature2.png"), dpi=300)
plt.show()

# ============================
# 3.a Center the data (mean = 0 for each feature)
# ============================
# Compute mean of each feature (13-dimensional vector)
feature_means = np.mean(X, axis=0)

# Subtract the mean from each sample (centering)
X_centered = X - feature_means

# ============================
# 3.b. Standardize the data (variance = 1)
# ============================
# Compute standard deviation of each feature
stds = np.std(X_centered, axis=0, ddof=1)  # unbiased std (n-1)
# Avoid division by zero if a feature has zero variance
stds[stds == 0] = 1
# Standardization: x_i → x_i / σ_i
X_standardized = X_centered / stds


# ============================
# 4. Compute correlation matrix C
# ============================
C = np.corrcoef(X_standardized, rowvar=False)
print("Correlation matrix shape:", C.shape)
eigvals, eigvecs = np.linalg.eig(C)
# Sort eigenvalues (and eigenvectors) in descending order
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# ============================
# 5. Compute explained variance ratios
# ============================
total_variance = np.sum(eigvals)
explained_variance_ratio = eigvals / total_variance
cumulative_variance = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(8, 5))
components = np.arange(1, len(eigvals) + 1)
plt.plot(components, cumulative_variance, marker='o')
plt.xlabel("Eigen values Number")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance by PCA Components")
plt.xticks(components)
plt.grid(True)
plt.tight_layout()
# Saved inside outputs folder
plt.savefig(os.path.join(output_dir, "cumulative_variance.png"), dpi=300)
plt.show()

# Print required values
print("Explained variance ratio for each component:")
print(explained_variance_ratio)

print("\nCumulative explained variance:")
print(cumulative_variance)

print("\nExplained variance by first two PCs:")
print("PC1 + PC2 =", cumulative_variance[1])


# ============================
# 6. Project data onto PC1 and PC2
# ============================
PC1 = X_standardized @ eigvecs[:, 0]   # projection on the first principal component
PC2 = X_standardized @ eigvecs[:, 1]   # projection on the second principal component

plt.figure(figsize=(7, 6))

for cls in sorted(labels.unique()):
    idx = (labels == cls)
    plt.scatter(PC1[idx], PC2[idx],
                color=colors[cls],
                label=f"Class {cls}",
                alpha=0.7)

plt.xlabel("PC1 (first principal component)")
plt.ylabel("PC2 (second principal component)")
plt.title("Projection of Wine Data onto PC1 and PC2")
plt.legend()
plt.grid(True)
plt.tight_layout()
# Saved inside outputs folder
plt.savefig(os.path.join(output_dir, "Proj_PC1_PC2.png"), dpi=300)
plt.show()

# ============================
# 7. Project data onto PC1 and PC13
# ============================
PC1  = X_standardized @ eigvecs[:, 0]      # first principal component
PC13 = X_standardized @ eigvecs[:, -1]     # last principal component

plt.figure(figsize=(7, 6))

for cls in sorted(labels.unique()):
    idx = (labels == cls)
    plt.scatter(PC1[idx], PC13[idx],
                color=colors[cls],
                label=f"Class {cls}",
                alpha=0.7)

plt.xlabel("PC1 (first principal component)")
plt.ylabel("PC13 (last principal component)")
plt.title("Projection of Wine Data onto PC1 and PC13")
plt.legend()
plt.grid(True)
plt.tight_layout()
# Saved inside outputs folder
plt.savefig(os.path.join(output_dir, "Proj_PC1_PC13.png"), dpi=300)
plt.show()

print(f"\nAll plots have been successfully saved to the '{output_dir}' directory!")