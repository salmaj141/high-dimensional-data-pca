# High-Dimensional Data Analysis & Dimensionality Reduction via PCA

## Overview
This project implements an end-to-end unsupervised learning pipeline focused on **Principal Component Analysis (PCA)** to uncover hidden structures in high-dimensional datasets. Using a multi-feature dataset of Italian wines produced by three different cultivators, the system standardizes raw feature scales, computes covariance structures, and projects high-dimensional metrics into an optimized lower-dimensional space to achieve distinct, unsupervised class separation.

## Core Features & Methodology

1. **Data Preprocessing & Standardization:**
   * Transformed a 13-dimensional feature space to have a zero-mean ($\mathbb{E}[x_i] = 0$)[cite: 6].
   * Normalized variance to 1 by dividing each feature by its standard deviation ($\frac{x_i}{\sigma_{x_i}}$), preventing higher-magnitude units from dominating the eigen-decomposition[cite: 6].

2. **Eigen-Decomposition & Covariance Analysis:**
   * Constructed the data correlation/covariance matrix $C$ from standardized inputs[cite: 6].
   * Extracted principal components by calculating eigenvalues and corresponding orthogonal eigenvectors to capture directions of maximum variance[cite: 6].

3. **Dimensionality Reduction & Projection:**
   * Generated a **Cumulative Explained Variance** graph to analyze the information retention of top-tier components[cite: 6].
   * Projected raw instances onto the primary principal components (**PC1** and **PC2**), successfully transforming overlapping raw features into linearly separable clusters representing the underlying cultivators[cite: 6].
   * Analyzed information degradation by contrasting primary projection with the lowest-variance component (**PC13**)[cite: 6].

## Repository Structure
* `wines.csv`: The raw dataset containing 13 chemical features across 3 cultivator classes[cite: 6].
* `pca_analysis.py`: Clean, modular Python implementation leveraging efficient matrix operations[cite: 6].
* `outputs/`: Generated plots showing cumulative variance and low-dimensional clustering visualizations[cite: 6].

## Tech Stack
* **Language:** Python
* **Libraries:** NumPy, Pandas, Matplotlib, SciPy[cite: 6]

## Key Findings
* **Variance Explained:** The first two principal components (PC1 and PC2) successfully capture a significant portion of the total variance, turning uninterpretable 13D noise into distinct visual clusters[cite: 6].
* **Feature Optimization:** Standardizing by standard deviation rather than variance proved essential to preserve geometric proportions across diverse physical measurement scales[cite: 6].
