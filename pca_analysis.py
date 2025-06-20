import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

output_dir = 'results/pca_latest/'
os.makedirs(output_dir, exist_ok=True)
print(f"All results will be saved to: {output_dir}")

print("\n--- Step 1: Loading Data ---")
# change here for different datasets after parsing the raw files
df = pd.read_parquet('clinvar_parquet/')
print(f"Successfully loaded {len(df)} records.")


# i am excluding genomic location and whatnot for simplicity for now...
print("\n--- Step 2: Cleaning and Selecting Features ---")
features_for_pca = [
    'ClinicalSignificance',
    'ReviewStatus',
    'VariantType',
    'Chromosome'
]
df_pca = df[features_for_pca].copy()
for col in df_pca.columns:
    df_pca[col] = df_pca[col].fillna('Missing')
print("Features selected and cleaned.")


# 1s and 0s one-hot
print("\n--- Step 3: One-Hot Encoding ---")
numerical_df = pd.get_dummies(df_pca, columns=features_for_pca, sparse=False)
print(f"Data converted to numerical matrix with shape: {numerical_df.shape}")


# pca
print("\n--- Step 4: Performing PCA ---")
scaler = StandardScaler()
scaled_numerical_df = scaler.fit_transform(numerical_df)
n_components = 20
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(numerical_df)
print(f"PCA fitting complete for {n_components} components.")


# plot explained variance 
print("\n--- Step 5: Analyzing and Visualizing ---")

plt.figure(figsize=(12, 6))
plt.bar(range(n_components), pca.explained_variance_ratio_, alpha=0.7, align='center', label='Individual explained variance')
plt.step(range(n_components), np.cumsum(pca.explained_variance_ratio_), where='mid', label='Cumulative explained variance')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Component Index')
plt.title('Scree Plot')
plt.legend(loc='best')
plt.xticks(range(n_components))
plt.tight_layout()

# same them plots 
scree_plot_path = os.path.join(output_dir, 'scree_plot.png')
plt.savefig(scree_plot_path, dpi=300)
print(f"Scree plot saved to: {scree_plot_path}")
plt.show()

# pcs
pc_df = pd.DataFrame(data=principal_components, columns=[f'PC_{i+1}' for i in range(n_components)])
pc_df['ClinicalSignificance'] = df['ClinicalSignificance'].fillna('Missing').values

top_sigs = ['Benign', 'Pathogenic', 'Uncertain_significance', 'Likely_benign', 'Likely_pathogenic']
pc_df_subset = pc_df[pc_df['ClinicalSignificance'].isin(top_sigs)]

plt.figure(figsize=(14, 10))
sns.scatterplot(x='PC_1', y='PC_2', hue='ClinicalSignificance', data=pc_df_subset, alpha=0.6, s=50)
plt.title('PCA of ClinVar Data (PC1 vs PC2)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.legend(title='Clinical Significance')
#scatter 
scatter_plot_path = os.path.join(output_dir, 'pca_scatter_plot.png')
plt.savefig(scatter_plot_path, dpi=300)
print(f"PCA scatter plot saved to: {scatter_plot_path}")
plt.show()


# save all data eigen values, vectors, whatever 
print("\n--- Step 6: Saving Numerical Results ---")

#  Eigenvalues (Variance)
eigenvalues = pca.explained_variance_
eigenvalues_path = os.path.join(output_dir, 'pca_eigenvalues.csv')
pd.DataFrame(eigenvalues, index=[f'PC_{i+1}' for i in range(n_components)], columns=['Eigenvalue']).to_csv(eigenvalues_path)
print(f"Eigenvalues (explained variance) saved to: {eigenvalues_path}")

# b) Eigenvectors (Loadings)
eigenvectors_df = pd.DataFrame(pca.components_, columns=numerical_df.columns, index=[f'PC_{i+1}' for i in range(n_components)])
eigenvectors_path = os.path.join(output_dir, 'pca_eigenvectors_loadings.csv')
eigenvectors_df.to_csv(eigenvectors_path)
print(f"Eigenvectors (component loadings) saved to: {eigenvectors_path}")

transformed_data_path = os.path.join(output_dir, 'transformed_principal_components.parquet')
pc_df.to_parquet(transformed_data_path)
print(f"Transformed data (principal components) saved to: {transformed_data_path}")

print("\nAnalysis complete.")