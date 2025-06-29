# python mascara_model.py

import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

print("--- IMPLEMENTING MASCARA MODEL ---")

# Define parameters
years = [2020, 2021, 2022, 2023, 2024]
output_dir = 'results/mascara/'
os.makedirs(output_dir, exist_ok=True)

# Step 1: Load data from each year
print("Loading data from multiple years...")
data_by_year = {}
for year in tqdm(years, desc="Loading datasets"):
    data_path = f'data/processed/clinvar_{year}_parquet/'
    parquet_files = glob.glob(os.path.join(data_path, '*.parquet'))
    
    # Load a sample of data to keep memory usage manageable
    sample_size = 100000  # Based on memory constraints 
    dfs = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        if len(df) > sample_size // len(parquet_files):
            df = df.sample(sample_size // len(parquet_files))
        dfs.append(df)
    
    data_by_year[year] = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(data_by_year[year])} records from {year}")

# Step 2: Prepare features for analysis
print("\nPreparing features...")
features = ['ClinicalSignificance', 'ReviewStatus', 'VariantType', 'Chromosome']

# Create a common set of variants across years using variant identifiers
common_variants = set()
for year, df in data_by_year.items():
    df['VariantID'] = df['Chromosome'] + '_' + df['Position'].astype(str) + '_' + df['RefAllele'] + '_' + df['AltAllele']
    if len(common_variants) == 0:
        common_variants = set(df['VariantID'])
    else:
        common_variants &= set(df['VariantID'])

print(f"Found {len(common_variants)} variants common across all years")

# Filter to include only common variants
for year in years:
    data_by_year[year] = data_by_year[year][data_by_year[year]['VariantID'].isin(common_variants)]
    print(f"  Year {year}: {len(data_by_year[year])} records after filtering")

# Step 3: One-hot encode categorical features
print("\nOne-hot encoding features...")
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Combine data for fitting encoder
combined_features = pd.concat([data_by_year[year][features] for year in years])
combined_features = combined_features.fillna('Missing')
encoder.fit(combined_features)

# Transform each year's data
X_by_year = {}
for year in years:
    X_by_year[year] = encoder.transform(data_by_year[year][features].fillna('Missing'))
    print(f"  Year {year}: {X_by_year[year].shape} after encoding")

# Step 4: ASCA Decomposition
print("\nPerforming ASCA decomposition...")
# Calculate grand mean
X_combined = np.vstack([X_by_year[year] for year in years])
grand_mean = np.mean(X_combined, axis=0)

# Calculate year effect (factor A)
X_A = {}
for year in years:
    year_mean = np.mean(X_by_year[year], axis=0)
    X_A[year] = np.tile(year_mean - grand_mean, (X_by_year[year].shape[0], 1))

# Calculate residuals
E = {}
for year in years:
    E[year] = X_by_year[year] - np.tile(grand_mean, (X_by_year[year].shape[0], 1)) - X_A[year]

# Step 5: PCA on residuals (MASCARA)
print("\nPerforming PCA on residuals (MASCARA)...")
n_components = 10
pca = PCA(n_components=n_components)

# Combine residuals for PCA
E_combined = np.vstack([E[year] for year in years])
pca.fit(E_combined)

# Transform each year's residuals
scores_by_year = {}
for year in years:
    scores_by_year[year] = pca.transform(E[year])
    print(f"  Year {year}: {scores_by_year[year].shape} after PCA")

# Step 6: Save results
print("\nSaving results...")
# Save PCA model
import joblib
joblib.dump(pca, os.path.join(output_dir, 'mascara_pca_model.joblib'))

# Save explained variance
explained_variance_df = pd.DataFrame({
    'Component': [f'PC{i+1}' for i in range(n_components)],
    'ExplainedVariance': pca.explained_variance_ratio_,
    'CumulativeVariance': np.cumsum(pca.explained_variance_ratio_)
})
explained_variance_df.to_csv(os.path.join(output_dir, 'explained_variance.csv'), index=False)

# Save loadings
feature_names = encoder.get_feature_names_out()
loadings_df = pd.DataFrame(pca.components_.T, index=feature_names, 
                          columns=[f'PC{i+1}' for i in range(n_components)])
loadings_df.to_csv(os.path.join(output_dir, 'loadings.csv'))

# Save scores for each year
for year in years:
    scores_df = pd.DataFrame(scores_by_year[year], 
                            columns=[f'PC{i+1}' for i in range(n_components)])
    scores_df['VariantID'] = data_by_year[year]['VariantID'].values
    scores_df['Year'] = year
    scores_df.to_csv(os.path.join(output_dir, f'scores_{year}.csv'), index=False)

# Step 7: Visualize results
print("\nCreating visualizations...")
# Plot explained variance
plt.figure(figsize=(10, 6))
plt.bar(explained_variance_df['Component'], explained_variance_df['ExplainedVariance'])
plt.plot(explained_variance_df['Component'], explained_variance_df['CumulativeVariance'], 'ro-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Principal Component')
plt.savefig(os.path.join(output_dir, 'explained_variance.png'))

# Plot scores for each year (PC1 vs PC2)
plt.figure(figsize=(12, 10))
for i, year in enumerate(years):
    plt.scatter(scores_by_year[year][:, 0], scores_by_year[year][:, 1], 
               alpha=0.7, label=str(year))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('MASCARA Scores by Year (PC1 vs PC2)')
plt.legend()
plt.savefig(os.path.join(output_dir, 'scores_by_year.png'))

# Plot top loadings for PC1 and PC2
n_top = 20
plt.figure(figsize=(14, 10))
top_loadings_pc1 = loadings_df.iloc[:, 0].abs().nlargest(n_top)
top_loadings_pc2 = loadings_df.iloc[:, 1].abs().nlargest(n_top)

plt.subplot(1, 2, 1)
top_loadings_pc1.sort_values().plot(kind='barh')
plt.title(f'Top {n_top} Loadings for PC1')

plt.subplot(1, 2, 2)
top_loadings_pc2.sort_values().plot(kind='barh')
plt.title(f'Top {n_top} Loadings for PC2')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'top_loadings.png'))

print(f"\nMASCARA analysis complete. Results saved to {output_dir}")