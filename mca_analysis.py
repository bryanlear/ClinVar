import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import prince 
import os

output_dir = 'results/mca_2024/'
os.makedirs(output_dir, exist_ok=True)
print(f"All results will be saved to: {output_dir}")

print("\n--- Step 1: Loading Data ---")
df = pd.read_parquet('clinvar_202406_parquet/')
print(f"Successfully loaded {len(df)} records.")

print("\n--- Step 2: Cleaning and Selecting Features ---")
features_for_mca = [
    'ClinicalSignificance',
    'ReviewStatus',
    'VariantType',
    'Chromosome'
]
df_mca = df[features_for_mca].copy()
for col in df_mca.columns:
    df_mca[col] = df_mca[col].fillna('Missing')
print("Features selected and cleaned.")

print("\n--- Step 4: Performing MCA ---")
n_components = 20
mca = prince.MCA(
    n_components=n_components,
    n_iter=3,
    random_state=42
)

mca = mca.fit(df_mca)
print(f"MCA fitting complete for {n_components} components.")

print("\n--- Step 5: Visualizing ---")

explained_inertia_ratio = mca.eigenvalues_ / mca.total_inertia_

plt.figure(figsize=(12, 6))
plt.bar(range(len(explained_inertia_ratio)), explained_inertia_ratio, alpha=0.7, align='center', label='Individual explained inertia')
plt.step(range(len(explained_inertia_ratio)), np.cumsum(explained_inertia_ratio), where='mid', label='Cumulative explained inertia')
plt.ylabel('Explained Inertia Ratio')
plt.xlabel('Component Index')
plt.title('MCA Explained Inertia')
plt.legend(loc='best')
plt.tight_layout()

inertia_plot_path = os.path.join(output_dir, 'mca_explained_inertia.png')
plt.savefig(inertia_plot_path, dpi=300)
print(f"Explained inertia plot saved to: {inertia_plot_path}")
plt.show()

ax = mca.plot(df_mca)
ax = ax.properties(title='MCA of ClinVar Categories')  

biplot_path = os.path.join(output_dir, 'mca_biplot.png')
try:
    ax.save(biplot_path) 
    print(f"MCA biplot saved to: {biplot_path}")
except Exception as e:
    html_path = biplot_path.replace('.png', '.html')
    ax.save(html_path)
    print(f"Could not save as PNG, saved as HTML instead: {html_path}")

print("\n--- Step 6: Saving Numerical Results ---")

eigenvalues = mca.eigenvalues_
eigenvalues_path = os.path.join(output_dir, 'mca_eigenvalues.csv')
pd.DataFrame(eigenvalues, index=[f'Component_{i+1}' for i in range(len(eigenvalues))], columns=['Eigenvalue']).to_csv(eigenvalues_path)
print(f"Eigenvalues saved to: {eigenvalues_path}")

column_coords_df = mca.column_coordinates
column_coords_path = os.path.join(output_dir, 'mca_column_coordinates.csv')
column_coords_df.to_csv(column_coords_path)
print(f"Column (category) coordinates saved to: {column_coords_path}")

principal_coordinates = mca.transform(df_mca)
transformed_data_path = os.path.join(output_dir, 'transformed_principal_coordinates.parquet')
principal_coordinates.to_parquet(transformed_data_path)
print(f"Transformed data (principal coordinates) saved to: {transformed_data_path}")

print("\nAnalysis complete.")