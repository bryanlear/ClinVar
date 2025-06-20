import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import prince
import joblib
import os

print("--- GENERATING RESULTS FROM SAVED MCA MODEL ---")
print("Loading saved MCA model...")
model_path = 'mca_model.joblib'
mca = joblib.load(model_path)
print("Model loaded successfully.")

df_mca = pd.read_parquet('clinvar_parquet/')[['ClinicalSignificance', 'ReviewStatus', 'VariantType', 'Chromosome']].fillna('Missing')

output_dir = 'results/mca/'
os.makedirs(output_dir, exist_ok=True)
print(f"All results will be saved to: {output_dir}")

print("\n--- Step 5: Visualizing ---")
print("\n--- Step 6: Saving Numerical Results ---")

eigenvalues = mca.eigenvalues_
eigenvalues_path = os.path.join(output_dir, 'mca_eigenvalues.csv')
pd.DataFrame(eigenvalues, index=[f'Component_{i+1}' for i in range(len(eigenvalues))], columns=['Eigenvalue']).to_csv(eigenvalues_path)
print(f"Eigenvalues saved to: {eigenvalues_path}")

column_coords_df = mca.column_coordinates(df_mca)
column_coords_path = os.path.join(output_dir, 'mca_column_coordinates.csv')
column_coords_df.to_csv(column_coords_path)
print(f"Column (category) coordinates saved to: {column_coords_path}")

principal_coordinates = mca.transform(df_mca)
transformed_data_path = os.path.join(output_dir, 'transformed_principal_coordinates.parquet')
principal_coordinates.to_parquet(transformed_data_path)
print(f"Transformed data (principal coordinates) saved to: {transformed_data_path}")

print("\nAnalysis complete.")