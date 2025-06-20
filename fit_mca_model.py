import pandas as pd
import prince
import joblib 
import os

print("--- FITTING MCA MODEL (RUN ONCE) ---")

print("Loading Data...")
df = pd.read_parquet('clinvar_parquet/')

print("Cleaning Features...")
features_for_mca = ['ClinicalSignificance', 'ReviewStatus', 'VariantType', 'Chromosome']
df_mca = df[features_for_mca].copy()
for col in df_mca.columns:
    df_mca[col] = df_mca[col].fillna('Missing')

print("Performing MCA (this may take time)...")
n_components = 20
mca = prince.MCA(n_components=n_components, n_iter=3, random_state=42)
mca = mca.fit(df_mca)
print("MCA fitting complete.")

model_path = 'mca_model.joblib'
joblib.dump(mca, model_path)
print(f"Fitted MCA model saved to: {model_path}")

print("\nModel fitting and saving complete.")