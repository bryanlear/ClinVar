import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
import pickle
from tqdm import tqdm
import torch
import gc
import psutil  #monitor ram since this script is memory intensive! 

print("--- TRACKING VARIANT RECLASSIFICATION FROM 2020 TO 2025 ---")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Memory monitoring
def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / (1024 ** 3)  #GIGAbites
    print(f"Current memory usage: {memory_gb:.2f} GB")

years = [2020, 2021, 2022, 2023, 2024, 2025]
base_year = 2020
output_dir = 'results/reclassification/'
os.makedirs(output_dir, exist_ok=True)

data_dir = 'data/processed/' 

# Extract variant IDs from year's data
def extract_variant_ids(year):
    parquet_dir = os.path.join(data_dir, f"clinvar_{year}_parquet/")
    
    if not os.path.exists(parquet_dir):
        print(f"Directory not found: {parquet_dir}")
        return set()
    
    parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
    
    if not parquet_files:
        print(f"No parquet files found in {parquet_dir}")
        return set()
    
    variant_ids = set()
    for file in tqdm(parquet_files, desc=f"Extracting IDs from {year}"):
        df = pd.read_parquet(os.path.join(parquet_dir, file), columns=['Chromosome', 'Position', 'RefAllele'])
        # UNique identifier
        df['VariantID'] = df['Chromosome'] + '_' + df['Position'].astype(str) + '_' + df['RefAllele']
        variant_ids.update(df['VariantID'].tolist())
        del df
        gc.collect()
    
    return variant_ids

# Year's data in chunks but filtering for common variants
def process_year_data(year, common_variants):
    parquet_dir = os.path.join(data_dir, f"clinvar_{year}_parquet/")
    
    if not os.path.exists(parquet_dir):
        raise FileNotFoundError(f"Directory not found: {parquet_dir}")
    
    parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {parquet_dir}")

    result_df = pd.DataFrame()
    
    for file in tqdm(parquet_files, desc=f"Processing {year} data"):
        df = pd.read_parquet(os.path.join(parquet_dir, file))
    # Identifier
        df['VariantID'] = df['Chromosome'] + '_' + df['Position'].astype(str) + '_' + df['RefAllele']
        
        # Filter for common variants
        df = df[df['VariantID'].isin(common_variants)]
        if len(df) == 0:
            continue
        df['Year'] = year
        # Extract relevant columns 
        slim_df = df[['VariantID', 'ClinicalSignificance', 'ReviewStatus', 'GeneSymbol', 'VariantType']].copy()

        slim_df = slim_df.rename(columns={
            'ClinicalSignificance': f'ClinicalSignificance_{year}',
            'ReviewStatus': f'ReviewStatus_{year}',
            'GeneSymbol': f'GeneSymbol_{year}',
            'VariantType': f'VariantType_{year}'
        })
    
        if result_df.empty:
            result_df = slim_df
        else:
            result_df = pd.concat([result_df, slim_df], ignore_index=True)
        del df
        del slim_df
        gc.collect()
    
    return result_df

# Step 1: Find common variants across all years
print("\nStep 1: Finding common variants across years...")
print_memory_usage()
year_variant_ids = {}
available_years = []

for year in years:
    try:
        variant_ids = extract_variant_ids(year)
        if variant_ids:
            year_variant_ids[year] = variant_ids
            available_years.append(year)
            print(f"  Found {len(variant_ids)} variants in {year}")
        else:
            print(f"  No variants found for {year}")
    except Exception as e:
        print(f"  Error processing {year}: {e}")

print_memory_usage()

# Common variants between base year and each other year
if base_year not in year_variant_ids:
    if not available_years:
        print("No data available for any year. Exiting.")
        exit(1)
    base_year = min(available_years)
    print(f"Base year {base_year} not available. Using {base_year} instead.")

latest_year = max(available_years)
# Common variants between base and latest year
common_variants = year_variant_ids[base_year].intersection(year_variant_ids[latest_year])
print(f"Found {len(common_variants)} variants common to both {base_year} and {latest_year}")

# Free up memory
del year_variant_ids
gc.collect()
print_memory_usage()

# Step 2: Process each year's data filtering for common variants
print("\nStep 2: Processing data for each year...")

# Process base year
print(f"Processing base year {base_year}...")
base_df = process_year_data(base_year, common_variants)
print(f"  Processed {len(base_df)} variants from {base_year}")
print_memory_usage()

# Process latest year
print(f"Processing latest year {latest_year}...")
latest_df = process_year_data(latest_year, common_variants)
print(f"  Processed {len(latest_df)} variants from {latest_year}")
print_memory_usage()

# Merge base and latest year data
print("\nMerging data...")
master_df = base_df.merge(latest_df, on='VariantID', how='inner')
print(f"Master table created with {len(master_df)} common variants")
print_memory_usage()

# Clean up free up memory
del base_df
del latest_df
gc.collect()
print_memory_usage()

master_df.to_parquet(os.path.join(output_dir, 'master_longitudinal_table.parquet'))
print(f"Saved master table to {os.path.join(output_dir, 'master_longitudinal_table.parquet')}")

# VariantID as index
master_df = master_df.set_index('VariantID')

# Step 3: Train baseline UMAP model with base_year data
print("\nStep 3: Training baseline UMAP model...")
print(f"Reloading {base_year} data for UMAP...")
base_df = pd.DataFrame()

# Base year data in chunks
parquet_dir = os.path.join(data_dir, f"clinvar_{base_year}_parquet/")
parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]

for file in tqdm(parquet_files, desc=f"Loading {base_year} for UMAP"):
    df = pd.read_parquet(os.path.join(parquet_dir, file))
    df['VariantID'] = df['Chromosome'] + '_' + df['Position'].astype(str) + '_' + df['RefAllele']
    df = df[df['VariantID'].isin(common_variants)]
    if len(df) == 0:
        continue
    if base_df.empty:
        base_df = df
    else:
        base_df = pd.concat([base_df, df], ignore_index=True)
    
    # Clean up
    del df
    gc.collect()

print(f"  Loaded {len(base_df)} variants from {base_year} for UMAP")
print_memory_usage()

# Features for UMAP
features = ['ClinicalSignificance', 'ReviewStatus', 'VariantType', 'Chromosome']

# Categorical features
for feature in features:
    if feature in base_df.columns:
        base_df[feature] = base_df[feature].astype('category')

# One-hot 
base_features_encoded = pd.get_dummies(base_df[features])
print(f"  Created feature matrix with shape {base_features_encoded.shape}")
print_memory_usage()

# Sample data if it's more than 100,000 rows
MAX_UMAP_SAMPLES = 100000
if len(base_features_encoded) > MAX_UMAP_SAMPLES:
    print(f"  Sampling {MAX_UMAP_SAMPLES} variants for UMAP (out of {len(base_features_encoded)})")
    # Get indices for sampling
    sample_indices = np.random.choice(len(base_features_encoded), MAX_UMAP_SAMPLES, replace=False)
    # Sample the features
    base_features_encoded_sampled = base_features_encoded.iloc[sample_indices]
    # Keep track of sampled variant IDs
    sampled_variant_ids = base_df.iloc[sample_indices]['VariantID'].values
else:
    base_features_encoded_sampled = base_features_encoded
    sampled_variant_ids = base_df['VariantID'].values

# Convert to PyTorch tensor and move to GPU if available
if torch.cuda.is_available():
    print("Using GPU for UMAP computation")
    # Convert to PyTorch tensor
    features_tensor = torch.tensor(base_features_encoded_sampled.values, dtype=torch.float32).cuda()
    # Use PyTorch for computation
    base_features_encoded_np = features_tensor.cpu().numpy()
    # Free GPU memory
    del features_tensor
    torch.cuda.empty_cache()
else:
    base_features_encoded_np = base_features_encoded_sampled.values

# Free memory
del base_features_encoded
if 'base_features_encoded_sampled' in locals():
    del base_features_encoded_sampled
gc.collect()
print_memory_usage()

# Train UMAP model with optimized parameters for GPU and large datasets
print("  Training UMAP model...")
umap_model = UMAP(
    n_neighbors=15,  
    min_dist=0.1, 
    n_components=2, 
    random_state=42,
    metric='euclidean',  # Faster on GPU
    low_memory=False,
    n_epochs=200,       
    verbose=True
)

# Set a timeout for UMAP computation (30 minutes)
import signal
from contextlib import contextmanager

@contextmanager
def timeout(time):
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(time)
    
    try:
        yield
    except TimeoutError:
        print("  UMAP computation timed out, using PCA instead")
    finally:
        # Unregister signal so it won't be triggered if timeout not reached
        signal.signal(signal.SIGALRM, signal.SIG_IGN)

def raise_timeout(signum, frame):
    raise TimeoutError

# Try UMAP with timeout, fall back to PCA if it takes too long
try:
    with timeout(1800):  # 30 minutes timeout
        umap_embedding = umap_model.fit_transform(base_features_encoded_np)
    print("  UMAP training complete")
except Exception as e:
    print(f"  UMAP failed with error: {e}")
    print("  Falling back to PCA for dimensionality reduction")
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    umap_embedding = pca.fit_transform(base_features_encoded_np)
    print("  PCA complete")

print_memory_usage()

# Save model for later use
try:
    with open(os.path.join(output_dir, 'baseline_dim_reduction_model.pkl'), 'wb') as f:
        pickle.dump(umap_model, f)
except Exception as e:
    print(f"  Could not save model: {e}")

# DataFrame with embedding coordinates
if len(base_features_encoded_np) == len(base_df):
    # If we didn't sample
    umap_df = pd.DataFrame({
        'VariantID': base_df['VariantID'].values,
        'UMAP1': umap_embedding[:, 0],
        'UMAP2': umap_embedding[:, 1],
        'ClinicalSignificance': base_df['ClinicalSignificance'].values
    })
else:
    # If we sampled, we need to make sure all arrays have the same length
    # Create a mapping from VariantID to ClinicalSignificance
    clin_sig_map = dict(zip(base_df['VariantID'], base_df['ClinicalSignificance']))
    
    # Get clinical significance for sampled variants
    sampled_clin_sig = [clin_sig_map[var_id] for var_id in sampled_variant_ids]
    
    # Now create the DataFrame with arrays of the same length
    umap_df = pd.DataFrame({
        'VariantID': sampled_variant_ids,
        'UMAP1': umap_embedding[:, 0],
        'UMAP2': umap_embedding[:, 1],
        'ClinicalSignificance': sampled_clin_sig
    })

# Save the UMAP coordinates
umap_df.to_csv(os.path.join(output_dir, 'umap_coordinates.csv'), index=False)
print(f"  Saved UMAP coordinates to {os.path.join(output_dir, 'umap_coordinates.csv')}")

# Free memory
del base_features_encoded_np
if 'umap_model' in locals():
    del umap_model
gc.collect()
print_memory_usage()