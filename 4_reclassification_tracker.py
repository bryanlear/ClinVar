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

print("--- TRACKING VARIANT RECLASSIFICATION FROM 2020 TO 2025 ---")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

years = [2020, 2021, 2022, 2023, 2024, 2025]
base_year = 2020
output_dir = 'results/reclassification/'
os.makedirs(output_dir, exist_ok=True)

data_dir = 'data/processed/' 

def load_year_data(year):
    parquet_dir = os.path.join(data_dir, f"clinvar_{year}_parquet/")
    
    if not os.path.exists(parquet_dir):
        raise FileNotFoundError(f"Directory not found: {parquet_dir}")
    
    dfs = []
    
    parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {parquet_dir}")
    
    for file in tqdm(parquet_files, desc=f"Loading {year} data"):
        df = pd.read_parquet(os.path.join(parquet_dir, file))
# Unique Identifier
        df['VariantID'] = df['Chromosome'] + '_' + df['Position'].astype(str) + '_' + df['RefAllele']
    # append year column to df
        df['Year'] = year
        dfs.append(df)
    
    return pd.concat(dfs)

# Master DataFrame
master_df = pd.DataFrame()
year_dfs = {}
available_years = []

for year in years:
    try:
        year_df = load_year_data(year)
        year_dfs[year] = year_df
        available_years.append(year)
        
        # "Relevant" features
        slim_df = year_df[['VariantID', 'ClinicalSignificance', 'ReviewStatus', 'GeneSymbol', 'VariantType']]
        # Rename to include year
        slim_df = slim_df.rename(columns={
            'ClinicalSignificance': f'ClinicalSignificance_{year}',
            'ReviewStatus': f'ReviewStatus_{year}',
            'GeneSymbol': f'GeneSymbol_{year}',
            'VariantType': f'VariantType_{year}'
        })
        
        # Merge with master table
        if master_df.empty:
            master_df = slim_df
        else:
            master_df = master_df.merge(slim_df, on='VariantID', how='outer')
        
        print(f"  Added {len(year_df)} variants from {year}")
        
        # Free memory
        del year_df
        gc.collect()
        
    except Exception as e:
        print(f"  Error loading {year} data: {e}")

# for debugging
if master_df.empty:
    print("No data was loaded. Please check the paths to your parquet files.")
    print(f"Expected data directory: {data_dir}")
    print("You may need to run the data processing scripts first:")
    print("  python 1_parse_vcf_cyvcf2.py data/raw/datasets/YEAR/clinvar_YEAR-MM-DD.vcf.gz data/processed/clinvar_YEAR_parquet/")
    exit(1)

if base_year not in available_years and available_years:
    base_year = min(available_years)
    print(f"Base year {base_year} not available. Using {base_year} instead.")

# VArianID index
master_df = master_df.set_index('VariantID')
print(f"Master table created with {len(master_df)} unique variants")

master_df.to_parquet(os.path.join(output_dir, 'master_longitudinal_table.parquet'))
# it will stop if there is only one year of data
if len(available_years) < 2:
    print("Need at least two years of data to track reclassification. Exiting.")
    exit(1)

# Step 2: Generate feature vectors using MASCARA for baseline year
print(f"\nGenerating feature vectors for baseline year {base_year}...")

# Load UMAP coordinates
try:
    umap_coords = pd.read_csv('results/mascara/umap/umap_coords_n30_d0.1.csv')
    base_year_coords = umap_coords[umap_coords['Year'] == base_year]
    print(f"  Loaded {len(base_year_coords)} UMAP coordinates for {base_year}")
except Exception as e:
    print(f"  Could not load existing UMAP coordinates: {e}")
    print("  Will generate new UMAP coordinates")

# Step 3: Train baseline UMAP model with base_year data
print("\nTraining baseline UMAP model...")

# Get the baseline year data
base_df = year_dfs[base_year]

# Extract features for UMAP
features = ['ClinicalSignificance', 'ReviewStatus', 'VariantType', 'Chromosome']

# Prepare categorical features
for feature in features:
    if feature in base_df.columns:
        base_df[feature] = base_df[feature].astype('category')

# One-hot encode categorical features
base_features_encoded = pd.get_dummies(base_df[features])

# Convert to PyTorch tensor and move to GPU if available
if torch.cuda.is_available():
    print("Using GPU for UMAP computation")
    # Convert to PyTorch tensor
    features_tensor = torch.tensor(base_features_encoded.values, dtype=torch.float32).cuda()
    # Use PyTorch for computation
    # Note: UMAP doesn't directly use PyTorch, but we can use GPU for preprocessing
    base_features_encoded_np = features_tensor.cpu().numpy()
else:
    base_features_encoded_np = base_features_encoded.values

# Train UMAP model with optimized parameters for GPU
umap_model = UMAP(
    n_neighbors=30, 
    min_dist=0.1, 
    n_components=2, 
    random_state=42,
    metric='euclidean',  # Faster on GPU
    low_memory=False,    # Use more memory for speed
    verbose=True         # Show progress
)
umap_embedding = umap_model.fit_transform(base_features_encoded_np)

# Save UMAP model for later use
with open(os.path.join(output_dir, 'baseline_umap_model.pkl'), 'wb') as f:
    pickle.dump(umap_model, f)

# Create DataFrame with UMAP coordinates
umap_df = pd.DataFrame({
    'VariantID': base_df['VariantID'],
    'UMAP1': umap_embedding[:, 0],
    'UMAP2': umap_embedding[:, 1],
    'ClinicalSignificance': base_df['ClinicalSignificance']
})

# Step 4: Create baseline UMAP plot
print("\nCreating baseline UMAP visualization...")

plt.figure(figsize=(12, 10))
scatter = sns.scatterplot(
    data=umap_df,
    x='UMAP1',
    y='UMAP2',
    hue='ClinicalSignificance',
    palette='viridis',
    alpha=0.7,
    s=50
)

plt.title(f'UMAP Projection of {base_year} Variants')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend(title='Clinical Significance', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'baseline_umap_{base_year}.png'), dpi=300)

# Step 5: Detect and categorize reclassification events
print("\nDetecting and categorizing reclassification events...")

# Find the latest available year
latest_year = max(available_years)

# Find variants present in both base and latest year
common_variants = set(year_dfs[base_year]['VariantID']) & set(year_dfs[latest_year]['VariantID'])
print(f"  Found {len(common_variants)} variants present in both {base_year} and {latest_year}")

# Create reclassification table
reclass_df = master_df.loc[list(common_variants)].copy()

# Identify variants whose clinical significance has changed
reclass_df['Reclassified'] = reclass_df[f'ClinicalSignificance_{base_year}'] != reclass_df[f'ClinicalSignificance_{latest_year}']
reclass_count = reclass_df['Reclassified'].sum()
print(f"  Identified {reclass_count} reclassified variants between {base_year} and {latest_year}")

# Categorize reclassification types
reclass_df['ReClassType'] = 'Unchanged'
reclass_df.loc[reclass_df['Reclassified'], 'ReClassType'] = (
    reclass_df[f'ClinicalSignificance_{base_year}'] + ' → ' + 
    reclass_df[f'ClinicalSignificance_{latest_year}']
)

# Save reclassification table
reclass_df.to_parquet(os.path.join(output_dir, 'reclassification_table.parquet'))

# Create summary of reclassification types
if reclass_count > 0:
    reclass_summary = reclass_df[reclass_df['Reclassified']]['ReClassType'].value_counts().reset_index()
    reclass_summary.columns = ['Reclassification', 'Count']
    reclass_summary.to_csv(os.path.join(output_dir, 'reclassification_summary.csv'), index=False)

    # Visualize reclassification patterns
    plt.figure(figsize=(14, 8))
    sns.barplot(data=reclass_summary.head(20), x='Count', y='Reclassification')
    plt.title(f'Top 20 Variant Reclassification Patterns ({base_year} → {latest_year})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reclassification_patterns.png'), dpi=300)
    
    # Create additional visualizations for deeper analysis
    print("\nCreating additional visualizations...")
    
    # Analyze reclassification by review status
    plt.figure(figsize=(14, 8))
    review_reclass = pd.crosstab(
        reclass_df[f'ReviewStatus_{base_year}'], 
        reclass_df['Reclassified']
    ).reset_index()
    review_reclass.columns = ['Review Status', 'Not Reclassified', 'Reclassified']
    review_reclass['Reclassification Rate'] = review_reclass['Reclassified'] / (review_reclass['Reclassified'] + review_reclass['Not Reclassified'])
    
    sns.barplot(data=review_reclass, x='Review Status', y='Reclassification Rate')
    plt.title(f'Reclassification Rate by Review Status ({base_year} → {latest_year})')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reclassification_by_review_status.png'), dpi=300)
else:
    print("  No reclassified variants found.")

print("\nReclassification tracking complete!")
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
import psutil  # For memory monitoring

print("--- TRACKING VARIANT RECLASSIFICATION FROM 2020 TO 2025 ---")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Memory monitoring function
def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / (1024 ** 3)  # Convert to GB
    print(f"Current memory usage: {memory_gb:.2f} GB")

# Define parameters
years = [2020, 2021, 2022, 2023, 2024, 2025]
base_year = 2020
output_dir = 'results/reclassification/'
os.makedirs(output_dir, exist_ok=True)

# Update the path to match your directory structure
data_dir = 'data/processed/'  # This is where your parquet files are stored

# Function to extract variant IDs from a year's data
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
        # Only read the columns needed for the ID
        df = pd.read_parquet(os.path.join(parquet_dir, file), columns=['Chromosome', 'Position', 'RefAllele'])
        # Create unique variant identifier
        df['VariantID'] = df['Chromosome'] + '_' + df['Position'].astype(str) + '_' + df['RefAllele']
        variant_ids.update(df['VariantID'].tolist())
        del df
        gc.collect()
    
    return variant_ids

# Function to process a year's data in chunks, filtering for common variants
def process_year_data(year, common_variants):
    parquet_dir = os.path.join(data_dir, f"clinvar_{year}_parquet/")
    
    if not os.path.exists(parquet_dir):
        raise FileNotFoundError(f"Directory not found: {parquet_dir}")
    
    parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {parquet_dir}")
    
    # Create an empty DataFrame to store results
    result_df = pd.DataFrame()
    
    for file in tqdm(parquet_files, desc=f"Processing {year} data"):
        # Read the parquet file
        df = pd.read_parquet(os.path.join(parquet_dir, file))
        
        # Create unique variant identifier
        df['VariantID'] = df['Chromosome'] + '_' + df['Position'].astype(str) + '_' + df['RefAllele']
        
        # Filter for common variants
        df = df[df['VariantID'].isin(common_variants)]
        
        # If there are no common variants in this chunk, skip
        if len(df) == 0:
            continue
        
        # Add year column
        df['Year'] = year
        
        # Extract only the columns we need
        slim_df = df[['VariantID', 'ClinicalSignificance', 'ReviewStatus', 'GeneSymbol', 'VariantType']].copy()
        
        # Rename columns to include year
        slim_df = slim_df.rename(columns={
            'ClinicalSignificance': f'ClinicalSignificance_{year}',
            'ReviewStatus': f'ReviewStatus_{year}',
            'GeneSymbol': f'GeneSymbol_{year}',
            'VariantType': f'VariantType_{year}'
        })
        
        # Append to result
        if result_df.empty:
            result_df = slim_df
        else:
            result_df = pd.concat([result_df, slim_df], ignore_index=True)
        
        # Clean up
        del df
        del slim_df
        gc.collect()
    
    return result_df

# Step 1: Find common variants across all years
print("\nStep 1: Finding common variants across years...")
print_memory_usage()

# First, extract variant IDs from each year
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

# Find common variants between base year and each other year
if base_year not in year_variant_ids:
    if not available_years:
        print("No data available for any year. Exiting.")
        exit(1)
    base_year = min(available_years)
    print(f"Base year {base_year} not available. Using {base_year} instead.")

# Find the latest available year
latest_year = max(available_years)

# Find common variants between base and latest year
common_variants = year_variant_ids[base_year].intersection(year_variant_ids[latest_year])
print(f"Found {len(common_variants)} variants common to both {base_year} and {latest_year}")

# Free up memory
del year_variant_ids
gc.collect()
print_memory_usage()

# Step 2: Process each year's data, filtering for common variants
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

# Clean up to free memory
del base_df
del latest_df
gc.collect()
print_memory_usage()

# Save master table
master_df.to_parquet(os.path.join(output_dir, 'master_longitudinal_table.parquet'))
print(f"Saved master table to {os.path.join(output_dir, 'master_longitudinal_table.parquet')}")

# Set VariantID as index
master_df = master_df.set_index('VariantID')

# Step 3: Train baseline UMAP model with base_year data
print("\nStep 3: Training baseline UMAP model...")

# We need to reload the base year data to get the features for UMAP
print(f"Reloading {base_year} data for UMAP...")
base_df = pd.DataFrame()

# Process the base year data in chunks
parquet_dir = os.path.join(data_dir, f"clinvar_{base_year}_parquet/")
parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]

for file in tqdm(parquet_files, desc=f"Loading {base_year} for UMAP"):
    # Read the parquet file
    df = pd.read_parquet(os.path.join(parquet_dir, file))
    
    # Create unique variant identifier
    df['VariantID'] = df['Chromosome'] + '_' + df['Position'].astype(str) + '_' + df['RefAllele']
    
    # Filter for common variants
    df = df[df['VariantID'].isin(common_variants)]
    
    # If there are no common variants in this chunk, skip
    if len(df) == 0:
        continue
    
    # Append to base_df
    if base_df.empty:
        base_df = df
    else:
        base_df = pd.concat([base_df, df], ignore_index=True)
    
    # Clean up
    del df
    gc.collect()

print(f"  Loaded {len(base_df)} variants from {base_year} for UMAP")
print_memory_usage()

# Extract features for UMAP
features = ['ClinicalSignificance', 'ReviewStatus', 'VariantType', 'Chromosome']

# Categorical features
for feature in features:
    if feature in base_df.columns:
        base_df[feature] = base_df[feature].astype('category')

# One-hot
base_features_encoded = pd.get_dummies(base_df[features])
print(f"  Created feature matrix with shape {base_features_encoded.shape}")
print_memory_usage()

# Convert to PyTorch tensofr and move to GPU
if torch.cuda.is_available():
    print("Using GPU for UMAP computation")
    features_tensor = torch.tensor(base_features_encoded.values, dtype=torch.float32).cuda()
    base_features_encoded_np = features_tensor.cpu().numpy()
    # Free GPU memory
    del features_tensor
    torch.cuda.empty_cache()
else:
    base_features_encoded_np = base_features_encoded.values

# Free memory
del base_features_encoded
gc.collect()
print_memory_usage()

print("  Training UMAP model...")
umap_model = UMAP(
    n_neighbors=30, 
    min_dist=0.1, 
    n_components=2, 
    random_state=42,
    metric='euclidean',  # Faster on GPU
    low_memory=False,   
    verbose=True         
)
umap_embedding = umap_model.fit_transform(base_features_encoded_np)
print("  UMAP training complete")
print_memory_usage()

# Save UMAP model for later use
with open(os.path.join(output_dir, 'baseline_umap_model.pkl'), 'wb') as f:
    pickle.dump(umap_model, f)

# Create DataFrame with UMAP coordinates
umap_df = pd.DataFrame({
    'VariantID': base_df['VariantID'].values,
    'UMAP1': umap_embedding[:, 0],
    'UMAP2': umap_embedding[:, 1],
    'ClinicalSignificance': base_df['ClinicalSignificance'].values
})

# Free memory
del base_features_encoded_np
del umap_model
gc.collect()
print_memory_usage()

# Step 4: Create baseline UMAP plot
print("\nStep 4: Creating baseline UMAP visualization...")

plt.figure(figsize=(12, 10))
scatter = sns.scatterplot(
    data=umap_df,
    x='UMAP1',
    y='UMAP2',
    hue='ClinicalSignificance',
    palette='viridis',
    alpha=0.7,
    s=50
)

plt.title(f'UMAP Projection of {base_year} Variants')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend(title='Clinical Significance', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'baseline_umap_{base_year}.png'), dpi=300)
plt.close()

# Free memory
del umap_df
gc.collect()
print_memory_usage()

# Step 5: Detect and categorize reclassification events
print("\nStep 5: Detecting and categorizing reclassification events...")

# Identify variants whose clinical significance has changed
master_df['Reclassified'] = master_df[f'ClinicalSignificance_{base_year}'] != master_df[f'ClinicalSignificance_{latest_year}']
reclass_count = master_df['Reclassified'].sum()
print(f"  Identified {reclass_count} reclassified variants between {base_year} and {latest_year}")

# Categorize reclassification types
master_df['ReClassType'] = 'Unchanged'
master_df.loc[master_df['Reclassified'], 'ReClassType'] = (
    master_df[f'ClinicalSignificance_{base_year}'] + ' → ' + 
    master_df[f'ClinicalSignificance_{latest_year}']
)

# Save reclassification table
master_df.to_parquet(os.path.join(output_dir, 'reclassification_table.parquet'))
print(f"  Saved reclassification table to {os.path.join(output_dir, 'reclassification_table.parquet')}")
print_memory_usage()

# Summary
if reclass_count > 0:
    reclass_summary = master_df[master_df['Reclassified']]['ReClassType'].value_counts().reset_index()
    reclass_summary.columns = ['Reclassification', 'Count']
    reclass_summary.to_csv(os.path.join(output_dir, 'reclassification_summary.csv'), index=False)
    print(f"  Saved reclassification summary to {os.path.join(output_dir, 'reclassification_summary.csv')}")

    # Plot reclassification patterns
    plt.figure(figsize=(14, 8))
    sns.barplot(data=reclass_summary.head(20), x='Count', y='Reclassification')
    plt.title(f'Top 20 Variant Reclassification Patterns ({base_year} → {latest_year})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reclassification_patterns.png'), dpi=300)
    plt.close()
    
    print("\nCreating additional visualizations...")
    
    # Reclassification by review status
    plt.figure(figsize=(14, 8))
    review_reclass = pd.crosstab(
        master_df[f'ReviewStatus_{base_year}'], 
        master_df['Reclassified']
    ).reset_index()
    review_reclass.columns = ['Review Status', 'Not Reclassified', 'Reclassified']
    review_reclass['Reclassification Rate'] = review_reclass['Reclassified'] / (review_reclass['Reclassified'] + review_reclass['Not Reclassified'])
    
    sns.barplot(data=review_reclass, x='Review Status', y='Reclassification Rate')
    plt.title(f'Reclassification Rate by Review Status ({base_year} → {latest_year})')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reclassification_by_review_status.png'), dpi=300)
    plt.close()
else:
    print("  No reclassified variants found.")

print("\nReclassification tracking complete!")
print_memory_usage()