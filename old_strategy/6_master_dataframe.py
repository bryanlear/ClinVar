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
import psutil
from functools import reduce

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}") # Just in case
else:
    print("Using CPU (Apple Silicon M2 Pro)")
# monitor RAM
def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / (1024 ** 3)
    print(f"Current usage: {memory_gb:.2f} GB")
    
    if memory_gb > 8:  # 16gb
        print("  ⚠️  High memory usage - process data in smaller chunks")
    elif memory_gb > 4:
        print("  📊 Moderate memory usage")
    else:
        print("  ✅ Memory usage is optimal")

years = [2020, 2021, 2022, 2023, 2024, 2025]
data_dir = 'data/processed/'
output_dir = 'results/'
os.makedirs(output_dir, exist_ok=True)

print(f"INput: {data_dir}")
print(f"Output: {output_dir}")
print(f"Years: {years}")

# variant id  = Chromosome_Position_RefAllele
def extract_variant_ids(year):
    parquet_dir = os.path.join(data_dir, f"clinvar_{year}_parquet/")
    if not os.path.exists(parquet_dir):
        print(f"Not found: {parquet_dir}")
        return set()
    
    parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
    
    if not parquet_files:
        print(f"No parquet found in {parquet_dir}")
        return set()
    
    variant_ids = set()
    for file in tqdm(parquet_files, desc=f"ID from {year}"):
        df = pd.read_parquet(os.path.join(parquet_dir, file), columns=['Chromosome', 'Position', 'RefAllele'])
        df['VariantID'] = df['Chromosome'] + '_' + df['Position'].astype(str) + '_' + df['RefAllele']
        variant_ids.update(df['VariantID'].tolist())
        del df
        gc.collect()
    
    return variant_ids

# filter variants
def process_year_data(year, target_variants):
    parquet_dir = os.path.join(data_dir, f"clinvar_{year}_parquet/")
    if not os.path.exists(parquet_dir):
        raise FileNotFoundError(f"Directory not found: {parquet_dir}")
    
    parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {parquet_dir}")
    
    result_df = pd.DataFrame()
    
    for file in tqdm(parquet_files, desc=f"Processing {year} data"):
        df = pd.read_parquet(os.path.join(parquet_dir, file))
        
        df['VariantID'] = df['Chromosome'] + '_' + df['Position'].astype(str) + '_' + df['RefAllele']

        df = df[df['VariantID'].isin(target_variants)]
        if len(df) == 0:
            continue
            
        columns_to_keep = ['VariantID', 'ClinicalSignificance', 'ReviewStatus', 'GeneSymbol', 'VariantType']
        slim_df = df[columns_to_keep].copy()
 
 # Include years in column name
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
            
        del df, slim_df
        gc.collect()
    
    if not result_df.empty:
        result_df = result_df.drop_duplicates(subset=['VariantID'], keep='first') #drop duplicates
    
    return result_df

# Variants present in **ALL** years
print_memory_usage()

year_variant_ids = {}
available_years = []

# Extract
for year in years:
    try:
        variant_ids = extract_variant_ids(year)
        if variant_ids:
            year_variant_ids[year] = variant_ids
            available_years.append(year)
            print(f"  Found {len(variant_ids):,} variants in {year}")
        else:
            print(f"  No variants found for {year}")
    except Exception as e:
        print(f"  Error processing {year}: {e}")

if not available_years:
    print("No data available for any year. Please, go home.")
    exit(1)

print(f"\nData available for {available_years}")

# Intersection
all_year_sets = [year_variant_ids[year] for year in available_years]
common_variants = reduce(lambda x, y: x.intersection(y), all_year_sets)

print(f"Found {len(common_variants):,} variants in ALL {len(available_years)} years")

# Free up memory
del year_variant_ids, all_year_sets
gc.collect()
print_memory_usage()

year_dataframes = {}

for year in available_years:
    print(f"Processing {year}...")
    year_df = process_year_data(year, common_variants)
    year_dataframes[year] = year_df
    print(f"  Processed {len(year_df):,} variants from {year}")
    print_memory_usage()
# 2020 is base. MErge
base_year = min(available_years)
master_df = year_dataframes[base_year].copy()
print(f"Starting with {base_year} as base: {len(master_df):,} variants")

for year in available_years[1:]:
    print(f"Merging {year}...")
    master_df = master_df.merge(year_dataframes[year], on='VariantID', how='inner')
    print(f"  After merging {year}: {len(master_df):,} variants")

print(f"\nFinal: {len(master_df):,} variants across {len(available_years)} years")
print(f"Columns: {list(master_df.columns)}")

# Clean up intermediate dataframes to free up memory
del year_dataframes
gc.collect()
print_memory_usage()

# .parquet
master_parquet_path = os.path.join(output_dir, 'master_dataframe.parquet')
master_df.to_parquet(master_parquet_path, index=False)
print(f"Saved to: {master_parquet_path}")
# .csv
master_csv_path = os.path.join(output_dir, 'master_dataframe.csv')
master_df.to_csv(master_csv_path, index=False)
print(f"Saved to: {master_csv_path}")

# Summary stats
summary_stats = {
    'total_variants': len(master_df),
    'years_included': available_years,
    'total_years': len(available_years),
    'columns': list(master_df.columns)
}
# Unique values
for year in available_years:
    col_name = f'ClinicalSignificance_{year}'
    if col_name in master_df.columns:
        unique_values = master_df[col_name].value_counts()
        summary_stats[f'clinical_significance_{year}'] = unique_values.to_dict()
import json
summary_path = os.path.join(output_dir, 'dataset_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary_stats, f, indent=2, default=str)
print(f"Saved summary statistics to: {summary_path}")

print("\n" + "="*60)
print("COMPREHENSIVE MASTER DATASET COMPLETE")
print("="*60)
print(f"Final dataset: {len(master_df):,} variants")
print(f"Years included: {', '.join(map(str, available_years))}")
print(f"Output directory: {output_dir}")
print(f"Files created:")
print(f"  - {master_parquet_path}")
print(f"  - {master_csv_path}")
print(f"  - {summary_path}")

first_year = min(available_years)
last_year = max(available_years)

first_col = f'ClinicalSignificance_{first_year}'
last_col = f'ClinicalSignificance_{last_year}'

if first_col in master_df.columns and last_col in master_df.columns:
    # Create reclassification summary
    reclassification_df = master_df[[first_col, last_col]].copy()
    reclassification_df['Changed'] = reclassification_df[first_col] != reclassification_df[last_col]

    total_variants = len(reclassification_df)
    changed_variants = reclassification_df['Changed'].sum()
    unchanged_variants = total_variants - changed_variants

    print(f"\nSummary ({first_year} → {last_year}):")
    print(f"  Total variants: {total_variants:,}")
    print(f"  Changed: {changed_variants:,} ({changed_variants/total_variants*100:.1f}%)")
    print(f"  Unchanged: {unchanged_variants:,} ({unchanged_variants/total_variants*100:.1f}%)")

    reclassification_path = os.path.join(output_dir, 'reclass_analysis.csv')
    reclassification_df.to_csv(reclassification_path, index=False)
    print(f" Saved to: {reclassification_path}")
    crosstab = pd.crosstab(reclassification_df[first_col], reclassification_df[last_col], margins=True)
    crosstab_path = os.path.join(output_dir, 'reclassification_crosstab.csv')
    crosstab.to_csv(crosstab_path)
    print(f"  Saved cross-tabulation to: {crosstab_path}")

def bundle_clinical_significance(value):
    if pd.isna(value) or value == '':
        return 'Unknown'

    value_lower = str(value).lower()

    if any(term in value_lower for term in ['pathogenic', 'likely pathogenic']):
        if 'likely pathogenic' in value_lower and 'pathogenic' not in value_lower.replace('likely pathogenic', ''):
            return 'Likely Pathogenic'
        else:
            return 'Pathogenic'
    elif any(term in value_lower for term in ['benign', 'likely benign']):
        if 'likely benign' in value_lower and 'benign' not in value_lower.replace('likely benign', ''):
            return 'Likely Benign'
        else:
            return 'Benign'
    elif 'uncertain' in value_lower or 'vus' in value_lower or 'variant of uncertain' in value_lower:
        return 'VUS'
    elif 'conflicting' in value_lower:
        return 'Conflicting'
    else:
        return 'Other'

bundled_df = master_df.copy()

for year in available_years:
    col_name = f'ClinicalSignificance_{year}'
    if col_name in bundled_df.columns:
        bundled_df[f'Bundled_{col_name}'] = bundled_df[col_name].apply(bundle_clinical_significance)

bundled_parquet_path = os.path.join(output_dir, 'master_bundled_labels.parquet')
bundled_df.to_parquet(bundled_parquet_path, index=False)
print(f"Saved bundled dataset to: {bundled_parquet_path}")

bundled_csv_path = os.path.join(output_dir, 'master_bundled_labels.csv')
bundled_df.to_csv(bundled_csv_path, index=False)
print(f"Saved bundled dataset to: {bundled_csv_path}")

bundled_summary = {}
for year in available_years:
    bundled_col = f'Bundled_ClinicalSignificance_{year}'
    if bundled_col in bundled_df.columns:
        bundled_counts = bundled_df[bundled_col].value_counts()
        bundled_summary[f'bundled_counts_{year}'] = bundled_counts.to_dict()

bundled_summary_path = os.path.join(output_dir, 'bundled_summary.json')
with open(bundled_summary_path, 'w') as f:
    json.dump(bundled_summary, f, indent=2, default=str)
print(f"Saved bundled summary to: {bundled_summary_path}")

print(f"\nMemory usage at the end:")
print_memory_usage()
