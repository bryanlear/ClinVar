import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

input_dir = 'results/'
output_dir = 'results/subsets/'
os.makedirs(output_dir, exist_ok=True)

bundled_file = os.path.join(input_dir, 'master_bundled_labels.parquet')
if not os.path.exists(bundled_file):
    print(f"Not found: {bundled_file}")
    exit(1)

print(f"Loading: {bundled_file}")
master_df = pd.read_parquet(bundled_file)
print(f"Loaded {len(master_df):,} variants")

bundled_cols = [col for col in master_df.columns if col.startswith('Bundled_ClinicalSignificance_')]
years = sorted([int(col.split('_')[-1]) for col in bundled_cols])

if len(years) < 2:
    exit(1)

first_year = min(years)
last_year = max(years)
first_col = f'Bundled_ClinicalSignificance_{first_year}'
last_col = f'Bundled_ClinicalSignificance_{last_year}'

print(f"Comparing {first_year} vs {last_year}")
print(f"Using columns: {first_col} and {last_col}")

master_df['Changed'] = master_df[first_col] != master_df[last_col]

changed_df = master_df[master_df['Changed'] == True].copy()
unchanged_df = master_df[master_df['Changed'] == False].copy()

print(f"\nDivision:")
print(f"  Total variants: {len(master_df):,}")
print(f"  Changed: {len(changed_df):,} ({len(changed_df)/len(master_df)*100:.1f}%)")
print(f"  Unchanged: {len(unchanged_df):,} ({len(unchanged_df)/len(master_df)*100:.1f}%)")

changed_parquet = os.path.join(output_dir, 'changed_variants.parquet')
changed_df.to_parquet(changed_parquet, index=False)

changed_csv = os.path.join(output_dir, 'changed_variants.csv')
changed_df.to_csv(changed_csv, index=False)

unchanged_parquet = os.path.join(output_dir, 'unchanged_variants.parquet')
unchanged_df.to_parquet(unchanged_parquet, index=False)

unchanged_csv = os.path.join(output_dir, 'unchanged_variants.csv')
unchanged_df.to_csv(unchanged_csv, index=False)

def analyze_subset(df, subset_name):
    analysis = {
        'subset_name': subset_name,
        'total_variants': len(df),
        'percentage_of_total': len(df) / len(master_df) * 100
    }
    
    if len(df) > 0:
        first_year_dist = df[first_col].value_counts().to_dict()
        last_year_dist = df[last_col].value_counts().to_dict()
        
        analysis[f'distribution_{first_year}'] = first_year_dist
        analysis[f'distribution_{last_year}'] = last_year_dist
        
        if 'GeneSymbol_2020' in df.columns:
            top_genes = df['GeneSymbol_2020'].value_counts().head(10).to_dict()
            analysis['top_genes'] = top_genes
        
        if 'VariantType_2020' in df.columns:
            variant_types = df['VariantType_2020'].value_counts().to_dict()
            analysis['variant_types'] = variant_types
    
    return analysis

changed_analysis = analyze_subset(changed_df, 'changed')
unchanged_analysis = analyze_subset(unchanged_df, 'unchanged')

if len(changed_df) > 0:
    change_patterns = changed_df.groupby([first_col, last_col]).size().reset_index(name='count')
    change_patterns = change_patterns.sort_values('count', ascending=False)
    
    print(f"\nChange patterns (first 10):")
    for _, row in change_patterns.head(10).iterrows():
        print(f"  {row[first_col]} → {row[last_col]}: {row['count']:,} variants")
    
    change_patterns_dict = {}
    for _, row in change_patterns.iterrows():
        key = f"{row[first_col]} → {row[last_col]}"
        change_patterns_dict[key] = int(row['count'])
    
    changed_analysis['change_patterns'] = change_patterns_dict
    
    change_patterns_path = os.path.join(output_dir, 'change_patterns.csv')
    change_patterns.to_csv(change_patterns_path, index=False)
    print(f"Saved to: {change_patterns_path}")

summary_report = {
    'analysis_date': datetime.now().isoformat(),
    'comparison_period': f"{first_year} to {last_year}",
    'total_variants': len(master_df),
    'changed_subset': changed_analysis,
    'unchanged_subset': unchanged_analysis
}

summary_path = os.path.join(output_dir, 'subset_analysis.json')
with open(summary_path, 'w') as f:
    json.dump(summary_report, f, indent=2, default=str)
print(f"Saved to: {summary_path}")

print(f"\n" + "="*50)
print("COMPLETE")
print("="*50)
print(f"Output directory: {output_dir}")
print(f"Output:")
print(f"changed_variants.parquet/.csv ({len(changed_df):,} variants)")
print(f"unchanged_variants.parquet/.csv ({len(unchanged_df):,} variants)")
print(f"change_patterns.csv")
print(f"subset_analysis.json")
