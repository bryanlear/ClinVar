import os
import pandas as pd
import numpy as pickle
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import plotly.express as px
import plotly.graph_objects as go

input_dir = "results/reclassification"
output_dir = os.path.join(input_dir, "visualizations")
os.makedirs(output_dir, exist_ok=True)

print("Loading UMAP coordinates...")
umap_df = pd.read_csv(os.path.join(input_dir, "umap_coordinates.csv"))
print(f"Loaded {len(umap_df)} points with columns: {umap_df.columns.tolist()}")
#load master tabnle
try:
    master_df = pd.read_parquet(os.path.join(input_dir, "master_longitudinal_table.parquet"))
    print(f"Loaded master table with {len(master_df)} rows")
    if 'VariantID' in master_df.columns:
        master_df_subset = master_df[['VariantID', '2020_ClinicalSignificance', '2025_ClinicalSignificance']]
        umap_df = pd.merge(umap_df, master_df_subset, on='VariantID', how='left')
        print("Merged with master table to get reclassification information")
        
        # Create reclassification flag
        umap_df['Reclassified'] = umap_df['2020_ClinicalSignificance'] != umap_df['2025_ClinicalSignificance']
        print(f"Found {umap_df['Reclassified'].sum()} reclassified variants")
except Exception as e:
    print(f"Could not load master table: {e}")
    print("Continuing with UMAP coordinates only")

print(umap_df[['UMAP1', 'UMAP2']].describe())

clinical_sig_colors = {
    'Pathogenic': '#d62728',  # Red
    'Likely pathogenic': '#ff7f0e',  # Orange
    'Uncertain significance': '#9467bd',  # Purple
    'Likely benign': '#2ca02c',  # Green
    'Benign': '#1f77b4',  # Blue
    'not provided': '#7f7f7f',  # Gray
    'Conflicting interpretations of pathogenicity': '#bcbd22',  # Yellow
    'other': '#8c564b',  # Brown
    'drug response': '#e377c2',  # Pink
    'risk factor': '#17becf',  # Cyan
}

# 1. Scatter plot with ClinicalSignificance label
print("\nCreating basic scatter plot...")
plt.figure(figsize=(12, 10))
if 'ClinicalSignificance' in umap_df.columns:
    unique_sig = umap_df['ClinicalSignificance'].unique()
 
    color_map = {sig: clinical_sig_colors.get(sig, '#000000') for sig in unique_sig}
    for sig, color in color_map.items():
        mask = umap_df['ClinicalSignificance'] == sig
        plt.scatter(
            umap_df.loc[mask, 'UMAP1'], 
            umap_df.loc[mask, 'UMAP2'],
            c=color,
            label=sig,
            alpha=0.7,
            s=5,
            edgecolors='none'
        )
    plt.legend(title='Clinical Significance', loc='best', bbox_to_anchor=(1, 1))
else:
    plt.scatter(umap_df['UMAP1'], umap_df['UMAP2'], alpha=0.5, s=5)

plt.title('UMAP Projection of ClinVar Variants')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'umap_basic.png'), dpi=300)
plt.close()

# Density plot
print("Creating density plot...")
plt.figure(figsize=(12, 10))
sns.kdeplot(
    data=umap_df,
    x='UMAP1',
    y='UMAP2',
    fill=True,
    cmap='viridis',
    levels=20,
    alpha=0.7
)
plt.title('Density of ClinVar Variants in UMAP Space')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'umap_density.png'), dpi=300)
plt.close()

# 3. Reclassification visualization
if 'Reclassified' in umap_df.columns:
    print("Creating reclassification visualization...")
    plt.figure(figsize=(12, 10))
    
    # Plot non-reclassified variants
    plt.scatter(
        umap_df.loc[~umap_df['Reclassified'], 'UMAP1'],
        umap_df.loc[~umap_df['Reclassified'], 'UMAP2'],
        c='lightgray',
        alpha=0.3,
        s=5,
        label='Not Reclassified'
    )
    
    # Plot reclassified variants
    plt.scatter(
        umap_df.loc[umap_df['Reclassified'], 'UMAP1'],
        umap_df.loc[umap_df['Reclassified'], 'UMAP2'],
        c='red',
        alpha=0.7,
        s=10,
        label='Reclassified'
    )
    
    plt.title('Reclassified Variants in UMAP Space (2020 to 2025)')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'umap_reclassification.png'), dpi=300)
    plt.close()
    
    # 4. Reclassification heatmap
    print("Creating reclassification heatmap...")
    if '2020_ClinicalSignificance' in umap_df.columns and '2025_ClinicalSignificance' in umap_df.columns:
        reclass_matrix = pd.crosstab(
            umap_df['2020_ClinicalSignificance'], 
            umap_df['2025_ClinicalSignificance'],
            normalize='index'
        )
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(
            reclass_matrix,
            annot=True,
            cmap='YlGnBu',
            fmt='.2%',
            linewidths=0.5
        )
        plt.title('Reclassification Patterns (2020 to 2025)')
        plt.xlabel('2025 Clinical Significance')
        plt.ylabel('2020 Clinical Significance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'reclassification_heatmap.png'), dpi=300)
        plt.close()

# 5. Plotly (check html in results/reclassification/visualizations/)
print("Creating interactive visualization...")
try:
    if 'ClinicalSignificance' in umap_df.columns:
        fig = px.scatter(
            umap_df, 
            x='UMAP1', 
            y='UMAP2',
            color='ClinicalSignificance',
            hover_data=['VariantID'],
            title='Interactive UMAP Projection of ClinVar Variants',
            opacity=0.7,
            color_discrete_map=clinical_sig_colors
        )
    else:
        fig = px.scatter(
            umap_df, 
            x='UMAP1', 
            y='UMAP2',
            hover_data=['VariantID'],
            title='Interactive UMAP Projection of ClinVar Variants',
            opacity=0.7
        )
    
    fig.update_layout(
        width=1000,
        height=800,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        )
    )
    
    fig.write_html(os.path.join(output_dir, 'umap_interactive.html'))
    print(f"Interactive visualization saved to {os.path.join(output_dir, 'umap_interactive.html')}")
except Exception as e:
    print(f"Could not create interactive visualization: {e}")

if 'Reclassified' in umap_df.columns:
    try:
        # Create a new column for hover information
        umap_df['Reclassification'] = umap_df.apply(
            lambda x: f"From: {x['2020_ClinicalSignificance']}<br>To: {x['2025_ClinicalSignificance']}" 
            if x['Reclassified'] else "Not reclassified", 
            axis=1
        )
        
        fig = px.scatter(
            umap_df, 
            x='UMAP1', 
            y='UMAP2',
            color='Reclassified',
            hover_data=['VariantID', '2020_ClinicalSignificance', '2025_ClinicalSignificance'],
            title='Reclassified Variants in UMAP Space (2020 to 2025)',
            opacity=0.7,
            color_discrete_map={True: 'red', False: 'lightgray'}
        )
        
        fig.update_layout(
            width=1000,
            height=800
        )
        
        fig.write_html(os.path.join(output_dir, 'umap_reclassification_interactive.html'))
        print(f"Interactive reclassification visualization saved to {os.path.join(output_dir, 'umap_reclassification_interactive.html')}")
    except Exception as e:
        print(f"Could not create interactive reclassification visualization: {e}")

print(f"\nAll visualizations saved to {output_dir}")