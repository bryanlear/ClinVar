import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import plotly.express as px
import plotly.graph_objects as go
from umap import UMAP
from sklearn.preprocessing import LabelEncoder
import json

input_dir = "results/embeddings"
subset_dir = "results/subsets"
output_dir = "results/embedding_visualizations"
os.makedirs(output_dir, exist_ok=True)

embeddings_file = os.path.join(input_dir, "variant_embeddings.npz")
if not os.path.exists(embeddings_file):
    print(f"Not found: {embeddings_file}")
    exit(1)

print(f"Loading embeddings from: {embeddings_file}")
embeddings_data = np.load(embeddings_file, allow_pickle=True)

train_embeddings = embeddings_data['train_embeddings']
train_variant_ids = embeddings_data['train_variant_ids']
train_labels = embeddings_data['train_labels']

val_embeddings = embeddings_data['val_embeddings']
val_variant_ids = embeddings_data['val_variant_ids']
val_labels = embeddings_data['val_labels']

all_embeddings = np.vstack([train_embeddings, val_embeddings])
all_variant_ids = np.concatenate([train_variant_ids, val_variant_ids])
all_labels = np.concatenate([train_labels, val_labels])

print(f"Loaded embeddings: {all_embeddings.shape}")
print(f"Total variants: {len(all_variant_ids)}")

label_names = ['Pathogenic', 'Benign', 'VUS', 'Other', 'Unknown']
label_encoder = LabelEncoder()
label_encoder.fit(label_names)

decoded_labels = label_encoder.inverse_transform(all_labels)

umap_model = UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42,
    metric='euclidean'
)

umap_coords = umap_model.fit_transform(all_embeddings)

umap_df = pd.DataFrame({
    'VariantID': all_variant_ids,
    'UMAP1': umap_coords[:, 0],
    'UMAP2': umap_coords[:, 1],
    'EndLabel': decoded_labels,
    'Split': ['train'] * len(train_embeddings) + ['val'] * len(val_embeddings)
})

changed_variants_file = os.path.join(subset_dir, "changed_variants.parquet")
if os.path.exists(changed_variants_file):
    changed_df = pd.read_parquet(changed_variants_file)

    bundled_cols = [col for col in changed_df.columns if col.startswith('Bundled_ClinicalSignificance_')]
    years = sorted([int(col.split('_')[-1]) for col in bundled_cols])

    if len(years) >= 2:
        first_year = min(years)
        last_year = max(years)

        merge_cols = ['VariantID', f'Bundled_ClinicalSignificance_{first_year}', f'Bundled_ClinicalSignificance_{last_year}']
        if all(col in changed_df.columns for col in merge_cols):
            changed_subset = changed_df[merge_cols].copy()
            changed_subset = changed_subset.rename(columns={
                f'Bundled_ClinicalSignificance_{first_year}': 'StartLabel',
                f'Bundled_ClinicalSignificance_{last_year}': 'EndLabel_Original'
            })

            umap_df = pd.merge(umap_df, changed_subset, on='VariantID', how='left')
            print(f"Merged with changed variants data ({first_year} to {last_year})")

print(f"Final UMAP dataframe: {len(umap_df)} variants")
print(f"Columns: {umap_df.columns.tolist()}")

print(umap_df[['UMAP1', 'UMAP2']].describe())

bundled_colors = {
    'Pathogenic': '#d62728',  # Red
    'Benign': '#1f77b4',  # Blue
    'VUS': '#9467bd',  # Purple
    'Other': '#8c564b',  # Brown
    'Unknown': '#7f7f7f',  # Gray
}
#endlabel
plt.figure(figsize=(12, 10))
if 'EndLabel' in umap_df.columns:
    unique_labels = umap_df['EndLabel'].unique()

    color_map = {label: bundled_colors.get(label, '#000000') for label in unique_labels}
    for label, color in color_map.items():
        mask = umap_df['EndLabel'] == label
        plt.scatter(
            umap_df.loc[mask, 'UMAP1'],
            umap_df.loc[mask, 'UMAP2'],
            c=color,
            label=label,
            alpha=0.7,
            s=20,
            edgecolors='none'
        )
    plt.legend(title='End Clinical Significance', loc='best', bbox_to_anchor=(1, 1))
else:
    plt.scatter(umap_df['UMAP1'], umap_df['UMAP2'], alpha=0.5, s=20)

plt.title('UMAP Projection of LSTM Embeddings (Changed Variants)')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'lstm_umap_basic.png'), dpi=300)
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
plt.title('Density of LSTM Embeddings in UMAP Space')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'lstm_umap_density.png'), dpi=300)
plt.close()

# Train/Val split visualize 
plt.figure(figsize=(12, 10))

train_mask = umap_df['Split'] == 'train'
val_mask = umap_df['Split'] == 'val'

plt.scatter(
    umap_df.loc[train_mask, 'UMAP1'],
    umap_df.loc[train_mask, 'UMAP2'],
    c='blue',
    alpha=0.6,
    s=15,
    label='Training Set',
    edgecolors='none'
)

plt.scatter(
    umap_df.loc[val_mask, 'UMAP1'],
    umap_df.loc[val_mask, 'UMAP2'],
    c='red',
    alpha=0.6,
    s=15,
    label='Validation Set',
    edgecolors='none'
)

plt.title('LSTM Embeddings: Training vs Validation Split')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'lstm_umap_train_val.png'), dpi=300)
plt.close()

if 'StartLabel' in umap_df.columns and 'EndLabel_Original' in umap_df.columns:
    umap_df['Transition'] = umap_df['StartLabel'] + ' → ' + umap_df['EndLabel_Original']
    unique_transitions = umap_df['Transition'].unique()
    transition_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_transitions)))

    plt.figure(figsize=(15, 12))

    for i, transition in enumerate(unique_transitions):
        mask = umap_df['Transition'] == transition
        plt.scatter(
            umap_df.loc[mask, 'UMAP1'],
            umap_df.loc[mask, 'UMAP2'],
            c=[transition_colors[i]],
            alpha=0.7,
            s=20,
            label=transition,
            edgecolors='none'
        )

    plt.title('LSTM Embeddings: Clinical Significance Transitions')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lstm_umap_transitions.png'), dpi=300, bbox_inches='tight')
    plt.close()

 # Heatmap reclassification
    reclass_matrix = pd.crosstab(
        umap_df['StartLabel'],
        umap_df['EndLabel_Original'],
        normalize='index'
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        reclass_matrix,
        annot=True,
        cmap='YlGnBu',
        fmt='.2%',
        linewidths=0.5
    )
    plt.title('Reclassification Patterns in Changed Variants')
    plt.xlabel('End Clinical Significance')
    plt.ylabel('Start Clinical Significance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lstm_reclassification_heatmap.png'), dpi=300)
    plt.close()

try:
    if 'EndLabel' in umap_df.columns:
        fig = px.scatter(
            umap_df,
            x='UMAP1',
            y='UMAP2',
            color='EndLabel',
            hover_data=['VariantID', 'Split'],
            title='Interactive UMAP Projection LSTM Embeddings',
            opacity=0.7,
            color_discrete_map=bundled_colors
        )
    else:
        fig = px.scatter(
            umap_df,
            x='UMAP1',
            y='UMAP2',
            hover_data=['VariantID', 'Split'],
            title='Interactive UMAP Projection LSTM Embeddings',
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

    fig.write_html(os.path.join(output_dir, 'lstm_umap_interactive.html'))

except Exception as e:
    print(f"Could not create interactive visualization: {e}")

if 'StartLabel' in umap_df.columns and 'EndLabel_Original' in umap_df.columns:
    try:
        fig = px.scatter(
            umap_df,
            x='UMAP1',
            y='UMAP2',
            color='Transition',
            hover_data=['VariantID', 'StartLabel', 'EndLabel_Original', 'Split'],
            title='Interactive LSTM Embeddings: Clinical Significance Transitions',
            opacity=0.7
        )

        fig.update_layout(
            width=1200,
            height=800,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01
            )
        )

        fig.write_html(os.path.join(output_dir, 'lstm_umap_transitions_interactive.html'))
    except Exception as e:
        print(f"Could not create visualization: {e}")
        
umap_coords_path = os.path.join(output_dir, 'lstm_umap_coordinates.csv')
umap_df.to_csv(umap_coords_path, index=False)

print(f"\n" + "="*60)
print("LSTM COMPLETE")
print("="*60)
print(f"Processed {len(umap_df)} variants")
