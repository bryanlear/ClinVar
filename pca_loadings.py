import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- File paths ---
output_dir = 'results/mca_latest/'
coords_path = os.path.join(output_dir, 'mca_column_coordinates.csv')
eigen_path = os.path.join(output_dir, 'mca_eigenvalues.csv')

# --- Load data ---
coords = pd.read_csv(coords_path, index_col=0)
eigen = pd.read_csv(eigen_path, index_col=0)

# --- Extract group/category for coloring ---
coords['group'] = coords.index.str.split('__').str[0]

# --- Assign a color to each group ---
unique_groups = coords['group'].unique()
palette = sns.color_palette('tab10', n_colors=len(unique_groups))
group2color = dict(zip(unique_groups, palette))
coords['color'] = coords['group'].map(group2color)

# --- Plot 1: MCA "PCA" plot (categories on first two components) ---
plt.figure(figsize=(12, 8))
for group in unique_groups:
    subset = coords[coords['group'] == group]
    plt.scatter(subset['0'], subset['1'], 
                label=group, 
                color=group2color[group], 
                alpha=0.7, s=60)

plt.xlabel(f'Component 1 ({eigen.iloc[0,0]:.2f} inertia)')
plt.ylabel(f'Component 2 ({eigen.iloc[1,0]:.2f} inertia)')
plt.title('MCA Category Map (First Two Components)')
plt.legend(title='Category Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'mca_category_map_grouped.png'), dpi=300, bbox_inches='tight')
plt.show()

# --- Plot 2: Loadings (absolute value of coordinates for first two components) ---
loadings = coords[['0', '1']].abs()
loadings['label'] = coords.index

# Remove rows where both loadings are close to zero ( I set it to 0.2 since there are many categories with very small loadings)
threshold = 0.2
filtered_loadings = loadings[(loadings['0'] >= threshold) | (loadings['1'] >= threshold)]
filtered_loadings = filtered_loadings.sort_values('0', ascending=False)

plt.figure(figsize=(10, max(8, 0.3 * len(filtered_loadings))))
plt.barh(filtered_loadings['label'], filtered_loadings['0'], color='skyblue', label='Component 1')
plt.barh(filtered_loadings['label'], filtered_loadings['1'], color='orange', alpha=0.5, label='Component 2')
plt.xlabel('Absolute Loading')
plt.title('MCA Category Loadings (First Two Components)\n(filtered, abs(loading) >= 0.05)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'mca_loadings_filtered.png'), dpi=300)
plt.show()