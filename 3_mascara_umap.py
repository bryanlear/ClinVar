import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from tqdm import tqdm

# Import for 3D plotting (needed for set_zlabel method)
import mpl_toolkits.mplot3d  # noqa: F401

print("--- CREATING UMAP VISUALIZATION OF MASCARA RESULTS ---")

# Define parameters
years = [2020, 2021, 2022, 2023, 2024]
input_dir = 'results/mascara/'
output_dir = 'results/mascara/umap/'
os.makedirs(output_dir, exist_ok=True)

# Step 1: Load MASCARA scores from each year
print("Loading MASCARA scores from each year...")
scores_by_year = {}
all_scores = []
all_years = []

for year in tqdm(years, desc="Loading scores"):
    scores_path = os.path.join(input_dir, f'scores_{year}.csv')
    if os.path.exists(scores_path):
        df = pd.read_csv(scores_path)
        # Extract just the PC columns (exclude VariantID and Year)
        pc_cols = [col for col in df.columns if col.startswith('PC')]
        scores_by_year[year] = df[pc_cols].values
        
        # Add to combined arrays for UMAP fitting
        all_scores.append(df[pc_cols].values)
        all_years.extend([year] * len(df))
        
        print(f"  Loaded {len(df)} records from {year}")
    else:
        print(f"  Warning: No scores file found for {year}")

# Combine all scores for UMAP fitting
all_scores = np.vstack(all_scores)
all_years = np.array(all_years)

# Step 2: Fit UMAP
print("\nFitting UMAP model...")
# Parameters to try
n_neighbors_options = [15, 30, 50]  # Controls local vs global structure preservation
min_dist_options = [0.1, 0.5]       # Controls compactness of clusters

for n_neighbors in n_neighbors_options:
    for min_dist in min_dist_options:
        print(f"  Fitting UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}")
        
        umap_model = UMAP(n_neighbors=n_neighbors, 
                          min_dist=min_dist, 
                          n_components=2, 
                          random_state=42)
        
        umap_result = umap_model.fit_transform(all_scores)
        
        # Create DataFrame for plotting
        umap_df = pd.DataFrame({
            'UMAP1': umap_result[:, 0],
            'UMAP2': umap_result[:, 1],
            'Year': all_years
        })
        
        # Step 3: Visualize UMAP results
        plt.figure(figsize=(12, 10))
        
        # Plot with Seaborn for better aesthetics
        sns.scatterplot(
            data=umap_df, 
            x='UMAP1', 
            y='UMAP2', 
            hue='Year',
            palette='viridis',
            alpha=0.7,
            s=50
        )
        
        plt.title(f'UMAP Projection of MASCARA Results\n(n_neighbors={n_neighbors}, min_dist={min_dist})')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.legend(title='Year')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f'umap_n{n_neighbors}_d{min_dist}.png'), dpi=300)
        
        # Save the UMAP coordinates
        umap_df.to_csv(os.path.join(output_dir, f'umap_coords_n{n_neighbors}_d{min_dist}.csv'), index=False)

# Step 4: Create a 3D UMAP visualization
print("\nCreating 3D UMAP visualization...")
umap_3d = UMAP(n_neighbors=30, min_dist=0.1, n_components=3, random_state=42)
umap_3d_result = umap_3d.fit_transform(all_scores)

# Save 3D coordinates
umap_3d_df = pd.DataFrame({
    'UMAP1': umap_3d_result[:, 0],
    'UMAP2': umap_3d_result[:, 1],
    'UMAP3': umap_3d_result[:, 2],
    'Year': all_years
})
umap_3d_df.to_csv(os.path.join(output_dir, 'umap_3d_coords.csv'), index=False)

# Create interactive 3D plot with Plotly
try:
    import plotly.express as px
    
    fig = px.scatter_3d(
        umap_3d_df, x='UMAP1', y='UMAP2', z='UMAP3',
        color='Year', 
        title='3D UMAP Projection of MASCARA Results',
        opacity=0.7
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            zaxis_title='UMAP Dimension 3'
        ),
        width=900, height=800
    )
    
    fig.write_html(os.path.join(output_dir, 'umap_3d_interactive.html'))
    print("  3D interactive plot saved")
    
except ImportError:
    print("  Plotly not installed. Skipping interactive 3D plot.")
    
    # Create static 3D plot with Matplotlib instead
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for year in years:
        mask = umap_3d_df['Year'] == year
        ax.scatter(
            umap_3d_df.loc[mask, 'UMAP1'],
            umap_3d_df.loc[mask, 'UMAP2'],
            umap_3d_df.loc[mask, 'UMAP3'],
            label=str(year),
            alpha=0.7
        )
    
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.set_zlabel('UMAP Dimension 3')  # type: ignore
    ax.set_title('3D UMAP Projection of MASCARA Results')
    ax.legend(title='Year')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'umap_3d.png'), dpi=300)

# Step 5: Create density plots to better visualize overlapping points
print("\nCreating density plots...")

# Create a default 2D UMAP for density plots
umap_default = UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42)
umap_default_result = umap_default.fit_transform(all_scores)
umap_df = pd.DataFrame({
    'UMAP1': umap_default_result[:, 0],
    'UMAP2': umap_default_result[:, 1],
    'Year': all_years
})

plt.figure(figsize=(15, 10))

for i, year in enumerate(years):
    mask = umap_df['Year'] == year
    year_data = umap_df[mask]
    
    plt.subplot(2, 3, i+1)
    sns.kdeplot(
        data=year_data,
        x='UMAP1',
        y='UMAP2',
        fill=True,
        cmap='viridis',
        levels=10
    )
    plt.title(f'Year {year}')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'umap_density_by_year.png'), dpi=300)

print(f"\nUMAP visualizations complete. Results saved to {output_dir}")