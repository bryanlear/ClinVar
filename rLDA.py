import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import plotly.express as px
import os
import glob

# --- File paths ---
output_dir = 'results/regularized_lda_latest/'
os.makedirs(output_dir, exist_ok=True)
data_path = 'clinvar_parquet/'

# --- Read and concatenate all parquet files ---
parquet_files = glob.glob(os.path.join(data_path, '*.parquet'))
df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

# --- Select features and target ---
features = ['ClinicalSignificance', 'ReviewStatus', 'VariantType', 'Chromosome']
target_col = 'ClinicalSignificance'
df = df[features].fillna('Missing')

# --- Reduce number of categories in target for clarity ---
top_n = 10
top_categories = df[target_col].value_counts().nlargest(top_n).index
y = df[target_col].where(df[target_col].isin(top_categories), other='Other')

# --- One-hot encode features (excluding target) ---
X = df.drop(columns=[target_col])
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X)

# --- Fit regularized LDA ---
lda = LDA(n_components=2, shrinkage='auto', solver='eigen')
X_lda = lda.fit_transform(X_encoded, y)

# --- Prepare DataFrame for Plotly ---
plot_df = pd.DataFrame(X_lda, columns=['LDA 1', 'LDA 2'])
plot_df['Category'] = y.values

# --- Plot with Plotly ---
fig = px.scatter(
    plot_df, x='LDA 1', y='LDA 2', color='Category',
    title=f'Regularized LDA on One-Hot Encoded Features (Top {top_n} ClinicalSignificance categories)',
    opacity=0.7,
    width=900, height=700
)
fig.update_layout(legend=dict(title=target_col, x=1.02, y=1, bordercolor="Black", borderwidth=1))
fig.write_html(os.path.join(output_dir, 'regularized_lda_plotly.html'))
fig.show()