import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import os

# --- File paths ---
output_dir = 'results/mca_2024'
scores_path = os.path.join(output_dir, 'transformed_principal_coordinates.parquet')
original_data_path = 'clinvar_202406_parquet/'

# --- Load MCA scores ---
scores = pd.read_parquet(scores_path)

# --- Load original data for target labels ---
df = pd.read_parquet(original_data_path)
if not scores.index.equals(df.index):
    df = df.loc[scores.index]

# --- Choose target variable for LDA ---
target_col = 'ClinicalSignificance'
y = df[target_col].fillna('Missing')

# --- Reduce number of categories ---
top_n = 10
top_categories = y.value_counts().nlargest(top_n).index
y_reduced = y.where(y.isin(top_categories), other='Other')

# --- Fit LDA on the MCA scores ---
lda = LDA(n_components=2)
X_lda = lda.fit_transform(scores, y_reduced)

# --- Prepare DataFrame for Plotly ---
plot_df = pd.DataFrame(X_lda, columns=['LDA 1', 'LDA 2'])
plot_df['Category'] = y_reduced.values

# --- Plot with Plotly ---
fig = px.scatter(
    plot_df, x='LDA 1', y='LDA 2', color='Category',
    title=f'LDA on MCA Principal Coordinates (Top {top_n} ClinicalSignificance categories)',
    opacity=0.7,
    width=900, height=700
)
fig.update_layout(legend=dict(title=target_col, x=1.02, y=1, bordercolor="Black", borderwidth=1))
fig.write_html(os.path.join(output_dir, 'lda_on_mca_plotly.html'))
fig.show()