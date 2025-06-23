#FOR_23_JUNE_END_of_DAY
# GOAL: Track reclassification from 2020 to 2025 

# *** Feature Engineering ***

# 1. Define a unique variant identifier (C-Position-RefAllele). where C = chromosome
# 2. Create Master Longitudinal Table by oarsing over the yearly clinvar files and aggregate them into a single DataFrame. It should be indexed by the unique variant identifier and contain all relevant features
# 3. Generate high dimensional feature vector with MASCARA for every variant (to determine). Baseline will be 2020 (I should experiment with multiple embedding methods)
# 4. Baseline UMAP model is trained with only 2020 variant feature data --> Obtain foundational 2D embedding space. The model is saved to use it at a later stage. 
# 5. Create baseline UMAP Plot with 2020 data and color each data point accordint to clinicalSignificance_2020 label (benign, VUS, whatever)

# *** Detect and Categorize Reclassification Events ***