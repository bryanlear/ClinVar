# ClinVar: Tracking Variant Re-classification

Primary objective: To compare different temporal snapshots of the dataset (e.g., a recent vs. an older release) to identify, characterize, and understand the patterns behind variant re-classifications over time.

## Project Stages & Goals

### Stage 1: Data Ingestion & Processing Pipeline [x]
-   **Objective:** Establish a stable and memory-efficient pipeline to process large ClinVar VCF releases into a structured Apache Parquet format.
-   **Key Challenge & Resolution:** Initial attempt to parse ClinVar XML release failed due to insurmountable memory (OOM) errors on a 60 GB RAM instance. Had to pivot to using standard VCF format and `cyvcf2` library.

### Stage 2: Longitudinal Analysis [ ]
-   **Objective:** Compare both processed datasets ( 2025 release vs. 2024 release) to identify re-classified variants.
-   **Core Tasks:**
    1.  Identify set of variants that are present in both datasets
    2.  Pinpoint variants where `ClinicalSignificance` or `ReviewStatus` has changed between 2x points
    3.  Quantify and categorize re-classifications (e.g., count the number of VUS → Pathogenic, VUS → Benign,...,count_{-1})

### Stage 3: Pattern Discovery & Modeling [ ]
-   **Objective:** Analyze set of re-classified to find associated patterns
-   **is the re-classification linked to x gene(s), `VariantType`, `ReviewStatus`...?**

---

-   **Data Format:** ClinVar VCF (`.vcf.gz`) for GRCh38
-   **Version, file extensions and packages:**
    -   **Language:** Python 3.10+
    -   **Parsing:** `cyvcf2`
    -   **Data Manipulation:** `pandas`
    -   **Storage Format:** Apache Parquet
    -   **Analysis:** `scikit-learn`, `pandas`

---
### 2. Download Datasets
Download ClinVar VCF data for multiple timestamps

**Datasets are comprised of:** 

- .vcf.gz = main data
- .vcf.gz.tbi = index for fast access

-   **Latest Version (2025):**
-   Linux:
    ```bash 
    wget -O clinvar_latest.vcf.gz [https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz)
    wget -O clinvar_latest.vcf.gz.tbi [https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz.tbi](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz.tbi)
    ```
    Otherwise:

      ```bash
    curl -o clinvar_2025-01-06.vcf.gz https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2025/clinvar_20250601.vcf.gz
    curl -o clinvar_2025-01-06.vcf.gz.tbi https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2025/clinvar_20250601.vcf.gz.tbi
    ```  
-   **Version 2024:**
    ```bash
    curl -o clinvar_2024-01-07.vcf.gz https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2024/clinvar_20240107.vcf.gz
    curl -o clinvar_2024-01-07.vcf.gz.tbi https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2024/clinvar_20240107.vcf.gz.tbi
    ```
-   **Version 2023:**
    ```bash
    curl -o clinvar_2023-01-07.vcf.gz https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2023/clinvar_20230107.vcf.gz
    curl -o clinvar_2023-01-07.vcf.gz.tbi https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2023/clinvar_20230107.vcf.gz.tbi
    ```   
-   **Version 2022:**
    ```bash
    curl -o clinvar_2022-01-09.vcf.gz https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2022/clinvar_20220109.vcf.gz
    curl -o clinvar_2022-01-09.vcf.gz.tbi https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2022/clinvar_20220109.vcf.gz.tbi
    ```   
-   **Version 2021:**
    ```bash
    curl -o clinvar_2021-01-10.vcf.gz https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2021/clinvar_20210110.vcf.gz
    curl -o clinvar_2021-01-10.vcf.gz.tbi https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2021/clinvar_20210110.vcf.gz.tbi
    ```   

-   **Version 2020:**
    ```bash
    curl -o clinvar_2020-01-06.vcf.gz https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2020/clinvar_20200106.vcf.gz
    curl -o clinvar_2020-01-06.vcf.gz.tbi https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2020/clinvar_20200106.vcf.gz.tbi
    ```
```bash
# To process data into multiple .parquet chunks
python 1_parse_vcf_cyvcf2.py clinvar_<dataset_name>.vcf.gz clinvar_<dataset_name>_parquet/

```bash
python 1_parse_vcf_cyvcf2.py data/raw/datasets/2024/clinvar_2024-01-07.vcf.gz clinvar_2024_parquet/
python 1_parse_vcf_cyvcf2.py data/raw/datasets/2023/clinvar_2023-01-07.vcf.gz clinvar_2023_parquet/
python 1_parse_vcf_cyvcf2.py data/raw/datasets/2022/clinvar_2022-01-09.vcf.gz clinvar_2022_parquet/
python 1_parse_vcf_cyvcf2.py data/raw/datasets/2021/clinvar_2021-01-10.vcf.gz clinvar_2021_parquet/
python 1_parse_vcf_cyvcf2.py data/raw/datasets/2020/clinvar_2020-01-06.vcf.gz clinvar_2020_parquet/
```

After dividing the data, run the MASCARA model:
```bash
python mascara_model.py
```
PCA alone, despite being used for continuous variables, was used on the latest dataset (with categorical features). The output was, as expected, not very interpretable (see `results/pca_latest`, the results are not worth even looking at). I then switched to using MCA, which is a dimensionality reduction technique for categorical data. AFter MCA, I applied LDA to the resulting principal coordinates to attempt to discern any pattern among the classification labels for the year:

<p align="center">
  <img src="/results/mca_latest/lda_on_mca.png" alt="Pipeline" width="50%"/>
</p>

Note: the above results only apply to the latest dataset. I then moved on to implementing MASCARA, which is a two-stage process that involves decomposing the data according to the experimental design using ANOVA-Simultaneous Component Analysis (ASCA) and then applying PCA to the residuals from the first stage: 

<p align="center">
  <img src="results/mascara/explained_variance.png" alt="PCA on Residuals" width="500" />
  <img src="results/mascara/scores_by_year.png" alt="Scores by year projected onto PC" width="500" />
  <img src="results/mascara/top_loadings.png" alt="Top loadings for PC1 and PC2" width="500" />
</p>

### FOR_23_JUNE_END_of_DAY
***GOAL***: Track reclassification from 2020 to 2025 

*** Feature Engineering ***

1. Define a unique variant identifier (C-Position-RefAllele). where C = chromosome
2. Create Master Longitudinal Table by oarsing over the yearly clinvar files and aggregate them into a single DataFrame. It should be indexed by the unique variant identifier and contain all relevant features
3. Generate high dimensional feature vector with MASCARA for every variant (to determine). Baseline will be 2020 (I should experiment with multiple embedding methods)
4. Baseline UMAP model is trained with only 2020 variant feature data --> Obtain foundational 2D embedding space. The model is saved to use it at a later stage. 
5. Create baseline UMAP Plot with 2020 data and color each data point accordint to clinicalSignificance_2020 label (benign, VUS, whatever)

*** Detect and Categorize Reclassification Events ***


<p align="center">
  <img src="/results/mascara/umap/umap_density_by_year.png" alt="UMAP Density Plot" width="100%"/>
</p>
