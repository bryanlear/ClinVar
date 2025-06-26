# ClinVar: Tracking Variant Re-classification

Primary objective: To compare different temporal snapshots of the dataset (e.g., a recent vs. an older release) to identify, characterize, and understand the patterns behind variant re-classifications over time.

## Project Stages & Goals

### Stage 1: Data Ingestion & Processing Pipeline [✓]
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
--- 

*** Practice *** 

PCA alone, despite being used for continuous variables, was used on the latest dataset (with categorical features). The output was, as expected, not very interpretable (see `results/pca_latest` and the results are not even worth looking at). I switched to using MCA, which is a dimensionality reduction technique for categorical data. After it, I applied LDA to the resulting PCs to attempt to find PCs that would maximize the separation between the features.
Next, I moved on to implementing MASCARA, which is a two-stage process that involves decomposing the data according to the experimental design using ANOVA-Simultaneous Component Analysis (ASCA) and then applying PCA to the residuals from the first stage `results/mascara`.

---

*** TEST 1***

A master table was created by parsing over the yearly clinvar files and aggregating them into a single DataFrame. It is indexed by the unique variant identifier and contains all relevant features. The DataFrame is saved to `results/reclassification/master_longitudinal_table.parquet`.

For test 1, the master DataFrame was built using only 2020 (base) and 2025 (latest) data.  `4_reclassification_tracker.py` was ran on a RTX6000 Ada, 28 CPUs, 128GB RAM and 48 VRAM instance. Also, the UMAP coordinates produced by `3_mascara_umap.py` were used to plot the reclassification results in `5_UMAP_plot.py` (this is not the right way):

--- 

Table with variant reclassified 2020 vs. 2025 (a few are shown) ``results/reclassification/master_longitudinal_table.parquet`:

<div style="border: 1px solid black; padding: 10px;">

| Variant    | Clinical Significance (2020) | Review Status (2020)          | Gene (2020) | Variant Type (2020) | Clinical Significance (2025) | Review Status (2025)             | Gene (2025) | Variant Type (2025) |
|:-----------|:-----------------------------|:------------------------------|:------------|:--------------------|:-----------------------------|:---------------------------------|:------------|:--------------------|
| 1_930348_G | Likely_benign                | criteria_provided_single_submitter | SAMD11      | missense_variant    | Likely_benign                | criteria_provided_multiple_submitters_no_conflicts | SAMD11      | missense_variant    |
| 1_930388_GCCTCCCAGGAGCGTGACCGGTCCCAGCCATGAGCCCC | Benign                       | criteria_provided_single_submitter | SAMD11      | splice_donor_variant | Benign                       | criteria_provided_single_submitter | SAMD11      | splice_donor_variant |
| 1_934101_G | Benign                       | criteria_provided_single_submitter | SAMD11      | missense_variant    | Benign                       | criteria_provided_single_submitter | SAMD11      | missense_variant    |
| 1_934103_G | Benign                       | criteria_provided_single_submitter | SAMD11      | missense_variant    | Uncertain_significance       | criteria_provided_single_submitter | SAMD11      | missense_variant    |
| 1_945584_C | Benign                       | criteria_provided_single_submitter | NOC2L       | synonymous_variant  | Benign                       | criteria_provided_single_submitter | NOC2L       | synonymous_variant  |
| 1_953039_G | Benign                       | criteria_provided_single_submitter | NOC2L       | synonymous_variant  | Benign                       | criteria_provided_single_submitter | NOC2L       | synonymous_variant  |
| 1_953059_A | Benign                       | criteria_provided_single_submitter | NOC2L       | intron_variant      | Benign                       | criteria_provided_single_submitter | NOC2L       | intron_variant      |
| 1_954070_C | Benign                       | criteria_provided_single_submitter | NOC2L       | synonymous_variant  | Benign                       | criteria_provided_single_submitter | NOC2L       | synonymous_variant  |
| 1_955013_G | Likely_benign                | criteria_provided_single_submitter | NOC2L       | missense_variant    | Likely_benign                | criteria_provided_single_submitter | NOC2L       | 

--- 

</div>
<p align="center">
  <img src="results/reclassification/visualizations/umap_basic.png" alt="UMAP plot of reclassification" width="600" />
  <img src="results/reclassification/visualizations/umap_density.png" alt="Density plot of reclassification" width="600" />
</p>

---

Density Plot: ***Uniform Manifold Approximation and Projection (UMAP)*** is a dimensionality reduction technique that is used to visualize high-dimensional data in a lower-dimensional space. It is a non-linear technique that is able to preserve the local structure of the data, meaning that points that are close to each other in the high-dimensional space will also be close to each other in the lower-dimensional space.
Yellowish gradient indicates variants clustered in that UMAP space. Sort of like a hotspot where variants with similar characteristics are grouped together (similar features). The contour lines connect points of equal density. The closer the contour lines, the steeper the density gradient meaning that the rate of change of density is higher in that area. See ([Topolgical Geometry](https://en.wikipedia.org/wiki/Topological_geometry)) and [UMAP](https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction#Uniform_manifold_approximation_and_projection).
 
 ---

 A master DataFrame was built using *all available years:*

**Summary for ALL datasets 2020 → 2025:**

  - Total variants: 573,346
  - Changed: 169,722 (29.6%)
  - Unchanged: 403,624 (70.4%)

To simplify the problem, I bundled the ClinicalSignificance labels into ***4*** main categories (a 5th for missing values):

<div style="border: 1px solid black; padding: 10px;">
    <strong>Pathogenic</strong> - includes "Pathogenic" and similar<br>
    <strong>Benign</strong> - includes "Benign" and similars<br>
    <strong>VUS</strong> - Variants of Uncertain Significance<br>
    <strong>Other</strong> - Everything else that is not above<br>
    <strong>Unknown</strong> - Missing or empty values
</div>

---

After this, the dataset was divided into *changed* and *unchanged* subsets:

**Summary for changed variants (2020 → 2025):**
<div style="border: 1px solid black; padding: 10px;">
  <ul>
    <li>VUS → Pathogenic: 40,486 variants</li>
    <li>Benign → Pathogenic: 15,630 variants</li>
    <li>VUS → Benign: 10,451 variants</li>
    <li>Pathogenic → VUS: 7,046 variants</li>
    <li>Pathogenic → Benign: 7,007 variants</li>
    <li>Benign → VUS: 5,593 variants</li>
    <li>Other → Pathogenic: 1,915 variants</li>
    <li>Other → VUS: 1,475 variants</li>
    <li>Other → Benign: 922 variants</li>
    <li>Pathogenic → Other: 870 variants</li>
  </ul>
</div>

---

***TEST 2***

I decided to look into an appropriate method for encoding the data before running UMAP. 
***Temporal Embeddings:*** Vector representation of an entity that changes overtime. This is key because the classification of entities (variants) changes as more data comes in. Such embedding would capture the update (maybe).
***LSTM:*** Due to the vanishing gradient problem, a RNN will struggle to connect info over long sequences, its memory decays fast. LSTM gives it a longer memory which allows it to learn long-term dependencies.

---

<!DOCTYPE html>
<html>
<head>
<title>Model Config & Data Summary</title>
</head>
<body>

<h2>Label Encoding</h2>
<ul>
  <li><strong>Pathogenic:</strong> 2</li>
  <li><strong>Benign:</strong> 0</li>
  <li><strong>VUS (Variant of Unknown Significance):</strong> 4</li>
  <li><strong>Other:</strong> 1</li>
  <li><strong>Unknown:</strong> 3</li>
</ul>

<h2>Model Architecture</h2>
<ul>
  <li><strong>Vocabulary size:</strong> 5 (corresponds to the number of unique labels)</li>
  <li><strong>Embedding dimension:</strong> 64 (size of the vector representation for each label)</li>
  <li><strong>Hidden dimension:</strong> 128 (size of the hidden layers in the model)</li>
  <li><strong>Number of layers:</strong> 2 (number of recurrent or feed-forward layers)</li>
  <li><strong>Number of classes:</strong> 5 (the total number of distinct variant classifications the model predicts)</li>
</ul>

<h2>Data Summary</h2>
<ul>
  <li><strong>Training samples:</strong> 73933</li>
  <li><strong>Validation samples:</strong> 18484</li>
</ul>

</body>

</html>

---

<!DOCTYPE html>
<html>
<head>
<title>Model Performance Metrics</title>
</head>
<body>

<h2>Model Performance Summary</h2>
<ul>
  <li><strong>Training accuracy:</strong> 0.9227</li>
  <li><strong>Validation accuracy:</strong> 0.9200</li>
  <li><strong>Embedding dimension:</strong> 128</li>
  <li><strong>Variants processed:</strong> 92417</li>
</ul>

</body>
</html>

---