# Strategy has changed and all previous data/scripts/results have been stored in the `old_strategy` directory

<!DOCTYPE html>
<html>
<head>
LSTM-Based Model Workflow for Variant Reclassification
</head>
<body>

<hr>

<h3>1. Data Ingestion and Structuring</h3>
<ul>
    <li>Collect historical variant classification data with time-stamps (monthly) XML format.</li>
    <li>Each unique genetic variant forms an independent time series.</li>
</ul>

<hr>

<h3>2. Feature Engineering and Encoding</h3>
<ul>
    <li><strong>Primary Feature:</strong> Encode categorical classification labels (B, LB, VUS, LP, P) using <b>Entity Embeddings</b> for a dense numerical representation capturing semantic relationships.</li>
    <li><strong>Temporal Features:</strong> Add time-based features such as time elapsed since initial classification, time since last change, and binary flags for significant events (e.g., pre/post-2015 ACMG guidelines).</li>
</ul>

<hr>

<h3>3. Data Preparation for LSTM</h3>
<ul>
    <li>Use<b>sliding window approach</b> (e.g., via <code>tf.keras.utils.timeseries_dataset_from_array</code>) to transform each variant's time series into input-output samples.</li>
    <li>Input: Sequence of feature vectors (embedding + temporal features).</li>
    <li>Target: Probability distribution of classification at a future time step.</li>
</ul>

<hr>

<h3>4. Robust Validation and Hyperparameter Tuning</h3>
<ul>
    <li>Employ <b>Nested Cross-Validation</b> for unbiased performance estimation.</li>
    <li>Use <b>scikit-learn's TimeSeriesSplit</b> for both inner (hyperparameter tuning) and outer (evaluation) loops to maintain temporal order.</li>
    <li>Key hyperparameters to tune: LSTM's learning rate, number of hidden units, dropout rate, and input window size.</li>
</ul>

<hr>

<h3>5. Model Architecture and Training</h3>
<ul>
    <li>Develop an <b>LSTM-based architecture</b>, starting with a Stacked LSTM with an Attention mechanism.</li>
    <li>Final layer: Dense layer with <b>softmax activation</b> and five output neurons for probability distribution over classification tiers.</li>
    <li>Train by minimizing a proper scoring rule, such as <b>Categorical Cross-Entropy (Logloss)</b>.</li>
</ul>

<hr>

<h3>6. Evaluation and Calibration</h3>
<ul>
    <li>Evaluate final model using <b>Logloss</b> and the <b>Brier Score</b>.</li>
    <li>Assess probability reliability with <b>Calibration Plots</b>. Apply <b>Platt Scaling</b> if poorly calibrated.</li>
</ul>

<hr>

<h3>7. Deployment and Monitoring for Concept Drift</h3>
<ul>
    <li>Deploy the trained model within an <b>online learning framework</b>.</li>
    <li>Continuously monitor prediction error on new data. Use a system (e.g., inspired by LSTMDD) to detect significant error increases, signaling concept drift and triggering retraining.</li>
</ul>

<hr>

<h3>Strategies for Model Interpretability in a Clinical Context</h3>
<p>In a high-stakes clinical environment model interpretability is crucial for building trust and augmenting human expertise (goal is augmenting human expertise, not replacing it).</p>

<hr>

<h3>Human-in-the-Loop (HITL) Framework</h3>
<ul>
    <li>The model should function as a <b>clinical decision support tool</b> prioritizing VUSs for manual review by expert curators.</li>
    <li>It would flag variants with high predicted probability of reclassification to a pathogenic category allowing experts to focus on critical cases.</li>
</ul>

<hr>

<h3>Explainability Techniques</h3>
<ul>
    <li><strong>Post-Hoc Explanations:</strong> Apply model-agnostic methods like <b>LIME</b> or <b>SHAP</b> to explain individual predictions. These tools can highlight influential past classifications.</li>
    <li><strong>Intrinsic Interpretability:</strong>
        <ul>
            <li>Visualize <b>learned attention weights</b> to show which time steps the model "focused" on.</li>
            <li>Analyze <b>learned entity embeddings</b> (e.g., using t-SNE) to confirm semantic relationships between classification tiers.</li>
        </ul>
    </li>
</ul>

</body>
</html>

# 1. Data Ingestion and Structuring

## ClinVar VCV Release Downloader

Systematically downloadClinVar VCV release files from NCBI's FTP server. The script handles both the old format (`ClinVarVariationRelease`) and new format (`ClinVarVCVRelease`) files chronologically

## Overview

ClinVar provides monthly XML releases in two formats:
- **Old Format**: `ClinVarVariationRelease_YYYY-MM.xml.gz` (2023-01 to 2025-06)
- **New Format**: `ClinVarVCVRelease_YYYY-MM.xml.gz` (2024-02 to present)

This toolkit downloads files chronologically to build a complete historical dataset.

## Files

- `download_clinvar_vcv_releases.py` - Main download script
- `clinvar_download_utils.py` - Utility functions for status checking and validation
- `clinvar_download_config.py` - Configuration settings and file metadata

## Features

- ✅ **Chronological Downloads**: Processes files from earliest to latest
- ✅ **Resume Capability**: Skips already downloaded files
- ✅ **MD5 Verification**: Validates file integrity using checksums
- ✅ **Progress Tracking**: Real-time download progress and logging
- ✅ **Error Handling**: Retry logic with exponential backoff
- ✅ **Status Reporting**: Check download status and generate reports
- ✅ **File Validation**: Verify downloaded files for corruption

## Requirements

```bash
pip install requests
```

## Usage

### 1. Basic Download

Download all available ClinVar VCV release files:

```bash
python download_clinvar_vcv_releases.py
```

This will:
- Create `./clinvar_vcv_releases/` directory
- Download old format files to `./clinvar_vcv_releases/old_format/`
- Download new format files to `./clinvar_vcv_releases/new_format/`
- Skip files that already exist
- Verify MD5 checksums
- Log progress to `clinvar_download.log`

### 2. Custom Download Directory

```bash
python download_clinvar_vcv_releases.py --download-dir /path/to/your/data
```

### 3. Force Re-download

```bash
python download_clinvar_vcv_releases.py --no-skip-existing
```

### 4. Skip Checksum Verification

```bash
python download_clinvar_vcv_releases.py --no-verify-checksums
```

## Utility Commands

### Check Download Status

```bash
python clinvar_download_utils.py status --download-dir ./clinvar_vcv_releases
```

### Validate Downloaded Files

```bash
python clinvar_download_utils.py validate --download-dir ./clinvar_vcv_releases
```

### Validate with MD5 Checking

```bash
python clinvar_download_utils.py validate --download-dir ./clinvar_vcv_releases --check-md5
```

### Generate Download Report

```bash
python clinvar_download_utils.py report --download-dir ./clinvar_vcv_releases
```

### Save Report to File

```bash
python clinvar_download_utils.py report --download-dir ./clinvar_vcv_releases --output report.txt
```

## Configuration

Details:

```bash
python clinvar_download_config.py
```

This shows:
- Total expected download size (~150+ GB)
- Number of files to download
- Date ranges for each format
- Expected file sizes

## Directory Structure

```
clinvar_vcv_releases/
├── old_format/
│   ├── ClinVarVariationRelease_2023-01.xml.gz
│   ├── ClinVarVariationRelease_2023-02.xml.gz
│   └── ... (through 2025-06)
├── new_format/
│   ├── ClinVarVCVRelease_2024-02.xml.gz
│   ├── ClinVarVCVRelease_2024-03.xml.gz
│   └── ... (through 2025-06)
└── clinvar_download.log
```

## File Timeline

### Old Format (ClinVarVariationRelease)
- **Start**: January 2023 (`2023-01`)
- **End**: June 2025 (`2025-06`)
- **Total Files**: 30
- **Size Range**: ~2.2GB to ~4.5GB per file

### New Format (ClinVarVCVRelease)
- **Start**: February 2024 (`2024-02`)
- **End**: June 2025 (`2025-06`)
- **Total Files**: 17
- **Size Range**: ~3.2GB to ~4.6GB per file

## Processing Workflow

For building a complete historical dataset:

1. **Process Old Format First**: Start with `ClinVarVariationRelease_2023-01.xml.gz`
2. **Continue Chronologically**: Process through `ClinVarVariationRelease_2025-06.xml.gz`
3. **Switch to New Format**: Begin with `ClinVarVCVRelease_2024-02.xml.gz`
4. **Process to Present**: Continue through the latest available file

## Error Handling

The scripts include robust error handling:

- **Network Issues**: Automatic retry with exponential backoff
- **Partial Downloads**: Resume from where it left off
- **Corrupted Files**: MD5 verification catches corruption
- **Missing Files**: Clear reporting of what's missing

## Logging

All operations are logged to `clinvar_download.log`:

```
2025-06-29 10:30:15 - INFO - Starting ClinVar VCV release download
2025-06-29 10:30:16 - INFO - Downloading ClinVarVariationRelease_2023-01.xml.gz (attempt 1/3)
2025-06-29 10:32:45 - INFO - ✓ Downloaded ClinVarVariationRelease_2023-01.xml.gz (2185 MB)
2025-06-29 10:32:46 - INFO - ✓ MD5 checksum verified for ClinVarVariationRelease_2023-01.xml.gz
```

## Storage Requirements

- **Total Size**: ~150+ GB for complete dataset
- **Old Format**: ~95 GB (30 files)
- **New Format**: ~65 GB (17 files)
- **Recommended**: 200+ GB free space but I'd say 1 TB 🤣

## Best

1. **Verify Integrity**: Always run validation after downloads

## Troubleshooting

### Download Fails
```bash
# Check network connectivity
curl -I https://ftp.ncbi.nlm.nih.gov/pub/clinvar/xml/

# Retry with more verbose logging
python download_clinvar_vcv_releases.py --no-skip-existing
```

### Checksum Failures
```bash
# Re-download specific files
rm ./clinvar_vcv_releases/old_format/ClinVarVariationRelease_2023-01.xml.gz
python download_clinvar_vcv_releases.py
```

### Disk Space Issues
```bash
# Check available space
df -h

# Clean up partial downloads if needed 
find ./clinvar_vcv_releases -name "*.xml.gz" -size -1M -delete
```

## Integration with Analysis Pipelines

Files can be processed with various XML parsing libraries:

```python
import gzip
import xml.etree.ElementTree as ET
#e.g.,
with gzip.open('clinvar_vcv_releases/new_format/ClinVarVCVRelease_2025-06.xml.gz', 'rt') as f:
    tree = ET.parse(f)
    root = tree.getroot()
    # Process XML content...
```

## Support

NO support is offered since it's free! 🥱
