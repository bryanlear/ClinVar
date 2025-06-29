# Data Directory

This directory contains all data files for the ClinVar analysis project.

## Structure

```
data/
├── raw/                      # Raw, unprocessed data
│   └── clinvar_vcv_releases/ # Downloaded ClinVar XML files
│       ├── old_format/       # ClinVarVariationRelease files (2023-01 to 2025-06)
│       └── new_format/       # ClinVarVCVRelease files (2024-02 to 2025-06)
└── processed/                # Processed and cleaned data
    └── (analysis outputs)
```

## Raw Data

### ClinVar VCV Releases
- **Location**: `raw/clinvar_vcv_releases/`
- **Source**: NCBI ClinVar FTP server
- **Format**: Gzipped XML files
- **Size**: ~150+ GB total
- **Update Frequency**: Monthly (first Thursday of each month)

### File Naming Convention

**Old Format**: `ClinVarVariationRelease_YYYY-MM.xml.gz`
**New Format**: `ClinVarVCVRelease_YYYY-MM.xml.gz`

Where YYYY-MM is year and month of release.

## Processed Data

`processed/` contains:
- Parsed variant records
- Reclassification tracking data
- Analysis results
- Intermediate data files

## Data Management

### Download
Use the ClinVar toolkit to download raw data:
```bash
cd ../clinvar_toolkit/scripts
python download_clinvar_vcv_releases.py --download-dir ../data/raw/clinvar_vcv_releases
```

### Validation
Verify data integrity:
```bash
python clinvar_download_utils.py validate --download-dir ../data/raw/clinvar_vcv_releases --check-md5
```

### Storage Requirements
- **Minimum**: 200 GB free space
- **Recommended**: 300+ GB for processing workspace

## Data Usage Guidelines

1. **Raw Data**: Never modify files in `raw/` directory
2. **Processing**: Always work from copies in `processed/`
3. **Backup**: Consider backing up downloaded raw data
4. **Updates**: Re-run download script monthly for latest data

## File Formats

### XML Structure
ClinVar XML files contain:
- Variant records with clinical significance
- Review status and evidence
- Gene and genomic location information
- Submission and assertion details

### Processing Notes
- Files are gzipped and require decompression for parsing
- XML parsing should be done incrementally due to large file sizes
- Consider RAM usage when processing multiple files simultaneously
