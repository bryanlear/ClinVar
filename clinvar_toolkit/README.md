# ClinVar Data Toolkit

A comprehensive Python toolkit for downloading, processing, and analyzing ClinVar VCV release files from NCBI's FTP server.

## Overview

This toolkit provides systematic access to ClinVar's monthly XML releases in both old and new formats, enabling researchers to build complete historical datasets for genomic variant analysis.

## Directory Structure

```
clinvar_toolkit/
├── README.md                 # This file
├── scripts/                  # Main toolkit scripts
│   ├── download_clinvar_vcv_releases.py  # Primary download script
│   └── clinvar_download_utils.py         # Utility functions
├── config/                   # Configuration files
│   └── clinvar_download_config.py        # Download settings and metadata
├── examples/                 # Usage examples
│   └── example_usage.py                  # Example analysis script
└── docs/                     # Documentation
    └── (additional documentation files)
```

## Quick Start

### 1. Download ClinVar Data

```bash
cd clinvar_toolkit/scripts
python download_clinvar_vcv_releases.py --download-dir ../../data/raw/clinvar_vcv_releases
```

### 2. Check Download Status

```bash
python clinvar_download_utils.py status --download-dir ../../data/raw/clinvar_vcv_releases
```

### 3. Analyze Downloaded Files

```bash
cd ../examples
python example_usage.py --download-dir ../../data/raw/clinvar_vcv_releases --list-files
```

## Features

- **Systematic Downloads**: Chronological processing from 2023-01 to present
- **Dual Format Support**: Both old (ClinVarVariationRelease) and new (ClinVarVCVRelease) formats
- **Resume Capability**: Skip already downloaded files
- **Integrity Verification**: MD5 checksum validation
- **Progress Tracking**: Real-time download progress and logging
- **Error Handling**: Robust retry logic with exponential backoff
- **Status Reporting**: Comprehensive download status and validation reports

## Data Coverage

### Old Format (ClinVarVariationRelease)
- **Timeline**: January 2023 to June 2025
- **Files**: 30 monthly releases
- **Size**: ~95 GB total

### New Format (ClinVarVCVRelease)
- **Timeline**: February 2024 to June 2025
- **Files**: 17 monthly releases  
- **Size**: ~65 GB total

## Requirements

```bash
pip install requests
```

## Configuration

The toolkit uses `config/clinvar_download_config.py` for:
- FTP URLs and file patterns
- Date ranges for each format
- Expected file sizes
- Download settings
- Validation parameters

## Usage Examples

### Basic Download
```bash
python scripts/download_clinvar_vcv_releases.py
```

### Custom Directory
```bash
python scripts/download_clinvar_vcv_releases.py --download-dir /path/to/data
```

### Force Re-download
```bash
python scripts/download_clinvar_vcv_releases.py --no-skip-existing
```

### Generate Report
```bash
python scripts/clinvar_download_utils.py report --output download_report.txt
```

## Integration with Project

This toolkit is designed to integrate with the broader ClinVar analysis pipeline:

1. **Data Acquisition**: Download historical ClinVar releases
2. **Data Processing**: Parse XML files for variant information
3. **Analysis**: Track variant reclassifications over time
4. **Visualization**: Generate insights and plots

## Support

For issues with the toolkit, check the logs in `../../logs/` directory. For ClinVar data questions, contact clinvar@ncbi.nlm.nih.gov.

## License

This toolkit is provided for research and educational purposes. Please respect NCBI's usage policies and terms of service.
