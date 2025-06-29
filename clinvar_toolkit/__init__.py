"""
ClinVar Data Toolkit

A comprehensive Python toolkit for downloading, processing, and analyzing 
ClinVar VCV release files from NCBI's FTP server.

This package provides:
- Systematic downloading of ClinVar monthly releases
- Support for both old and new XML formats
- Data validation and integrity checking
- Progress tracking and error handling
- Analysis utilities and examples

Usage:
    from clinvar_toolkit.scripts import download_clinvar_vcv_releases
    from clinvar_toolkit.config import clinvar_download_config
"""

__version__ = "1.0.0"
__author__ = "ClinVar Analysis Project"
__description__ = "Toolkit for ClinVar VCV release data acquisition and analysis"

__all__ = [
    "scripts",
    "config", 
    "examples"
]
