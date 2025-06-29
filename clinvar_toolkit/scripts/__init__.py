"""
ClinVar Toolkit Scripts

Main scripts for downloading and managing ClinVar VCV release data.

Available scripts:
- download_clinvar_vcv_releases.py: Primary download script
- clinvar_download_utils.py: Utility functions for status checking and validation
"""

from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent.parent

__all__ = [
    "download_clinvar_vcv_releases",
    "clinvar_download_utils"
]
