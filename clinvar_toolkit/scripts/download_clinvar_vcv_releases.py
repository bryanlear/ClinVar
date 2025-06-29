#!/usr/bin/env python3
"""
ClinVar VCV Release Downloader

This script systematically downloads ClinVar VCV release files from NCBI FTP:
1. Downloads old format files (ClinVarVariationRelease_YYYY-MM.xml.gz) from 2023-01 onwards
2. Downloads new format files (ClinVarVCVRelease_YYYY-MM.xml.gz) from 2024-02 onwards

The script handles:
- Chronological downloading from earliest to latest
- MD5 checksum verification
- Resume capability (skips already downloaded files)
- Progress tracking and logging
- Error handling and retry logic
"""

import os
import sys
import hashlib
import logging
import requests
import time
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urljoin
import argparse


OLD_FORMAT_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/xml/VCV_xml_old_format/"
NEW_FORMAT_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/xml/"

OLD_FORMAT_PATTERN = "ClinVarVariationRelease_{year}-{month:02d}.xml.gz"
NEW_FORMAT_PATTERN = "ClinVarVCVRelease_{year}-{month:02d}.xml.gz"

OLD_FORMAT_START = (2023, 1)
OLD_FORMAT_END = (2025, 6)
NEW_FORMAT_START = (2024, 2)
NEW_FORMAT_END = (2025, 6)


def setup_logging(log_file: str = "../../logs/clinvar_download.log") -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def calculate_md5(file_path: str) -> str:
    """Calculate MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def verify_md5_checksum(file_path: str, md5_url: str, logger: logging.Logger) -> bool:
    """Verify file integrity using MD5 checksum."""
    try:
        response = requests.get(md5_url, timeout=30)
        response.raise_for_status()
        expected_md5 = response.text.strip().split()[0]

        actual_md5 = calculate_md5(file_path)

        if actual_md5 == expected_md5:
            logger.info(f"✓ MD5 checksum verified for {os.path.basename(file_path)}")
            return True
        else:
            logger.error(f"✗ MD5 checksum mismatch for {os.path.basename(file_path)}")
            logger.error(f"  Expected: {expected_md5}")
            logger.error(f"  Actual:   {actual_md5}")
            return False
    except Exception as e:
        logger.warning(f"Could not verify MD5 for {os.path.basename(file_path)}: {e}")
        return True


def download_file(url: str, local_path: str, logger: logging.Logger,
                 max_retries: int = 3) -> bool:
    """Download a file with retry logic and progress tracking."""
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading {os.path.basename(local_path)} (attempt {attempt + 1}/{max_retries})")

            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0

            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)

                        if downloaded_size % (100 * 1024 * 1024) == 0:
                            if total_size > 0:
                                progress = (downloaded_size / total_size) * 100
                                logger.info(f"  Progress: {progress:.1f}% ({downloaded_size // (1024*1024)} MB)")

            logger.info(f"✓ Downloaded {os.path.basename(local_path)} ({downloaded_size // (1024*1024)} MB)")
            return True

        except Exception as e:
            logger.error(f"✗ Download failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to download {url} after {max_retries} attempts")
                return False

    return False


def generate_date_range(start_year: int, start_month: int,
                       end_year: int, end_month: int) -> List[Tuple[int, int]]:
    """Generate a list of (year, month) tuples for the given range."""
    dates = []
    current_year, current_month = start_year, start_month

    while (current_year, current_month) <= (end_year, end_month):
        dates.append((current_year, current_month))

        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1

    return dates


def download_clinvar_files(download_dir: str, skip_existing: bool = True,
                          verify_checksums: bool = True) -> None:
    """Main function to download ClinVar VCV release files."""
    logger = setup_logging()

    download_path = Path(download_dir)
    download_path.mkdir(parents=True, exist_ok=True)

    old_format_dir = download_path / "old_format"
    new_format_dir = download_path / "new_format"
    old_format_dir.mkdir(exist_ok=True)
    new_format_dir.mkdir(exist_ok=True)

    logger.info("Starting ClinVar VCV release download")
    logger.info(f"Download directory: {download_path.absolute()}")

    total_files = 0
    downloaded_files = 0
    skipped_files = 0
    failed_files = 0

    logger.info("=" * 60)
    logger.info("DOWNLOADING OLD FORMAT FILES (ClinVarVariationRelease)")
    logger.info("=" * 60)

    old_dates = generate_date_range(*OLD_FORMAT_START, *OLD_FORMAT_END)

    for year, month in old_dates:
        filename = OLD_FORMAT_PATTERN.format(year=year, month=month)
        md5_filename = filename + ".md5"

        file_url = urljoin(OLD_FORMAT_BASE_URL, filename)
        md5_url = urljoin(OLD_FORMAT_BASE_URL, md5_filename)

        local_file_path = old_format_dir / filename

        total_files += 1

        if skip_existing and local_file_path.exists():
            logger.info(f"⏭ Skipping {filename} (already exists)")
            skipped_files += 1
            continue

        if download_file(file_url, str(local_file_path), logger):
            if verify_checksums:
                if not verify_md5_checksum(str(local_file_path), md5_url, logger):
                    logger.warning(f"Checksum verification failed for {filename}")
            downloaded_files += 1
        else:
            failed_files += 1

    logger.info("=" * 60)
    logger.info("DOWNLOADING NEW FORMAT FILES (ClinVarVCVRelease)")
    logger.info("=" * 60)

    new_dates = generate_date_range(*NEW_FORMAT_START, *NEW_FORMAT_END)

    for year, month in new_dates:
        filename = NEW_FORMAT_PATTERN.format(year=year, month=month)
        md5_filename = filename + ".md5"

        file_url = urljoin(NEW_FORMAT_BASE_URL, filename)
        md5_url = urljoin(NEW_FORMAT_BASE_URL, md5_filename)

        local_file_path = new_format_dir / filename

        total_files += 1

        if skip_existing and local_file_path.exists():
            logger.info(f"⏭ Skipping {filename} (already exists)")
            skipped_files += 1
            continue

        if download_file(file_url, str(local_file_path), logger):
            if verify_checksums:
                if not verify_md5_checksum(str(local_file_path), md5_url, logger):
                    logger.warning(f"Checksum verification failed for {filename}")
            downloaded_files += 1
        else:
            failed_files += 1

    logger.info("=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total files processed: {total_files}")
    logger.info(f"Successfully downloaded: {downloaded_files}")
    logger.info(f"Skipped (already exist): {skipped_files}")
    logger.info(f"Failed downloads: {failed_files}")

    if failed_files > 0:
        logger.warning(f"⚠ {failed_files} files failed to download. Check the log for details.")
    else:
        logger.info("✓ All files processed successfully!")


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Download ClinVar VCV release files systematically",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_clinvar_vcv_releases.py
  python download_clinvar_vcv_releases.py --download-dir ./clinvar_data
  python download_clinvar_vcv_releases.py --no-skip-existing --no-verify-checksums
        """
    )

    parser.add_argument(
        "--download-dir",
        default="../../data/raw/clinvar_vcv_releases",
        help="Directory to download files to (default: ../../data/raw/clinvar_vcv_releases)"
    )

    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-download files even if they already exist"
    )

    parser.add_argument(
        "--no-verify-checksums",
        action="store_true",
        help="Skip MD5 checksum verification"
    )

    args = parser.parse_args()

    download_clinvar_files(
        download_dir=args.download_dir,
        skip_existing=not args.no_skip_existing,
        verify_checksums=not args.no_verify_checksums
    )


if __name__ == "__main__":
    main()
