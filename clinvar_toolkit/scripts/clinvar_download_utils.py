#!/usr/bin/env python3
"""
ClinVar Download Utilities

Companion utilities for managing ClinVar VCV release downloads:
- List available files on FTP server
- Check download status
- Validate downloaded files
- Generate download reports
"""

import os
import sys
import hashlib
import requests
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import argparse
import json


def get_available_files_from_ftp(base_url: str, pattern: str) -> List[Dict]:
    """
    Get list of available files from FTP directory.
    Note: Use ftplib. This is initially for testing
    """
    try:
        files = []

        if "VCV_xml_old_format" in base_url:
            for year in range(2023, 2026):
                start_month = 1 if year > 2023 else 1
                end_month = 12 if year < 2025 else 6

                for month in range(start_month, end_month + 1):
                    filename = f"ClinVarVariationRelease_{year}-{month:02d}.xml.gz"
                    files.append({
                        'filename': filename,
                        'url': base_url + filename,
                        'year': year,
                        'month': month,
                        'format': 'old'
                    })
        else:
            for year in range(2024, 2026):
                start_month = 2 if year == 2024 else 1
                end_month = 12 if year < 2025 else 6

                for month in range(start_month, end_month + 1):
                    filename = f"ClinVarVCVRelease_{year}-{month:02d}.xml.gz"
                    files.append({
                        'filename': filename,
                        'url': base_url + filename,
                        'year': year,
                        'month': month,
                        'format': 'new'
                    })

        return files
    except Exception as e:
        print(f"Error getting file list: {e}")
        return []


def check_download_status(download_dir: str) -> Dict:
    """Check status of downloaded files."""
    download_path = Path(download_dir)

    status = {
        'old_format': {'downloaded': [], 'missing': [], 'total_size': 0},
        'new_format': {'downloaded': [], 'missing': [], 'total_size': 0},
        'summary': {}
    }

    old_format_dir = download_path / "old_format"
    if old_format_dir.exists():
        for year in range(2023, 2026):
            start_month = 1 if year > 2023 else 1
            end_month = 12 if year < 2025 else 6

            for month in range(start_month, end_month + 1):
                filename = f"ClinVarVariationRelease_{year}-{month:02d}.xml.gz"
                file_path = old_format_dir / filename

                if file_path.exists():
                    file_size = file_path.stat().st_size
                    status['old_format']['downloaded'].append({
                        'filename': filename,
                        'size': file_size,
                        'size_mb': file_size / (1024 * 1024),
                        'path': str(file_path)
                    })
                    status['old_format']['total_size'] += file_size
                else:
                    status['old_format']['missing'].append(filename)

    new_format_dir = download_path / "new_format"
    if new_format_dir.exists():
        for year in range(2024, 2026):
            start_month = 2 if year == 2024 else 1
            end_month = 12 if year < 2025 else 6

            for month in range(start_month, end_month + 1):
                filename = f"ClinVarVCVRelease_{year}-{month:02d}.xml.gz"
                file_path = new_format_dir / filename

                if file_path.exists():
                    file_size = file_path.stat().st_size
                    status['new_format']['downloaded'].append({
                        'filename': filename,
                        'size': file_size,
                        'size_mb': file_size / (1024 * 1024),
                        'path': str(file_path)
                    })
                    status['new_format']['total_size'] += file_size
                else:
                    status['new_format']['missing'].append(filename)

    total_old = len(status['old_format']['downloaded']) + len(status['old_format']['missing'])
    total_new = len(status['new_format']['downloaded']) + len(status['new_format']['missing'])

    status['summary'] = {
        'old_format_downloaded': len(status['old_format']['downloaded']),
        'old_format_missing': len(status['old_format']['missing']),
        'old_format_total': total_old,
        'new_format_downloaded': len(status['new_format']['downloaded']),
        'new_format_missing': len(status['new_format']['missing']),
        'new_format_total': total_new,
        'total_downloaded': len(status['old_format']['downloaded']) + len(status['new_format']['downloaded']),
        'total_missing': len(status['old_format']['missing']) + len(status['new_format']['missing']),
        'total_size_gb': (status['old_format']['total_size'] + status['new_format']['total_size']) / (1024 * 1024 * 1024)
    }

    return status


def validate_downloaded_files(download_dir: str, check_md5: bool = False) -> Dict:
    """Validate downloaded files for corruption."""
    download_path = Path(download_dir)
    validation_results = {
        'valid': [],
        'invalid': [],
        'errors': []
    }

    for subdir in ['old_format', 'new_format']:
        subdir_path = download_path / subdir
        if not subdir_path.exists():
            continue

        for file_path in subdir_path.glob("*.xml.gz"):
            try:
                if file_path.stat().st_size == 0:
                    validation_results['invalid'].append({
                        'filename': file_path.name,
                        'error': 'File is empty'
                    })
                    continue

                with open(file_path, 'rb') as f:
                    header = f.read(3)
                    if header[:2] != b'\x1f\x8b':
                        validation_results['invalid'].append({
                            'filename': file_path.name,
                            'error': 'Not a valid gzip file'
                        })
                        continue

                if check_md5:
                    md5_file = file_path.with_suffix(file_path.suffix + '.md5')
                    if md5_file.exists():
                        with open(md5_file, 'r') as f:
                            expected_md5 = f.read().strip().split()[0]

                        actual_md5 = calculate_md5(str(file_path))

                        if actual_md5 != expected_md5:
                            validation_results['invalid'].append({
                                'filename': file_path.name,
                                'error': f'MD5 mismatch: expected {expected_md5}, got {actual_md5}'
                            })
                            continue

                validation_results['valid'].append(file_path.name)

            except Exception as e:
                validation_results['errors'].append({
                    'filename': file_path.name,
                    'error': str(e)
                })

    return validation_results


def calculate_md5(file_path: str) -> str:
    """Calculate MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def generate_download_report(download_dir: str, output_file: str = None) -> str:
    """Generate a comprehensive download report."""
    status = check_download_status(download_dir)
    validation = validate_downloaded_files(download_dir)

    report_lines = []
    report_lines.append("ClinVar VCV Release Download Report")
    report_lines.append("=" * 50)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Download Directory: {download_dir}")
    report_lines.append("")

    report_lines.append("SUMMARY")
    report_lines.append("-" * 20)
    summary = status['summary']
    report_lines.append(f"Total files downloaded: {summary['total_downloaded']}")
    report_lines.append(f"Total files missing: {summary['total_missing']}")
    report_lines.append(f"Total download size: {summary['total_size_gb']:.2f} GB")
    report_lines.append("")

    report_lines.append("OLD FORMAT FILES (ClinVarVariationRelease)")
    report_lines.append("-" * 45)
    report_lines.append(f"Downloaded: {summary['old_format_downloaded']}/{summary['old_format_total']}")
    if status['old_format']['missing']:
        report_lines.append("Missing files:")
        for filename in status['old_format']['missing']:
            report_lines.append(f"  - {filename}")
    report_lines.append("")

    report_lines.append("NEW FORMAT FILES (ClinVarVCVRelease)")
    report_lines.append("-" * 40)
    report_lines.append(f"Downloaded: {summary['new_format_downloaded']}/{summary['new_format_total']}")
    if status['new_format']['missing']:
        report_lines.append("Missing files:")
        for filename in status['new_format']['missing']:
            report_lines.append(f"  - {filename}")
    report_lines.append("")

    report_lines.append("FILE VALIDATION")
    report_lines.append("-" * 20)
    report_lines.append(f"Valid files: {len(validation['valid'])}")
    report_lines.append(f"Invalid files: {len(validation['invalid'])}")
    report_lines.append(f"Validation errors: {len(validation['errors'])}")

    if validation['invalid']:
        report_lines.append("\nInvalid files:")
        for item in validation['invalid']:
            report_lines.append(f"  - {item['filename']}: {item['error']}")

    if validation['errors']:
        report_lines.append("\nValidation errors:")
        for item in validation['errors']:
            report_lines.append(f"  - {item['filename']}: {item['error']}")

    report_text = "\n".join(report_lines)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"Report saved to: {output_file}")

    return report_text


def main():
    """Command line interface for utilities."""
    parser = argparse.ArgumentParser(
        description="ClinVar download utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    status_parser = subparsers.add_parser('status', help='Check download status')
    status_parser.add_argument('--download-dir', default='../../data/raw/clinvar_vcv_releases',
                              help='Download directory to check')

    validate_parser = subparsers.add_parser('validate', help='Validate downloaded files')
    validate_parser.add_argument('--download-dir', default='../../data/raw/clinvar_vcv_releases',
                                help='Download directory to validate')
    validate_parser.add_argument('--check-md5', action='store_true',
                                help='Also verify MD5 checksums')

    report_parser = subparsers.add_parser('report', help='Generate download report')
    report_parser.add_argument('--download-dir', default='../../data/raw/clinvar_vcv_releases',
                              help='Download directory to report on')
    report_parser.add_argument('--output', help='Output file for report')

    args = parser.parse_args()

    if args.command == 'status':
        status = check_download_status(args.download_dir)
        print(json.dumps(status, indent=2))

    elif args.command == 'validate':
        validation = validate_downloaded_files(args.download_dir, args.check_md5)
        print(json.dumps(validation, indent=2))

    elif args.command == 'report':
        report = generate_download_report(args.download_dir, args.output)
        if not args.output:
            print(report)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
