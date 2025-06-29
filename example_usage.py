#!/usr/bin/env python3
"""
Example usage of downloaded ClinVar VCV release files.

This script demonstrates how to:
1. List and organize downloaded files chronologically
2. Parse XML content from the files
3. Extract basic statistics
4. Compare old vs new format files
"""

import gzip
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
from datetime import datetime
import json


def list_downloaded_files(download_dir: str) -> dict:
    """List all downloaded files organized by format and date."""
    download_path = Path(download_dir)

    files = {
        'old_format': [],
        'new_format': []
    }

    old_format_dir = download_path / "old_format"
    if old_format_dir.exists():
        for file_path in sorted(old_format_dir.glob("ClinVarVariationRelease_*.xml.gz")):
            filename = file_path.name
            date_part = filename.replace("ClinVarVariationRelease_", "").replace(".xml.gz", "")
            year, month = date_part.split("-")

            files['old_format'].append({
                'filename': filename,
                'path': str(file_path),
                'year': int(year),
                'month': int(month),
                'date': f"{year}-{month}",
                'size_mb': file_path.stat().st_size / (1024 * 1024)
            })

    new_format_dir = download_path / "new_format"
    if new_format_dir.exists():
        for file_path in sorted(new_format_dir.glob("ClinVarVCVRelease_*.xml.gz")):
            filename = file_path.name
            date_part = filename.replace("ClinVarVCVRelease_", "").replace(".xml.gz", "")
            year, month = date_part.split("-")

            files['new_format'].append({
                'filename': filename,
                'path': str(file_path),
                'year': int(year),
                'month': int(month),
                'date': f"{year}-{month}",
                'size_mb': file_path.stat().st_size / (1024 * 1024)
            })

    return files


def parse_xml_sample(file_path: str, max_records: int = 100) -> dict:
    """Parse a sample of records from a ClinVar XML file."""
    stats = {
        'total_records': 0,
        'sample_records': [],
        'clinical_significance_counts': {},
        'review_status_counts': {},
        'file_format': 'unknown'
    }

    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            context = ET.iterparse(f, events=('start', 'end'))
            context = iter(context)
            event, root = next(context)

            if root.tag == 'ClinVarVariationRelease':
                stats['file_format'] = 'old'
                record_tag = 'VariationArchive'
            elif root.tag == 'ClinVarVCVRelease':
                stats['file_format'] = 'new'
                record_tag = 'VariationArchive'
            else:
                stats['file_format'] = 'unknown'
                return stats

            for event, elem in context:
                if event == 'end' and elem.tag == record_tag:
                    stats['total_records'] += 1

                    if len(stats['sample_records']) < max_records:
                        record_data = extract_record_data(elem, stats['file_format'])
                        stats['sample_records'].append(record_data)

                        if record_data.get('clinical_significance'):
                            sig = record_data['clinical_significance']
                            stats['clinical_significance_counts'][sig] = stats['clinical_significance_counts'].get(sig, 0) + 1

                        if record_data.get('review_status'):
                            status = record_data['review_status']
                            stats['review_status_counts'][status] = stats['review_status_counts'].get(status, 0) + 1

                    elem.clear()
                    root.clear()

                    if stats['total_records'] % 10000 == 0:
                        print(f"  Processed {stats['total_records']} records...")

                    if stats['total_records'] >= 50000:
                        break

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        stats['error'] = str(e)

    return stats


def extract_record_data(elem, file_format: str) -> dict:
    """Extract key data from a variation record."""
    data = {}

    try:
        data['accession'] = elem.get('Accession', '')
        data['version'] = elem.get('Version', '')
        data['record_type'] = elem.get('RecordType', '')

        if file_format == 'old':
            interp_elem = elem.find('.//ClinicalSignificance')
            if interp_elem is not None:
                desc_elem = interp_elem.find('Description')
                if desc_elem is not None:
                    data['clinical_significance'] = desc_elem.text

                review_elem = interp_elem.find('ReviewStatus')
                if review_elem is not None:
                    data['review_status'] = review_elem.text

        elif file_format == 'new':
            interp_elem = elem.find('.//Interpretation')
            if interp_elem is not None:
                desc_elem = interp_elem.find('Description')
                if desc_elem is not None:
                    data['clinical_significance'] = desc_elem.text

                review_elem = elem.find('.//ReviewStatus')
                if review_elem is not None:
                    data['review_status'] = review_elem.text

        variant_elem = elem.find('.//SimpleAllele') or elem.find('.//Haplotype') or elem.find('.//Genotype')
        if variant_elem is not None:
            data['variant_type'] = variant_elem.tag

            gene_elem = variant_elem.find('.//Gene')
            if gene_elem is not None:
                data['gene_symbol'] = gene_elem.get('Symbol', '')

    except Exception as e:
        data['parse_error'] = str(e)

    return data


def analyze_file(file_info: dict) -> dict:
    """Analyze a single ClinVar file."""
    print(f"\nAnalyzing {file_info['filename']}...")
    print(f"  Size: {file_info['size_mb']:.1f} MB")

    stats = parse_xml_sample(file_info['path'], max_records=1000)

    print(f"  Format: {stats['file_format']}")
    print(f"  Total records processed: {stats['total_records']}")
    print(f"  Sample records: {len(stats['sample_records'])}")

    if stats['clinical_significance_counts']:
        print("  Clinical Significance Distribution (sample):")
        for sig, count in sorted(stats['clinical_significance_counts'].items()):
            print(f"    {sig}: {count}")

    return stats


def compare_formats(old_stats: dict, new_stats: dict) -> dict:
    """Compare statistics between old and new format files."""
    comparison = {
        'old_format_records': old_stats.get('total_records', 0),
        'new_format_records': new_stats.get('total_records', 0),
        'format_differences': []
    }

    old_sigs = set(old_stats.get('clinical_significance_counts', {}).keys())
    new_sigs = set(new_stats.get('clinical_significance_counts', {}).keys())

    if old_sigs != new_sigs:
        comparison['format_differences'].append({
            'type': 'clinical_significance_terms',
            'old_only': list(old_sigs - new_sigs),
            'new_only': list(new_sigs - old_sigs),
            'common': list(old_sigs & new_sigs)
        })

    return comparison


def main():
    """Main function for example usage."""
    parser = argparse.ArgumentParser(
        description="Example usage of downloaded ClinVar VCV files"
    )

    parser.add_argument(
        '--download-dir',
        default='./clinvar_vcv_releases',
        help='Directory containing downloaded files'
    )

    parser.add_argument(
        '--analyze-latest',
        action='store_true',
        help='Analyze the latest files from each format'
    )

    parser.add_argument(
        '--list-files',
        action='store_true',
        help='List all downloaded files'
    )

    parser.add_argument(
        '--compare-formats',
        action='store_true',
        help='Compare old vs new format files'
    )

    args = parser.parse_args()

    files = list_downloaded_files(args.download_dir)

    if args.list_files:
        print("Downloaded ClinVar VCV Files")
        print("=" * 40)

        print(f"\nOld Format Files ({len(files['old_format'])}):")
        for file_info in files['old_format']:
            print(f"  {file_info['date']}: {file_info['filename']} ({file_info['size_mb']:.1f} MB)")

        print(f"\nNew Format Files ({len(files['new_format'])}):")
        for file_info in files['new_format']:
            print(f"  {file_info['date']}: {file_info['filename']} ({file_info['size_mb']:.1f} MB)")

    if args.analyze_latest:
        print("\nAnalyzing Latest Files")
        print("=" * 30)

        if files['old_format']:
            latest_old = files['old_format'][-1]
            old_stats = analyze_file(latest_old)

        if files['new_format']:
            latest_new = files['new_format'][-1]
            new_stats = analyze_file(latest_new)

    if args.compare_formats and files['old_format'] and files['new_format']:
        print("\nComparing Formats")
        print("=" * 20)

        latest_old = files['old_format'][-1]
        latest_new = files['new_format'][-1]

        old_stats = parse_xml_sample(latest_old['path'], max_records=1000)
        new_stats = parse_xml_sample(latest_new['path'], max_records=1000)

        comparison = compare_formats(old_stats, new_stats)
        print(json.dumps(comparison, indent=2))

    if not any([args.list_files, args.analyze_latest, args.compare_formats]):
        print("ClinVar VCV Files Summary")
        print("=" * 30)
        print(f"Old format files: {len(files['old_format'])}")
        print(f"New format files: {len(files['new_format'])}")

        total_size = sum(f['size_mb'] for f in files['old_format'] + files['new_format'])
        print(f"Total size: {total_size / 1024:.1f} GB")

        if files['old_format']:
            print(f"Old format date range: {files['old_format'][0]['date']} to {files['old_format'][-1]['date']}")

        if files['new_format']:
            print(f"New format date range: {files['new_format'][0]['date']} to {files['new_format'][-1]['date']}")

        print("\nUse --help to see analysis options")


if __name__ == "__main__":
    main()
