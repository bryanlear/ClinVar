#!/usr/bin/env python3
"""
Configuration file for ClinVar VCV Release downloads.

This file contains all the configuration parameters for downloading
ClinVar VCV release files. Modify these settings as needed.
"""

OLD_FORMAT_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/xml/VCV_xml_old_format/"
NEW_FORMAT_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/xml/"

OLD_FORMAT_PATTERN = "ClinVarVariationRelease_{year}-{month:02d}.xml.gz"
NEW_FORMAT_PATTERN = "ClinVarVCVRelease_{year}-{month:02d}.xml.gz"

OLD_FORMAT_DATE_RANGE = {
    'start': (2023, 1),
    'end': (2025, 6)
}

NEW_FORMAT_DATE_RANGE = {
    'start': (2024, 2),
    'end': (2025, 6)
}

DOWNLOAD_SETTINGS = {
    'max_retries': 3,
    'timeout_seconds': 60,
    'chunk_size': 8192,
    'progress_report_interval': 100 * 1024 * 1024,
    'verify_checksums': True,
    'skip_existing': True
}

DIRECTORY_STRUCTURE = {
    'old_format_subdir': 'old_format',
    'new_format_subdir': 'new_format',
    'log_file': 'clinvar_download.log'
}

EXPECTED_FILE_SIZES = {
    'old_format': {
        '2023-01': 2185,
        '2023-02': 2779,
        '2023-03': 2798,
        '2023-04': 2860,
        '2023-05': 2889,
        '2023-06': 2888,
        '2023-07': 2886,
        '2023-08': 2943,
        '2023-09': 2938,
        '2023-10': 2941,
        '2023-11': 3018,
        '2023-12': 3046,
        '2024-01': 3141,
        '2024-02': 3151,
        '2024-03': 3587,
        '2024-04': 3630,
        '2024-05': 3745,
        '2024-06': 3730,
        '2024-07': 3735,
        '2024-08': 3750,
        '2024-09': 3799,
        '2024-10': 3866,
        '2024-11': 3893,
        '2024-12': 3905,
        '2025-01': 3921,
        '2025-02': 4291,
        '2025-03': 4424,
        '2025-04': 4425,
        '2025-05': 4497,
        '2025-06': 4538,
    },

    'new_format': {
        '2024-02': 3180,
        '2024-03': 3620,
        '2024-04': 3663,
        '2024-05': 3782,
        '2024-06': 3765,
        '2024-07': 3770,
        '2024-08': 3785,
        '2024-09': 3835,
        '2024-10': 3905,
        '2024-11': 3932,
        '2024-12': 3946,
        '2025-01': 3962,
        '2025-02': 4340,
        '2025-03': 4476,
        '2025-04': 4478,
        '2025-05': 4551,
        '2025-06': 4594,
    }
}

VALIDATION_SETTINGS = {
    'check_gzip_header': True,
    'check_file_size': True,
    'size_tolerance_percent': 10,
    'check_xml_structure': False,
}

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S'
}

NETWORK_SETTINGS = {
    'user_agent': 'ClinVar-Downloader/1.0',
    'connection_timeout': 30,
    'read_timeout': 300,
    'max_concurrent_downloads': 1,
}

def get_total_expected_size():
    """Calculate total expected download size in GB."""
    total_mb = 0

    for size_mb in EXPECTED_FILE_SIZES['old_format'].values():
        total_mb += size_mb

    for size_mb in EXPECTED_FILE_SIZES['new_format'].values():
        total_mb += size_mb

    return total_mb / 1024


def get_file_list():
    """Generate complete list of files to download."""
    files = []

    start_year, start_month = OLD_FORMAT_DATE_RANGE['start']
    end_year, end_month = OLD_FORMAT_DATE_RANGE['end']

    current_year, current_month = start_year, start_month
    while (current_year, current_month) <= (end_year, end_month):
        filename = OLD_FORMAT_PATTERN.format(year=current_year, month=current_month)
        files.append({
            'filename': filename,
            'format': 'old',
            'year': current_year,
            'month': current_month,
            'url': OLD_FORMAT_BASE_URL + filename,
            'expected_size_mb': EXPECTED_FILE_SIZES['old_format'].get(f'{current_year}-{current_month:02d}', 0)
        })

        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1

    start_year, start_month = NEW_FORMAT_DATE_RANGE['start']
    end_year, end_month = NEW_FORMAT_DATE_RANGE['end']

    current_year, current_month = start_year, start_month
    while (current_year, current_month) <= (end_year, end_month):
        filename = NEW_FORMAT_PATTERN.format(year=current_year, month=current_month)
        files.append({
            'filename': filename,
            'format': 'new',
            'year': current_year,
            'month': current_month,
            'url': NEW_FORMAT_BASE_URL + filename,
            'expected_size_mb': EXPECTED_FILE_SIZES['new_format'].get(f'{current_year}-{current_month:02d}', 0)
        })

        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1

    return files


if __name__ == "__main__":
    print("ClinVar VCV Release Download Configuration")
    print("=" * 50)
    print(f"Total expected download size: {get_total_expected_size():.1f} GB")
    print(f"Total files to download: {len(get_file_list())}")
    print(f"Old format files: {len([f for f in get_file_list() if f['format'] == 'old'])}")
    print(f"New format files: {len([f for f in get_file_list() if f['format'] == 'new'])}")
    print()

    print("Date Ranges:")
    print(f"Old format: {OLD_FORMAT_DATE_RANGE['start'][0]}-{OLD_FORMAT_DATE_RANGE['start'][1]:02d} to {OLD_FORMAT_DATE_RANGE['end'][0]}-{OLD_FORMAT_DATE_RANGE['end'][1]:02d}")
    print(f"New format: {NEW_FORMAT_DATE_RANGE['start'][0]}-{NEW_FORMAT_DATE_RANGE['start'][1]:02d} to {NEW_FORMAT_DATE_RANGE['end'][0]}-{NEW_FORMAT_DATE_RANGE['end'][1]:02d}")
