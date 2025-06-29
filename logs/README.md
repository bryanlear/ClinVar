# Logs Directory

This directory contains log files from various project operations.

## Log Files

### ClinVar Download Logs
- **File**: `clinvar_download.log`
- **Content**: Download progress, errors, and status messages
- **Format**: Timestamped entries with log levels (INFO, WARNING, ERROR)

### Log Rotation
- Logs are appended to existing files
- Consider rotating logs periodically for large operations
- Archive old logs to prevent disk space issues

## Log Levels

- **INFO**: Normal operation messages
- **WARNING**: Non-critical issues that don't stop execution
- **ERROR**: Critical errors that may cause failures

## Usage

### Monitoring Downloads
```bash
tail -f logs/clinvar_download.log
```

### Searching for Errors
```bash
grep "ERROR" logs/clinvar_download.log
```

### Checking Download Progress
```bash
grep "Downloaded" logs/clinvar_download.log | tail -10
```
