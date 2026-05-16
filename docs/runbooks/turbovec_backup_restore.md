# Portable Memory Appliance: Backup & Restore Runbook

## Purpose
This runbook details the procedures for backing up and restoring the `TurbovecTier` index footprint. The Portable Memory Appliance relies on three physical files (`index.tvim`, `metadata.sqlite`, and `manifest.json`) instead of a distributed vector database, meaning traditional file backups must be transactionally safe and hash-verified.

## Protected State
The backup encapsulates:
- `index.tvim`: The compressed, SIMD-oriented dense vector payload.
- `metadata.sqlite`: The lexical FTS5 engine and governance metadata mapping.
- `manifest.json`: Configuration bounds linking embeddings to SIMD precision.

## Commands

### 1. Execute Backup
Run the backup CLI against the target runtime directory:

```powershell
python -m mnemos.tools.turbovec_backup `
  --profile-dir runtime/turbovec_storage `
  --out backups/turbovec_backup_YYYYMMDD_HHMMSS.zip
```
*(Optionally use `--force` to overwrite an existing zip file.)*

### 2. Execute Restore
Run the restore CLI to atomically rebuild the directory:

```powershell
python -m mnemos.tools.turbovec_restore `
  --archive backups/turbovec_backup_YYYYMMDD_HHMMSS.zip `
  --target-dir runtime/turbovec_storage `
  --replace
```
*(Without `--replace`, it will fail-closed if `runtime/turbovec_storage` already exists.)*

## Validation & Receipts
Both commands automatically emit a JSON receipt to `runtime/receipts/`. Receipts guarantee that operations only succeeded after proving:
- SHA-256 hash matching against the internal payload.
- `PRAGMA integrity_check` returning `ok` for SQLite metadata.
- Pre-load schema validation by temporarily instantiating a `TurbovecTier` over the unzipped payload.

## Failure Modes & Rollback Behavior
- **Tampered Hashes / Missing Files**: The `turbovec_restore` process extracts to a temporary folder and validates the entire payload. If it detects a missing or modified file, it aborts instantly without touching your active `target-dir`.
- **Atomic Rollback**: If `--replace` is used, the system safely moves the existing profile directory into `.rollback_<timestamp>` before moving the newly validated profile into position.

## Operator Checklist
- [ ] Ensure the MNEMOS process is suspended or blocking ingestions to prevent partial writes to `metadata.sqlite` during the backup read operation.
- [ ] Confirm receipt generated properly after `restore` pointing to `status: success`.

## Known Limitations
- The backup CLI does not currently issue SQLite write-locks or snapshot requests. For highly concurrent ingestion patterns, stop the core router before pulling the backup to avoid SQLite WAL race conditions.
