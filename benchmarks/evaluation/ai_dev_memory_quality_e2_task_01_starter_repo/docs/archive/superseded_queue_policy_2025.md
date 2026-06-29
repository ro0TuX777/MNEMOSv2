# Superseded Queue Policy - 2025 Archive

Status: superseded

This document is retained only as historical context. It is not current
implementation authority.

Old behavior from 2025:

- prepare the queue for future cloud sync;
- include `syncEnabled` in migrated state;
- sort by severity only;
- treat `deferred` as equivalent to `approved` during migration;
- keep `accepted` as an active status.

These rules were replaced by ADR 0007.

