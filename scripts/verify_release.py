"""Verify the public repository manifest and basic release hygiene."""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
INVENTORY = REPOSITORY / "release" / "FILE_INVENTORY.csv"
FORBIDDEN_SUFFIXES = {".sqlite", ".db", ".joblib", ".pkl", ".zip", ".pdf"}
FORBIDDEN_PARTS = {"__pycache__", ".ipynb_checkpoints"}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    if not INVENTORY.is_file():
        raise FileNotFoundError("Run scripts/generate_manifest.py before verification.")
    with INVENTORY.open(newline="", encoding="utf-8") as handle:
        expected = {row["path"]: row for row in csv.DictReader(handle)}

    errors: list[str] = []
    for relative, row in expected.items():
        path = REPOSITORY / relative
        if not path.is_file():
            errors.append(f"Missing: {relative}")
        elif digest(path) != row["sha256"]:
            errors.append(f"Checksum mismatch: {relative}")

    for path in REPOSITORY.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(REPOSITORY).as_posix()
        if relative in {"release/FILE_INVENTORY.csv", "release/SHA256SUMS.csv"}:
            continue
        if relative not in expected:
            errors.append(f"Not listed in manifest: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"Excluded binary/archive type: {relative}")
        if any(part in FORBIDDEN_PARTS for part in path.parts):
            errors.append(f"Excluded cache path: {relative}")
        if path.stat().st_size >= 100 * 1024 * 1024:
            errors.append(f"Exceeds GitHub 100 MB limit: {relative}")

    if errors:
        print("Release verification failed:", *errors, sep="\n- ")
        sys.exit(1)
    print(f"Release verification passed ({len(expected)} manifest entries).")


if __name__ == "__main__":
    main()
