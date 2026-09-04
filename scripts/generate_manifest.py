"""Create a file inventory and SHA-256 manifest for a repository release."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
RELEASE = REPOSITORY / "release"
EXCLUDED = {"release/FILE_INVENTORY.csv", "release/SHA256SUMS.csv"}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    rows: list[tuple[str, int, str]] = []
    for path in sorted(REPOSITORY.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(REPOSITORY).as_posix()
        if relative in EXCLUDED:
            continue
        rows.append((relative, path.stat().st_size, digest(path)))

    RELEASE.mkdir(exist_ok=True)
    with (RELEASE / "FILE_INVENTORY.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "bytes", "sha256"])
        writer.writerows(rows)
    with (RELEASE / "SHA256SUMS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sha256", "path"])
        writer.writerows((checksum, relative) for relative, _, checksum in rows)
    print(f"Wrote manifests for {len(rows)} files.")


if __name__ == "__main__":
    main()
