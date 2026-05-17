from __future__ import annotations

import re
from pathlib import Path

from Bio import SeqIO


def load_records(fasta_file: Path):
    return list(SeqIO.parse(str(fasta_file), "fasta"))


def slugify_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def unique_record_name(record, index: int, seen: set[str]) -> str:
    base = slugify_name(getattr(record, "id", "") or getattr(record, "description", ""), f"sequence_{index + 1}")
    candidate = base
    suffix = 2
    while candidate in seen:
        candidate = f"{base}_{suffix}"
        suffix += 1
    seen.add(candidate)
    return candidate

