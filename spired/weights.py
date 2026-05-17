from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import click

MODEL_FILENAMES = ("SPIRED-Fitness.pth", "SPIRED-Stab.pth")
MODEL_ARCHIVE_URL = "https://zenodo.org/records/10675405/files/model.zip"


def _cache_root() -> Path:
    cache_home = os.environ.get("XDG_CACHE_HOME")
    if cache_home:
        return Path(cache_home).expanduser() / "spired" / "model"
    return Path.home() / ".cache" / "spired" / "model"


def default_model_dir() -> Path:
    env_model_dir = os.environ.get("SPIRED_MODEL_DIR")
    if env_model_dir:
        return Path(env_model_dir).expanduser()

    repo_model_dir = Path(__file__).resolve().parent.parent / "model"
    if all((repo_model_dir / name).exists() for name in MODEL_FILENAMES):
        return repo_model_dir

    return _cache_root()


def ensure_model_dir(model_dir: Path | None = None) -> Path:
    resolved = Path(model_dir).expanduser() if model_dir else default_model_dir()
    resolved.mkdir(parents=True, exist_ok=True)
    if all((resolved / name).exists() for name in MODEL_FILENAMES):
        return resolved

    _download_model_archive(resolved)
    if not all((resolved / name).exists() for name in MODEL_FILENAMES):
        raise click.ClickException(
            f"Model weights were not found in {resolved}. Downloaded archive did not expose the expected files."
        )
    return resolved


def model_path(filename: str, model_dir: Path | None = None) -> Path:
    resolved = ensure_model_dir(model_dir)
    path = resolved / filename
    if not path.exists():
        raise click.ClickException(f"Missing model weights: {path}")
    return path


def _download_model_archive(dest_dir: Path) -> None:
    click.echo(f"Downloading SPIRED model weights to {dest_dir} ...")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with urllib.request.urlopen(MODEL_ARCHIVE_URL) as response, open(tmp_path, "wb") as fh:
            shutil.copyfileobj(response, fh)
        _safe_extract_zip(tmp_path, dest_dir)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _safe_extract_zip(archive_path: Path, dest_dir: Path) -> None:
    dest_dir = dest_dir.resolve()
    with zipfile.ZipFile(archive_path) as zf:
        for member in zf.infolist():
            member_path = (dest_dir / member.filename).resolve()
            if dest_dir not in member_path.parents and member_path != dest_dir:
                raise click.ClickException("Refusing to extract unsafe model archive contents.")
        zf.extractall(dest_dir)

    nested_dir = dest_dir / "model"
    for filename in MODEL_FILENAMES:
        root_file = dest_dir / filename
        if root_file.exists():
            continue
        nested_matches = list(dest_dir.rglob(filename))
        if nested_matches:
            nested_file = nested_matches[0]
            nested_file.replace(root_file)
    if nested_dir.exists():
        try:
            nested_dir.rmdir()
        except OSError:
            pass

