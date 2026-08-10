"""Verify release weights, metadata, configuration, and optional local inputs."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import nibabel as nib
import torch
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root_default)
    args = parser.parse_args()
    root = args.root.resolve()

    expected = {}
    for line in (root / "models" / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        checksum, relative = line.split(maxsplit=1)
        expected[relative.strip()] = checksum
    for relative, checksum in expected.items():
        path = root / "models" / relative
        if not path.is_file() or sha256(path) != checksum:
            raise RuntimeError(f"Checksum mismatch: {relative}")

    for directory, task in (("classification", "classifier"), ("regression", "regressor")):
        checkpoint = torch.load(root / "models" / directory / "model.pth", map_location="cpu", weights_only=True)
        if set(checkpoint) != {
            "format_version", "task", "model_name", "fold", "selection",
            "source_sha256", "model_state_dict",
        }:
            raise RuntimeError(f"Unexpected public checkpoint keys: {directory}: {set(checkpoint)}")
        serialized = repr({key: value for key, value in checkpoint.items() if key != "model_state_dict"})
        if ":\\" in serialized or "Users\\" in serialized:
            raise RuntimeError(f"Local path found in public checkpoint: {directory}")

    config = yaml.safe_load((root / "config" / "inference.yaml").read_text(encoding="utf-8"))
    if "classification" not in config or "regression" not in config:
        raise RuntimeError("Classification or regression configuration is missing")

    for path in sorted((root / "examples" / "input").glob("*.nii.gz")):
        header = nib.load(str(path)).header
        if bytes(header["descrip"]).rstrip(b"\x00") or bytes(header["aux_file"]).rstrip(b"\x00"):
            raise RuntimeError(f"Non-empty identifying NIfTI header field: {path.name}")

    local_inputs = len(list((root / "examples" / "input").glob("*.nii.gz")))
    print(f"Verified {len(expected)} weights and {local_inputs} optional local input images")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Release verification failed: {error}", file=sys.stderr)
        raise
