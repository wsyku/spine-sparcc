from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import torch
import yaml
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "models"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import load_model
from preprocessing import SEGMENT_TO_ID, prepare_case


SEGMENT_SUFFIX = {
    "C": "cervical",
    "T": "thoracic",
    "L": "lumbar",
}
CASE_PATTERN = re.compile(
    r"^(?P<patient>.+)_(?P<segment>[CTL])_0000\.nii\.gz$",
    re.IGNORECASE,
)


def resolve_model_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else MODEL_ROOT / path


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def configure_reproducible_inference() -> None:
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def discover_cases(input_dir: Path) -> list[dict]:
    cases = []
    for raw_path in sorted(input_dir.glob("*_0000.nii.gz")):
        match = CASE_PATTERN.match(raw_path.name)
        if match is None:
            raise ValueError(
                f"Unsupported filename {raw_path.name!r}; expected "
                "<patient_id>_<C|T|L>_0000.nii.gz"
            )
        segment_code = match.group("segment").upper()
        sample_id = raw_path.name[: -len("_0000.nii.gz")]
        cases.append(
            {
                "patient_id": match.group("patient"),
                "sample_id": sample_id,
                "segment": SEGMENT_SUFFIX[segment_code],
                "segment_id": SEGMENT_TO_ID[
                    SEGMENT_SUFFIX[segment_code]
                ],
                "raw_path": raw_path,
            }
        )
    if not cases:
        raise FileNotFoundError(
            f"No *_0000.nii.gz cases were found in {input_dir}"
        )
    sample_ids = [case["sample_id"] for case in cases]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample IDs were found")
    return cases


def load_prediction_model(checkpoint_path: Path, task: str, device: torch.device):
    return load_model(checkpoint_path, task, device)


@contextmanager
def prepare_nnunet_model_layout(segmentation: dict):
    """Adapt flat public checkpoints to nnU-Net's fold-directory layout."""
    model_dir = resolve_model_path(segmentation["model_dir"])
    folds = tuple(int(fold) for fold in segmentation["folds"])
    weight_pattern = str(
        segmentation.get("weight_pattern", "fold_{fold}.pth")
    )
    metadata_names = ("dataset.json", "plans.json")

    missing = [
        str(model_dir / name)
        for name in metadata_names
        if not (model_dir / name).is_file()
    ]
    weight_paths = {
        fold: model_dir / weight_pattern.format(fold=fold)
        for fold in folds
    }
    missing.extend(
        str(path) for path in weight_paths.values() if not path.is_file()
    )
    if missing:
        raise FileNotFoundError(
            "Missing nnU-Net model files:\n" + "\n".join(missing)
        )

    with tempfile.TemporaryDirectory(
        prefix=".nnunet_runtime_",
        dir=model_dir,
    ) as runtime_dir_string:
        runtime_dir = Path(runtime_dir_string)
        for name in metadata_names:
            shutil.copy2(model_dir / name, runtime_dir / name)
        for fold, source in weight_paths.items():
            fold_dir = runtime_dir / f"fold_{fold}"
            fold_dir.mkdir()
            destination = fold_dir / "checkpoint_inference.pth"
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        yield runtime_dir


def run_segmentation(
    input_dir: Path,
    output_dir: Path,
    config: dict,
    device: torch.device,
) -> None:
    segmentation = config["segmentation"]
    predictor = nnUNetPredictor(
        tile_step_size=float(segmentation["tile_step_size"]),
        use_gaussian=bool(segmentation["use_gaussian"]),
        use_mirroring=bool(segmentation["use_mirroring"]),
        perform_everything_on_device=(device.type == "cuda"),
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    with prepare_nnunet_model_layout(segmentation) as runtime_model_dir:
        predictor.initialize_from_trained_model_folder(
            str(runtime_model_dir),
            use_folds=tuple(segmentation["folds"]),
            checkpoint_name="checkpoint_inference.pth",
        )
        predictor.predict_from_files(
            str(input_dir),
            str(output_dir),
            save_probabilities=True,
            overwrite=True,
            num_processes_preprocessing=int(
                segmentation["num_processes_preprocessing"]
            ),
            num_processes_segmentation_export=int(
                segmentation["num_processes_export"]
            ),
        )


@torch.inference_mode()
def predict_models(classifier, regressor, prepared: dict, segment_id: int, device: torch.device):
    raw = prepared["raw_image"].unsqueeze(0).to(device)
    probability = prepared["probability"].unsqueeze(0).to(device)
    mask = prepared["mask"].unsqueeze(0).to(device)
    slice_mask = prepared["slice_mask"].unsqueeze(0).to(device)
    segment = torch.tensor([segment_id], dtype=torch.long, device=device)
    classification = classifier(raw, probability, mask, segment, slice_mask)
    regression = regressor(raw, probability, mask, segment, slice_mask)
    return classification, regression


def remove_unneeded_nnunet_metadata(
    segmentation_dir: Path,
    keep_probabilities: bool = False,
) -> None:
    """Remove nnU-Net metadata and optionally retain probability arrays."""
    for name in (
        "dataset.json",
        "plans.json",
        "predict_from_raw_data_args.json",
    ):
        path = segmentation_dir / name
        if path.exists():
            path.unlink()
    for path in segmentation_dir.glob("*.pkl"):
        path.unlink()
    if not keep_probabilities:
        for path in segmentation_dir.glob("*.npz"):
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end BME segmentation, slice/segment classification, "
            "and SPARCC regression."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "inference.yaml",
        help="Inference YAML path.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "examples" / "input",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "examples" / "output",
    )
    parser.add_argument(
        "--save-probabilities",
        action="store_true",
        help=(
            "Retain nnU-Net foreground probability arrays as "
            "<sample_id>.npz files in the segmentation output directory."
        ),
    )
    parser.add_argument(
        "--skip-segmentation",
        action="store_true",
        help="Reuse existing <sample_id>.nii.gz/.npz files in the output segmentation directory.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    configure_reproducible_inference()
    device = resolve_device(str(config.get("device", "auto")))
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    segmentation_dir = output_dir / "segmentation"
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    cases = discover_cases(input_dir)

    print(f"Device: {device}")
    print(f"Cases: {len(cases)}")
    if not args.skip_segmentation:
        run_segmentation(input_dir, segmentation_dir, config, device)

    for case in cases:
        mask_path = segmentation_dir / f"{case['sample_id']}.nii.gz"
        probability_path = (
            segmentation_dir / f"{case['sample_id']}.npz"
        )
        if not mask_path.exists() or not probability_path.exists():
            raise FileNotFoundError(
                f"Missing nnU-Net output for {case['sample_id']}: "
                f"{mask_path}, {probability_path}"
            )
        case["mask_path"] = mask_path
        case["probability_path"] = probability_path

    classifier = load_prediction_model(
        resolve_model_path(config["classification"]["checkpoint"]),
        "classifier",
        device,
    )
    regressor = load_prediction_model(
        resolve_model_path(config["regression"]["checkpoint"]),
        "regressor",
        device,
    )

    slice_rows: list[dict] = []
    segment_rows: list[dict] = []
    for case in cases:
        prepared = prepare_case(
            case["raw_path"],
            case["probability_path"],
            size=config["preprocessing"]["hw_size"],
            depth=int(config["preprocessing"]["max_slices"]),
        )
        class_output, regression_output = predict_models(
            classifier,
            regressor,
            prepared,
            case["segment_id"],
            device,
        )
        slice_probabilities = class_output["slice_probabilities"][0].detach().cpu()
        valid_mask = prepared["slice_mask"]
        original_indices = prepared["original_slice_indices"]
        slice_threshold = float(
            config["classification"]["slice_threshold"]
        )
        for model_index in range(int(valid_mask.numel())):
            if not bool(valid_mask[model_index]):
                continue
            slice_probability = float(slice_probabilities[model_index])
            slice_rows.append(
                {
                    "patient_id": case["patient_id"],
                    "sample_id": case["sample_id"],
                    "segment": case["segment"],
                    "model_slice_index": model_index,
                    "original_slice_index": int(original_indices[model_index]),
                    "slice_probability": slice_probability,
                    "slice_predicted_bme": int(
                        slice_probability >= slice_threshold
                    ),
                }
            )

        segment_probability = float(class_output["prob_bme"][0].detach().cpu())
        segment_threshold = float(config["classification"]["segment_threshold"])
        sparcc_score_raw = float(regression_output["sparcc_score"][0].detach().cpu())
        sparcc_score_raw = min(
            max(sparcc_score_raw, float(config["regression"]["score_min"])),
            float(config["regression"]["score_max"]),
        )
        regression_indices = original_indices[valid_mask].tolist()
        crop_start = int(prepared["crop_start"])
        segment_predicted_bme = int(
            segment_probability >= segment_threshold
        )
        sparcc_score = (
            sparcc_score_raw if segment_predicted_bme else 0.0
        )
        segment_rows.append(
            {
                "patient_id": case["patient_id"],
                "sample_id": case["sample_id"],
                "segment": case["segment"],
                "num_original_slices": int(prepared["original_depth"]),
                "segment_probability": segment_probability,
                "segment_predicted_bme": segment_predicted_bme,
                "sparcc_score_raw": sparcc_score_raw,
                "sparcc_score": sparcc_score,
                "regression_crop_start": crop_start,
                "regression_original_slice_indices": ";".join(
                    map(str, regression_indices)
                ),
                "bme_mask_path": str(
                    Path("segmentation") / case["mask_path"].name
                ),
            }
        )

    slice_frame = pd.DataFrame(slice_rows)
    segment_frame = pd.DataFrame(segment_rows)
    patient_frame = (
        segment_frame.groupby("patient_id", as_index=False)
        .agg(
            n_segments=("sample_id", "nunique"),
            n_bme_positive_segments=("segment_predicted_bme", "sum"),
            sparcc_score_sum=("sparcc_score", "sum"),
        )
        .sort_values("patient_id")
    )
    slice_frame.to_csv(
        output_dir / "slice_classification.csv",
        index=False,
        encoding="utf-8-sig",
    )
    segment_frame.to_csv(
        output_dir / "segment_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    patient_frame.to_csv(
        output_dir / "patient_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    remove_unneeded_nnunet_metadata(
        segmentation_dir,
        keep_probabilities=bool(args.save_probabilities),
    )
    print("\nSegment results")
    print(
        segment_frame[
            [
                "sample_id",
                "segment_predicted_bme",
                "sparcc_score",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
