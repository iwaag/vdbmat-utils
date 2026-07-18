"""Slice-PNG + sidecar-manifest export: validation, atomic publish, IO.

Builds the output in a sibling temporary directory and publishes it with a
single atomic rename, so an interrupted or failing export never leaves a
partial ``<out>/<name>/`` (mirrors the ``vdbmat`` pipeline runner's publish
convention). PNG writing, sha256 accumulation, and the real-color-usage
recheck all happen per slice so the full output is never held in memory at
once.
"""

import dataclasses
import hashlib
import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import numpy.typing as npt
from vdbmat.core import MaterialRole
from vdbmat.io import inspect_material_label_manifest, read_material_label_manifest

from vdbmat_utils.core import config_digest
from vdbmat_utils.image.png import read_indexed_png, write_indexed_png
from vdbmat_utils.printer import PrintSlicesError
from vdbmat_utils.printer.sampler import build_sampling_plan, sample_slice
from vdbmat_utils.printer.types import PrintSlicesConfig

FORMAT_NAME = "vdbmat.print-slices"
FORMAT_VERSION = "1.0.0"


@dataclasses.dataclass(frozen=True, slots=True)
class ExportResult:
    """Summary of a completed export, for the CLI success message."""

    output_dir: Path
    manifest_path: Path
    n_slices: int
    width: int
    height: int
    physical_mm: tuple[float, float, float]
    material_pixel_counts: dict[int, int]


def export_print_slices(
    manifest_path: Path, config: PrintSlicesConfig, out_dir: Path, name: str
) -> ExportResult:
    """Export ``manifest_path`` to ``<out_dir>/<name>/`` per ``config``."""
    output_dir = out_dir / name
    if output_dir.exists():
        raise PrintSlicesError(
            "out", f"output already exists, refusing to overwrite: {output_dir}"
        )

    volume = read_material_label_manifest(manifest_path)

    non_background_ids = {
        material.material_id
        for material in volume.palette
        if material.role is not MaterialRole.BACKGROUND
    }
    config_ids = {int(key) for key in config.palette}
    missing = non_background_ids - config_ids
    extra = config_ids - non_background_ids
    if missing:
        raise PrintSlicesError(
            "palette",
            f"input has material ids not declared in the palette: "
            f"{sorted(missing)}",
        )
    if extra:
        raise PrintSlicesError(
            "palette",
            f"palette declares material ids absent from the input: "
            f"{sorted(extra)}",
        )

    plan = build_sampling_plan(
        volume.geometry.shape_zyx, volume.geometry.voxel_size_xyz_m, config
    )

    ordered_ids = sorted(config_ids)
    index_by_material_id: dict[int, int] = {0: 0}
    palette_rgb: list[tuple[int, int, int]] = [config.background_rgb]
    for index, material_id in enumerate(ordered_ids, start=1):
        index_by_material_id[material_id] = index
        palette_rgb.append(config.palette[str(material_id)])

    lookup = np.zeros(max(index_by_material_id) + 1, dtype=np.uint8)
    for material_id, index in index_by_material_id.items():
        lookup[material_id] = index

    max_number = config.index_start + plan.grid.n_slices - 1
    digits = max(4, len(str(max_number)))

    out_dir.mkdir(parents=True, exist_ok=True)
    temporary = out_dir / f".{name}.tmp-{uuid.uuid4().hex[:12]}"
    temporary.mkdir(parents=True)

    material_pixel_counts: dict[int, int] = dict.fromkeys(index_by_material_id, 0)
    slice_files: list[dict[str, str]] = []
    try:
        for output_index in range(plan.grid.n_slices):
            material_2d = sample_slice(volume.material_id, plan, output_index)
            indices_2d = lookup[material_2d]

            number = output_index + config.index_start
            filename = f"{config.name_prefix}{number:0{digits}d}.png"
            slice_path = temporary / filename
            write_indexed_png(slice_path, indices_2d, palette_rgb)

            _recheck_actual_colors(slice_path, filename, len(palette_rgb))
            _accumulate_material_counts(material_2d, material_pixel_counts)

            slice_files.append(
                {
                    "name": filename,
                    "sha256": hashlib.sha256(slice_path.read_bytes()).hexdigest(),
                }
            )

        manifest = _build_manifest(
            manifest_path=manifest_path,
            config=config,
            plan=plan,
            index_by_material_id=index_by_material_id,
            palette_rgb=palette_rgb,
            volume=volume,
            slice_files=slice_files,
            digits=digits,
        )
        (temporary / f"{name}.printslices.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    temporary.rename(output_dir)

    return ExportResult(
        output_dir=output_dir,
        manifest_path=output_dir / f"{name}.printslices.json",
        n_slices=plan.grid.n_slices,
        width=plan.grid.width,
        height=plan.grid.height,
        physical_mm=(
            plan.grid.width * plan.grid.pitch_x_m * 1000.0,
            plan.grid.height * plan.grid.pitch_y_m * 1000.0,
            plan.grid.n_slices * plan.grid.pitch_z_m * 1000.0,
        ),
        material_pixel_counts=material_pixel_counts,
    )


def _recheck_actual_colors(slice_path: Path, filename: str, palette_size: int) -> None:
    decoded_indices, _ = read_indexed_png(slice_path)
    declared = set(range(palette_size))
    observed = set(np.unique(decoded_indices).tolist())
    if not observed.issubset(declared):
        raise PrintSlicesError(
            "slice",
            f"{filename}: decoded palette indices {sorted(observed)} are not "
            f"a subset of the declared set {sorted(declared)} (internal bug)",
        )


def _accumulate_material_counts(
    material_2d: npt.NDArray[np.uint16], counts: dict[int, int]
) -> None:
    ids, occurrences = np.unique(material_2d, return_counts=True)
    for material_id, occurrence in zip(ids.tolist(), occurrences.tolist(), strict=True):
        counts[material_id] += occurrence


def _build_manifest(
    *,
    manifest_path: Path,
    config: PrintSlicesConfig,
    plan,
    index_by_material_id: dict[int, int],
    palette_rgb: list[tuple[int, int, int]],
    volume,
    slice_files: list[dict[str, str]],
    digits: int,
) -> dict:
    inspection = inspect_material_label_manifest(manifest_path)
    manifest_sha256 = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()

    palette_manifest: dict[str, dict] = {}
    for material_id in sorted(index_by_material_id):
        material = volume.palette.by_id(material_id)
        palette_manifest[str(material_id)] = {
            "name": material.name,
            "role": material.role.value,
            "rgb": list(palette_rgb[index_by_material_id[material_id]]),
        }

    grid = plan.grid
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "source": {
            "manifest_path": Path(manifest_path).name,
            "manifest_sha256": manifest_sha256,
            "payload_sha256": inspection.payload_sha256,
        },
        "config_digest": config_digest(config),
        "printer": {
            "dpi_x": config.dpi_x,
            "dpi_y": config.dpi_y,
            "layer_thickness_m": config.layer_thickness_m,
            "pitch_x_mm": grid.pitch_x_m * 1000.0,
            "pitch_y_mm": grid.pitch_y_m * 1000.0,
            "pitch_z_mm": grid.pitch_z_m * 1000.0,
        },
        "grid": {
            "n_slices": grid.n_slices,
            "width_px": grid.width,
            "height_px": grid.height,
            "physical_mm": {
                "x": grid.width * grid.pitch_x_m * 1000.0,
                "y": grid.height * grid.pitch_y_m * 1000.0,
                "z": grid.n_slices * grid.pitch_z_m * 1000.0,
            },
        },
        "palette": palette_manifest,
        "background_rgb": list(config.background_rgb),
        "slices": {
            "name_prefix": config.name_prefix,
            "index_start": config.index_start,
            "digits": digits,
            "files": slice_files,
        },
    }
