from __future__ import annotations

from pathlib import Path

from conftest import load_module


REQUIRED_DIRS = [
    "configs",
    "data",
    "scripts",
    "scripts/geo",
    "scripts/air",
    "scripts/met",
    "scripts/fire",
    "src/data",
    "src/features",
    "src/models",
    "src/utils",
    "results/logs",
    "results/models",
]

REQUIRED_FILES = [
    "configs/counties_sjv.json",
    "configs/dates_config.json",
    "scripts/geo/get_county_centroids.py",
    "scripts/air/fetch_airnow_history.py",
    "scripts/met/fetch_openmeteo_history.py",
    "scripts/fire/fetch_eonet_wildfire_data.py",
    "src/data/merge_datasets.py",
    "src/features/build_features.py",
    "src/models/train_models.py",
    "src/utils/date_config.py",
]

MODULE_FILES = [
    "scripts/geo/get_county_centroids.py",
    "scripts/air/fetch_airnow_history.py",
    "scripts/met/fetch_openmeteo_history.py",
    "scripts/fire/fetch_eonet_wildfire_data.py",
    "src/data/merge_datasets.py",
    "src/features/build_features.py",
    "src/models/train_models.py",
    "src/utils/date_config.py",
]

SCRIPT_MODULE_FILES = [
    "scripts/geo/get_county_centroids.py",
    "scripts/air/fetch_airnow_history.py",
    "scripts/met/fetch_openmeteo_history.py",
    "scripts/fire/fetch_eonet_wildfire_data.py",
    "src/data/merge_datasets.py",
    "src/features/build_features.py",
    "src/models/train_models.py",
]


def test_repo_smoke_structure_and_imports() -> None:
    root = Path(__file__).resolve().parents[1]

    for rel_dir in REQUIRED_DIRS:
        assert (root / rel_dir).exists(), f"Missing directory: {rel_dir}"

    for rel_file in REQUIRED_FILES:
        assert (root / rel_file).exists(), f"Missing file: {rel_file}"

    for rel_file in MODULE_FILES:
        module = load_module(rel_file)
        if rel_file in SCRIPT_MODULE_FILES:
            assert hasattr(module, "main")
            assert callable(module.main)
        else:
            assert hasattr(module, "get_data_collection_range")
            assert hasattr(module, "get_train_val_test_ranges")
