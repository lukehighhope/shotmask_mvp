import json
from pathlib import Path

import detectors.config_paths as config_paths


def test_project_root_contains_calibrated_basename_repo():
    root = config_paths.project_root()
    assert (Path(root) / config_paths.CALIBRATED_BASENAME).is_file()


def test_resolve_model_path_absolute_unchanged(tmp_path):
    p = tmp_path / "abs_only.pt"
    p.write_bytes(b"x")
    assert config_paths.resolve_model_path(str(p)) == str(p.resolve())


def test_resolve_model_path_relative_joins_root():
    rel = "outputs/foo.pt"
    expected = Path(config_paths.project_root()) / rel
    assert config_paths.resolve_model_path(rel) == str(Path(expected))


def test_get_training_data_root_env_override(monkeypatch, tmp_path):
    from training_data_root import get_training_data_root

    root = Path(tmp_path) / "data root"
    root.mkdir()
    monkeypatch.setenv("SHOTMASK_TRAINING_DATA_ROOT", str(root))
    assert get_training_data_root() == str(root.resolve())


def test_explicit_val_paths(monkeypatch, tmp_path):
    from dataset_split import get_train_video_paths, get_val_video_paths

    root = Path(tmp_path) / "td"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir(parents=True)
    f_train = root / "a" / "t.mp4"
    f_val = root / "b" / "v.mp4"
    f_train.write_bytes(b"")
    f_val.write_bytes(b"")
    split = root / "dataset_split.json"
    split.write_text(
        json.dumps(
            {
                "train": ["a/t.mp4"],
                "val": ["b/v.mp4"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SHOTMASK_TRAINING_DATA_ROOT", str(root))
    val_abs = get_val_video_paths()
    train_abs = get_train_video_paths()
    assert len(val_abs) == 1
    assert Path(val_abs[0]).resolve() == f_val.resolve()
    assert len(train_abs) == 1
    assert Path(train_abs[0]).resolve() == f_train.resolve()


def test_last_video_per_folder_legacy(monkeypatch, tmp_path):
    from dataset_split import get_train_video_paths, get_val_video_paths

    root = Path(tmp_path) / "td"
    fold = root / "fold"
    fold.mkdir(parents=True)
    (fold / "1.mp4").write_bytes(b"")
    (fold / "2.mp4").write_bytes(b"")
    (root / "dataset_split.json").write_text(
        json.dumps({"split_type": "last_video_per_folder", "folders": ["fold"]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("SHOTMASK_TRAINING_DATA_ROOT", str(root))
    train = get_train_video_paths()
    val = get_val_video_paths()
    assert len(train) == 1
    assert train[0].endswith("1.mp4")
    assert len(val) == 1
    assert val[0].endswith("2.mp4")
