"""
Read train/val split for traning data.
Split is defined in traning data/dataset_split.json.

Current rule: last_video_per_folder — in each folder, the last video (by name) is val, the rest are train.
"""
import os
import json


def _root():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "traning data")


def _load_split():
    path = os.path.join(_root(), "dataset_split.json")
    root = _root()
    if not os.path.isfile(path):
        data = {"split_type": "last_video_per_folder", "folders": []}
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    # Auto-discover folders when empty: all subdirs of traning data that contain .mp4 (new data folders included)
    if data.get("split_type") == "last_video_per_folder":
        folders = data.get("folders") or []
        if not folders and os.path.isdir(root):
            folders = [
                d for d in sorted(os.listdir(root))
                if os.path.isdir(os.path.join(root, d))
                and _mp4_in_folder(os.path.join(root, d))
            ]
        data["folders"] = folders
    return data


def _mp4_in_folder(folder_path):
    """Sorted list of .mp4 filenames in folder."""
    if not os.path.isdir(folder_path):
        return []
    return sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(".mp4")]
    )


def get_train_video_paths():
    """All videos that are NOT the last in their folder. Returns list of absolute paths."""
    root = _root()
    data = _load_split()
    if data.get("split_type") != "last_video_per_folder":
        return []
    paths = []
    for name in data.get("folders", []):
        folder = os.path.join(root, name)
        mp4s = _mp4_in_folder(folder)
        # all but last
        for f in mp4s[:-1]:
            paths.append(os.path.join(folder, f))
    return paths


def get_val_video_paths():
    """Last video in each folder. Returns list of absolute paths."""
    root = _root()
    data = _load_split()
    if data.get("split_type") != "last_video_per_folder":
        return []
    paths = []
    for name in data.get("folders", []):
        folder = os.path.join(root, name)
        mp4s = _mp4_in_folder(folder)
        if mp4s:
            paths.append(os.path.join(folder, mp4s[-1]))
    return paths


def get_train_folders():
    """All folders (train videos come from each folder, excluding last video per folder)."""
    root = _root()
    data = _load_split()
    return [os.path.join(root, d) for d in data.get("folders", [])]


def get_val_folders():
    """Same as get_train_folders(); val is per-video, not per-folder. Prefer get_val_video_paths()."""
    return get_train_folders()


def get_train_and_val_video_paths():
    """Return (train_paths, val_paths) as lists of absolute .mp4 paths."""
    return get_train_video_paths(), get_val_video_paths()


def get_train_folders_with_videos():
    """For training scripts: list of (folder_abs_path, set_of_train_video_basenames).
    Each folder gets only its train videos (all .mp4 in folder except the last by name).
    New subfolders under traning data are included when folders in JSON is empty."""
    root = _root()
    data = _load_split()
    if data.get("split_type") != "last_video_per_folder":
        return []
    out = []
    for name in data.get("folders", []):
        folder = os.path.join(root, name)
        mp4s = _mp4_in_folder(folder)
        train_basenames = set(mp4s[:-1])  # all but last
        if train_basenames:
            out.append((folder, train_basenames))
    return out
