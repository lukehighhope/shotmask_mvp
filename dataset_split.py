"""
Read train/val split for training data.
Split is defined in training data/dataset_split.json.

Supported:
- Explicit lists: \"train\" / \"val\" with paths relative to training data/
- Legacy: last_video_per_folder + \"folders\"
"""
import os
import json
from collections import defaultdict


from training_data_root import get_training_data_root


def _root():
    return get_training_data_root()


def _load_split():
    path = os.path.join(_root(), "dataset_split.json")
    root = _root()
    if not os.path.isfile(path):
        data = {"split_type": "last_video_per_folder", "folders": []}
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    # Auto-discover folders when empty: all subdirs of training data that contain .mp4 (new data folders included)
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


def _explicit_mp4_paths_relative_to_root(root, rels):
    """Resolve list of paths relative to training data root; skip missing files."""
    out = []
    for rel in rels or []:
        if not isinstance(rel, str):
            continue
        rel = rel.replace("\\", "/").strip()
        if not rel.lower().endswith(".mp4"):
            continue
        full = os.path.normpath(os.path.join(root, rel))
        if os.path.isfile(full):
            out.append(full)
    return out


def get_train_video_paths():
    """Training video absolute paths: explicit \"train\" list in dataset_split.json if present, else all but last mp4 per folder (legacy)."""
    root = _root()
    path = os.path.join(root, "dataset_split.json")
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8-sig") as f:
                data = json.load(f)
        except Exception:
            data = {}
        train_list = data.get("train")
        if isinstance(train_list, list) and train_list:
            return _explicit_mp4_paths_relative_to_root(root, train_list)
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
    """Validation video absolute paths: explicit \"val\" list in dataset_split.json if present, else last mp4 per folder (legacy)."""
    root = _root()
    path = os.path.join(root, "dataset_split.json")
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8-sig") as f:
                data = json.load(f)
        except Exception:
            data = {}
        val_list = data.get("val")
        if isinstance(val_list, list) and val_list:
            return _explicit_mp4_paths_relative_to_root(root, val_list)
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


def _train_folders_from_explicit_train_list(root, train_rels):
    """Group explicit train relative paths by parent folder -> [(folder_abs, set(basenames)), ...]."""
    by_folder = defaultdict(set)
    for rel in train_rels:
        rel = rel.replace("\\", "/").strip()
        if not rel.lower().endswith(".mp4"):
            continue
        full = os.path.normpath(os.path.join(root, rel))
        folder = os.path.dirname(full)
        base = os.path.basename(full)
        by_folder[folder].add(base)
    return [(fd, names) for fd, names in sorted(by_folder.items())]


def get_train_folders_with_videos():
    """For training scripts: list of (folder_abs_path, set_of_train_video_basenames).
    Uses explicit \"train\" list in dataset_split.json when present; otherwise
    last_video_per_folder (all .mp4 in folder except last by name).
    """
    root = _root()
    path = os.path.join(root, "dataset_split.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
        train_list = data.get("train")
        if isinstance(train_list, list) and train_list:
            return _train_folders_from_explicit_train_list(root, train_list)

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


def get_all_folders_with_videos_from_split():
    """Like get_train_folders_with_videos but uses explicit \"train\" + \"val\" lists (full labeled set)."""
    root = _root()
    path = os.path.join(root, "dataset_split.json")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    train_list = data.get("train") or []
    val_list = data.get("val") or []
    combined = []
    if isinstance(train_list, list):
        combined.extend(train_list)
    if isinstance(val_list, list):
        combined.extend(val_list)
    if not combined:
        return []
    return _train_folders_from_explicit_train_list(root, combined)
