"""Resolve model paths in calibrated_detector_params.json relative to repo root."""
import os

CALIBRATED_BASENAME = "calibrated_detector_params.json"


def project_root():
    """Directory containing calibrated_detector_params.json (repository root)."""
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def resolve_model_path(path_str):
    """
    If path_str is relative, resolve against project root. Absolute paths unchanged.
    """
    if not path_str:
        return path_str
    if os.path.isabs(path_str):
        return os.path.normpath(path_str)
    return os.path.normpath(os.path.join(project_root(), path_str))
