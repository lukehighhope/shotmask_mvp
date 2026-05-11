"""
Training / split data root: same layout as repo's default ``traning data/`` (dataset_split.json,
``outdoor/``, ``indoor/``, etc.).

Override with environment variable **SHOTMASK_TRAINING_DATA_ROOT** (absolute path).

Example: if videos live under ``D:\\shotmask_data\\traning data\\outdoor\\...`` and JSON paths
are ``outdoor/foo/bar.mp4``, set::

    set SHOTMASK_TRAINING_DATA_ROOT=D:\\shotmask_data\\traning data

(not the ``...\\outdoor`` folder alone, unless you change split paths accordingly).
"""
import os

_ENV = "SHOTMASK_TRAINING_DATA_ROOT"


def get_training_data_root():
    v = os.environ.get(_ENV, "").strip()
    if v:
        return os.path.normpath(os.path.expandvars(v))
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "traning data")
    )
