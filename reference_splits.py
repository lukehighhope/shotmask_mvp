"""
Reference gunshot splits: first = beep→1st shot (s), rest = inter-shot intervals (s).
Shot times = beep_t + cumsum(splits).
"""
import numpy as np

# Row 1–4 + Row 5 first value; 29 shots total. First split 2.08 = beep→1st shot, then 0.26, 0.44, ...
REFERENCE_SPLITS = [
    2.08, 0.26, 0.44, 0.26, 0.48, 0.25, 3.56,   # row 1
    0.31, 0.61, 0.31, 0.69, 0.34, 1.30, 0.31,   # row 2
    0.49, 0.25, 0.63, 0.38, 0.83, 0.32, 1.56,   # row 3
    0.28, 0.50, 0.28, 0.21, 0.41, 0.23, 0.41,   # row 4
    0.56,                                       # row 5: (29) → split 0.56
]


def ref_shot_times(beep_t):
    """Reference shot times (s) from beep time: beep_t + cumsum(splits)."""
    return (beep_t + np.cumsum(REFERENCE_SPLITS)).tolist()
