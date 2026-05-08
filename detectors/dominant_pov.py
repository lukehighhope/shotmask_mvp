"""
Assume Hawkveil-style POV: one dominant gun + recording chain per clip.
Given final shot detections + per-candidate spectral features from Stage2,
cluster shots in feature space and keep only the largest cluster (dominant shooter).

This is a prototype: features are engineered (not neural embeddings).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

FEATURE_KEYS = (
    "onset_log",
    "r1_log",
    "r2_log",
    "flatness",
    "attack_norm",
    "E_low_ratio",
    "E_high_ratio",
    "mfcc_band",
)


def _row(feat: Dict[str, Any]) -> np.ndarray:
    return np.array([float(feat.get(k, 0.0)) for k in FEATURE_KEYS], dtype=np.float64)


def match_shots_to_candidates(
    shots: List[Dict[str, Any]],
    candidate_features: List[Dict[str, Any]],
    time_tol: float = 0.10,
) -> List[Optional[Dict[str, Any]]]:
    out: List[Optional[Dict[str, Any]]] = []
    for s in shots:
        t = float(s["t"])
        best_cf: Optional[Dict[str, Any]] = None
        best_dt = time_tol + 1.0
        for cf in candidate_features:
            dt = abs(float(cf["t"]) - t)
            if dt < best_dt:
                best_dt, best_cf = dt, cf
        out.append(best_cf if best_dt <= time_tol else None)
    return out


def _pick_dominant_label(labels: np.ndarray, confidences: np.ndarray) -> int:
    k = int(labels.max()) + 1 if len(labels) else 0
    best_c = 0
    best_key = (-1, -1.0)
    for c in range(k):
        m = labels == c
        n = int(m.sum())
        if n == 0:
            continue
        key = (n, float(confidences[m].mean()))
        if key > best_key:
            best_key, best_c = key, c
    return best_c


def filter_shots_dominant_pov(
    shots: List[Dict[str, Any]],
    candidate_features: List[Dict[str, Any]],
    *,
    k_clusters: int = 2,
    time_match_tol: float = 0.10,
    min_matched: int = 6,
    unmatched_policy: str = "keep",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    unmatched_policy: "keep" = do not drop shots we could not tie to Stage2 feats;
                       "drop" = remove unmatched (aggressive).
    """
    meta: Dict[str, Any] = {
        "n_in": len(shots),
        "matched": 0,
        "k_clusters_requested": k_clusters,
        "skipped": None,
        "dominant_label": None,
    }
    if not shots:
        return [], meta

    mates = match_shots_to_candidates(shots, candidate_features, time_tol=time_match_tol)
    meta["matched"] = sum(1 for m in mates if m is not None)

    idx_ok = [i for i, m in enumerate(mates) if m is not None]
    if len(idx_ok) < min_matched:
        meta["skipped"] = f"matched {len(idx_ok)} < min_matched {min_matched}"
        return list(shots), meta

    X = np.stack([_row(mates[i]) for i in idx_ok], axis=0)
    xf = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    std = xf.std(axis=0)
    std[std < 1e-9] = 1.0
    xf = (xf - xf.mean(axis=0)) / std

    n_samples = xf.shape[0]
    k_use = max(2, min(k_clusters, n_samples - 1)) if n_samples >= 4 else min(2, n_samples)
    if k_use < 2 or n_samples < 4:
        meta["skipped"] = "too few samples for k>=2 clustering"
        return list(shots), meta

    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k_use, random_state=0, n_init=10)
    sub_labels = km.fit_predict(xf)
    confs = np.array([float(shots[i].get("confidence", 0.5)) for i in idx_ok])
    dominant = _pick_dominant_label(sub_labels, confs)
    meta["dominant_label"] = int(dominant)
    meta["k_clusters_used"] = int(k_use)

    keep = set()
    sub_dominant = set(int(i) for i, lb in zip(idx_ok, sub_labels) if int(lb) == dominant)

    for i, s in enumerate(shots):
        if mates[i] is None:
            if unmatched_policy == "keep":
                keep.add(i)
        else:
            if i in sub_dominant:
                keep.add(i)

    filt = [shots[i] for i in sorted(keep)]
    meta["n_out"] = len(filt)
    meta["n_dropped"] = len(shots) - len(filt)
    return filt, meta


def annotate_shots_pov_cluster(
    shots: List[Dict[str, Any]],
    candidate_features: List[Dict[str, Any]],
    **kwargs: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Like filter_shots_dominant_pov but adds pov_dominant: bool to each shot; no drops."""
    mates = match_shots_to_candidates(shots, candidate_features, time_tol=float(kwargs.get("time_match_tol", 0.10)))
    meta: Dict[str, Any] = {"n_in": len(shots), "matched": sum(1 for m in mates if m is not None)}
    idx_ok = [i for i, m in enumerate(mates) if m is not None]
    min_matched = int(kwargs.get("min_matched", 6))
    k_clusters = int(kwargs.get("k_clusters", 2))

    out = [dict(s) for s in shots]
    if len(idx_ok) < min_matched or len(idx_ok) < 4:
        meta["skipped"] = "not enough matched shots"
        for s in out:
            s["pov_dominant"] = True
        return out, meta

    X = np.stack([_row(mates[i]) for i in idx_ok], axis=0)
    xf = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    std = xf.std(axis=0)
    std[std < 1e-9] = 1.0
    xf = (xf - xf.mean(axis=0)) / std
    n_samples = xf.shape[0]
    k_use = max(2, min(k_clusters, n_samples - 1))
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k_use, random_state=0, n_init=10)
    sub_labels = km.fit_predict(xf)
    confs = np.array([float(shots[i].get("confidence", 0.5)) for i in idx_ok])
    dominant = _pick_dominant_label(sub_labels, confs)
    meta["dominant_label"] = int(dominant)
    meta["k_clusters_used"] = int(k_use)

    dom_set = set()
    for i, lb in zip(idx_ok, sub_labels):
        if int(lb) == dominant:
            dom_set.add(i)

    for i, s in enumerate(out):
        if i in dom_set:
            s["pov_dominant"] = True
        elif mates[i] is None:
            s["pov_dominant"] = True
        else:
            s["pov_dominant"] = False
    return out, meta
