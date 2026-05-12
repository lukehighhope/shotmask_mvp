# shotmask_mvp

Offline Python pipeline for **beep timing** + **gunshot detection** from video audio, optional **motion cross-check**, and **overlay export** (`outputs/overlay.webm`). CNN / AST / heuristic scoring live under `detectors/`.

---

## Prerequisites

- **Python** 3.10+ recommended  
- **FFmpeg** on `PATH` (or standard Windows installs; some scripts probe `C:\ffmpeg\…`)  
- **PyTorch** (CPU or CUDA — match your installer to your GPU)  

---

## Setup

```powershell
cd path\to\shotmask_mvp
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Versions are pinned in `requirements.txt`. For **CUDA** PyTorch, follow the comment at the top of that file (official `download.pytorch.org` wheel index) or https://pytorch.org/.

**Windows & Unicode paths** (Chinese folder names, etc.): prefer UTF-8 in the terminal when training/logging:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
```

---

## Configuration

- **`calibrated_detector_params.json`** (repo root): thresholds, `cnn_gunshot_path`, `ast_gunshot_path`, `use_cnn_only` / `use_ast_gunshot`, clustering, etc.  
- Resolved model paths use **`detectors/config_paths.py`** (paths relative to repo root are OK).

---

## Training data layout

Default layout is a folder **`training data/`** next to the repo (tracked JSON / small assets; `.mp4` usually ignored via `.gitignore`).  

Override the root directory with:

```powershell
set SHOTMASK_TRAINING_DATA_ROOT=D:\shotmask_data\training data
```

That directory should contain subsets (e.g. `outdoor/`, `indoor/`) and **`dataset_split.json`** defining explicit `train` / `val` lists (paths relative to this root).

- **Mirror of split in git:** `training data/dataset_split.json` (for layout reference; live data stays on disk).  
- **Regenerate split from all `*cali.txt`:**  
  `python refresh_dataset_split_from_cali.py` (with `SHOTMASK_TRAINING_DATA_ROOT` set if data is external).

Design notes and knob meanings: **`keyfactors.txt`**.

---

## Quick start — full pipeline

```powershell
python main.py --video "path\to\video.mp4" --mode all
```

Modes: `beep` | `shots` | `motion` | `all`. Outputs include `outputs/events.json`, `outputs/shot_times_since_beep.txt`, and overlay encode.

Optional: `--nms 0.06`, `--shots-filter strict|balanced`, `--sweep-threshold`, `--grid-search` (when reference labels exist beside the video).

---

## Offline evaluation (val set, pooled P/R/F1)

Uses the same **default match tolerance** as `evaluate_multivideo.TOL` (**±0.06 s** unless you pass `--tol`).  
`main.py` imports this value as **`GT_MATCH_TOLERANCE`** so GT diagnostics match offline eval defaults.

 CNN-only scoring on **`dataset_split.json` → val**:

```powershell
set SHOTMASK_TRAINING_DATA_ROOT=D:\shotmask_data\training data
python evaluate_multivideo.py --use-split --cnn-only --tol 0.06
```

**Track progress across training runs** (append one JSON line per run — includes pooled metrics, per-video breakdown, split fingerprint):

```powershell
python evaluate_multivideo.py --use-split --cnn-only --record-jsonl outputs/detection_val_benchmark.jsonl --record-tag my_run_label
```

- **Training-time AST Val R/F1** (`train_audio_gunshot.py`) measures **classification on detector candidates**, not the same thing as pooled timeline metrics above — do not compare the raw numbers directly.

---

## Training (short)

**CNN gunshot classifier (Mel, train paths from split):**

```powershell
set SHOTMASK_TRAINING_DATA_ROOT=D:\shotmask_data\training data
python train_cnn_gunshot.py --use-split --epochs 40 --augment --save-config --out outputs/cnn_gunshot.pt
```

**AST (see script docstring for flags):**

```powershell
python train_audio_gunshot.py --epochs 50 --augment
```

After training, run **`evaluate_multivideo`** (and optionally **`--record-jsonl`**) on the **same split fingerprint** before claiming improvement.

---

## Annotation & calibration UIs

- **`annotate_shots.py`** — in-browser waveform + markers; env **`SHOTMASK_PICKER_DIR`**, **`SHOTMASK_TRAINING_DATA_ROOT`**.  
- **`extract_audio_plot.py --calibration`** — Calibration Viewer (HTTP server; supports video **Range** requests for seeking).  
- **`start_annotate.bat`** — example launcher (edit data paths inside).

---

## Tests and CI

Automated checks use **pytest**:

```powershell
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
```

- Unit tests (`tests/test_evaluate_core.py`, `tests/test_dataset_split_and_paths.py`) need no FFmpeg/GPU.
- **Golden regression** (`tests/test_golden_gunshot.py`): a short PD gunshot-derived clip (**`golden_gunshot.wav`**) runs **`detect_shots`** with **`outputs/cnn_gunshot.pt`** and the calibrated **confidence gate**, matching **five** onset times ±50 ms (**`golden_gunshot_expected.json`**; snapshot from **CPU** via `scripts/refresh_golden_expected.py` because CI runs CPU-only Torch). Attribution: **`tests/fixtures/GOLDEN_AUDIO_ATTRIBUTION.txt`**.
- Rebuild WAV from Commons + refresh expected JSON:

  ```powershell
  pip install -r requirements-dev.txt
  python scripts/build_real_golden_wav.py
  python scripts/refresh_golden_expected.py
  ```

- **ffmpeg** (`test_golden_mp4_*`): CI installs apt `ffmpeg`; for local PowerShell installs, use your own FFmpeg or rely on **`imageio-ffmpeg`** bundled by `requirements-dev.txt` (picked up indirectly by the build helper).

On **GitHub**, **`.github/workflows/ci.yml`** runs `ffmpeg` setup (for the mux test), installs dependencies, then `pytest` on **Ubuntu** / Python 3.11 for pushes and pull requests.

### Error taxonomy (FP/FN pareto)

Structured labels live in **`error_analysis/taxonomy.json`**. Export one row per FP/FN plus automatic **hints**:

`python error_analysis/export_manifest.py --use-split --cnn-only -o outputs/error_audit_manifest.jsonl`  
Annotate **`human_primary`** (taxonomy ids, e.g. `FP_ECHO`, `FN_RAPID_PAIR`) then:  
`python error_analysis/summarize_labels.py outputs/error_audit_manifest.jsonl --combined --target-pct 0.80`

---

## Docs

Markdown notes live in **`docs/`** (accuracy analyses, detector options, etc.).
