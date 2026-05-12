"""
手动标注工具：0.5 倍速播放 + 可拖拽竖线标注枪声/beep。
保存结果写入 *cali.txt、*beep.txt，审计另存 *fp.txt（多检时刻）、*fn.txt（漏检=补真枪时刻，同时也会并入 cali）。

Usage:
  python annotate_shots.py --video "test data/v1.mp4"
  python annotate_shots.py --video "training data/jeff 03-04/S1-main.mp4"
  （默认可用校准置信度阈值 + 枪声NMS；机检枪声不再按 beep 时刻裁掉 beep 之前的峰。无 *beep.txt 时用 detect_all_beeps 可多 beep。超多候选时用 --annotate-loose）

操作说明：
  - 右键点击波形       → 新增枪声标注线（黑线）
  - 右键点击已有黑线   → 删除该枪声
  - 拖拽黑线          → 移动枪声时间
  - Ctrl+右键         → 新增/删除 Beep（绿线）
  - 拖拽绿线          → 移动 Beep 时间
  - 点击波形空白处     → 跳转视频到该时间并播放
  - Shift+左键波形空白 → FP 审计竖线（多检疑点，洋红虚线）→ Save 写入 *fp.txt，不进 cali
  - Alt+左键波形空白   → FN（漏检真枪）：橙色标记 + **黑线并进 cali**；同一位置再点此带可删掉该 FN 及对应用黑线
  - 波形角「清空审计标记」「复制审计 JSON」；清空会顺带去掉 FN 记录在 cali 上对应那条黑线（容差对齐）
  - 保存校准          → 写入 *cali.txt、*fp.txt、*fn.txt（与 beep）
  - 保存 Beep         → 写入 *beep.txt
  - 速度按钮          → 0.25x / 0.5x（默认）/ 1x

首轮可选：python annotate_shots.py --audit-json audit.json （{"fp":[...],"fn":[...]} 单位为秒）

枪声机检后端（仅在缺少 *cali.txt 需要做机器探测时生效）优先级：
  1) 同目录  视频基名.shot_detector.txt   单行：auto | ast | cnn
  2) *cali.txt 顶部的注释行（须连续 # 开头）例如：  # shot_detector=ast
  3) 命令行  --shot-backend auto|ast|cnn  （默认为 auto → 沿用 calibrated_detector_params.json）
"""
import os
import sys
# Force unbuffered stdout so progress prints appear immediately in terminals
sys.stdout.reconfigure(line_buffering=True)
import argparse
import re
import time
import html
import hashlib
import secrets

# 把项目根目录加入 import 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_audio_plot import (
    write_calibration_viewer_html,
    get_waveform_data,
    get_audio_start_time,
    extract_audio as extract_audio_ch,
)
from detectors.beep import detect_all_beeps
from detectors.shot_audio import detect_shots
from main import get_ffmpeg_cmd, get_ffprobe_cmd, ffprobe_info, non_maximum_suppression
import subprocess
import json
import threading
import tempfile
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import tkinter as tk
from tkinter import filedialog


_ALLOWED_BACKENDS = frozenset({"auto", "ast", "cnn"})

# Repo root (annotate_shots.py directory). Used as default startup picker folder.
_ANNOTATE_SHOTS_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_picker_root(cli_dir: str | None) -> str | None:
    """Pinned folder for Tk file dialogs: --picker-dir overrides env SHOTMASK_PICKER_DIR."""
    d = (cli_dir or os.environ.get("SHOTMASK_PICKER_DIR") or "").strip()
    if not d:
        return None
    d = os.path.abspath(os.path.expanduser(os.path.expandvars(d)))
    if os.path.isdir(d):
        return d
    print(f"Warning: --picker-dir / SHOTMASK_PICKER_DIR is not a directory (ignored): {d}", flush=True)
    return None


def _tk_sync_before_dialog(root: tk.Tk) -> None:
    """Improves chances Windows honors ``initialdir`` instead of reusing last browse path for py.exe."""
    try:
        root.update_idletasks()
        root.update()
    except Exception:
        pass


def _split_cali_preamble_lines(lines):
    pre = []
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        pre.append(lines[i])
        i += 1
    return pre, lines[i:]


def _shot_detector_token_from_comment_line(line):
    stripped = line.strip()
    if not stripped.startswith("#"):
        return None
    inner = stripped[1:].strip()
    m = re.match(r"(?:shot_detector|shot_backend)\s*=\s*(auto|cnn|ast)\s*$", inner, re.I)
    if m:
        return m.group(1).lower()
    m = re.match(r"detector\s+(auto|cnn|ast)\s*$", inner, re.I)
    if m:
        return m.group(1).lower()
    return None


def _backend_from_preamble(pre_lines):
    for line in pre_lines:
        d = _shot_detector_token_from_comment_line(line)
        if d:
            return d
    return None


def _read_shot_detector_sidecar(path):
    try:
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as f:
            txt = f.read().strip().lower()
        return txt if txt in _ALLOWED_BACKENDS else None
    except OSError:
        return None


def _near_dedupe_sorted(times, tol_s=0.03):
    """Merge sorted-ish floats dropping entries within tol of previous."""
    merged = sorted(float(x) for x in times if isinstance(x, (int, float)) and x == x)
    out = []
    for t in merged:
        if out and abs(t - out[-1]) < tol_s:
            continue
        out.append(t)
    return out


def _read_float_txt_lines(path):
    """One float per line; skip blanks and #-leading comments."""
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        rows = []
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            rows.append(float(s))
        return rows


def _parse_cali_txt_file(cali_path):
    """Return (shot_times, preamble_comment_lines_from_top_of_file)."""
    with open(cali_path, encoding="utf-8") as f:
        raw_lines = [ln.rstrip("\n\r") for ln in f]
    pre, rest = _split_cali_preamble_lines(raw_lines)
    shot_times = []
    for line in rest:
        s = line.strip()
        if not s:
            continue
        shot_times.append(float(s))
    return shot_times, pre


def merge_cali_preamble_on_save(cali_path, new_floats_body):
    """Preserve leading # ... block when rewriting *cali.txt."""
    preamble_block = ""
    if os.path.isfile(cali_path):
        try:
            with open(cali_path, encoding="utf-8") as f:
                pre, _ = _split_cali_preamble_lines(
                    [ln.rstrip("\n\r") for ln in f.readlines()]
                )
            if pre:
                preamble_block = "\n".join(pre)
        except OSError:
            pass
    nb = (new_floats_body or "").strip()
    if preamble_block:
        sep = "\n\n" if nb else "\n"
        return preamble_block + sep + nb + ("\n" if nb else "")
    return nb + ("\n" if nb else "")


def _resolve_shot_detector_backend(anno_dir, video_base, cali_path, session_default):
    side = os.path.join(anno_dir, f"{video_base}.shot_detector.txt")
    sb = _read_shot_detector_sidecar(side)
    if sb:
        return sb, f"sidecar {os.path.basename(side)}"
    if cali_path and os.path.isfile(cali_path):
        try:
            _, pre = _parse_cali_txt_file(cali_path)
            sb = _backend_from_preamble(pre)
            if sb:
                return sb, "top of existing *cali.txt"
        except (OSError, ValueError):
            pass
    d = (session_default or "auto").strip().lower()
    d = d if d in _ALLOWED_BACKENDS else "auto"
    return d, f"session default ({d})"


def _detect_shots_with_backend(audio_mono, fps, backend, loose_min_confidence=False):
    """Pass loose_min_confidence=True to bypass calibrated min_confidence_threshold (hundreds of lines; old behavior)."""
    kw = {}
    if loose_min_confidence:
        kw["min_confidence_threshold_override"] = None
    backend = (backend or "auto").lower()
    if backend == "ast":
        return detect_shots(audio_mono, fps, use_ast_gunshot=True, **kw)
    if backend == "cnn":
        return detect_shots(audio_mono, fps, use_ast_gunshot=False, **kw)
    return detect_shots(audio_mono, fps, **kw)


_TOOLBAR_STATIC_MARK = 'id="btnAnnotateSave"'


def _inject_static_toolbar_html(html: str) -> str:
    """Insert Stage / Time / Save time into #ctrl if missing (idempotent).

    Handles markup variants: ``<button id="btnPlay"`` vs ``<button type="button" id="btnPlay"``,
    or content between ``<div id="ctrl"...>`` and Play that breaks the naive regex.
    """
    if _TOOLBAR_STATIC_MARK in html:
        return html
    block = (
        '<!-- shotmask: Time / Save time toolbar -->'
        '<span id="stageTimeDisplay" style="color:#ccc;font-size:13px;font-weight:bold;min-width:3em;margin-right:6px;"></span>'
        '<button type="button" id="btnAnnotateTime" title="Last shot − earliest beep (stage time)" '
        'style="padding:6px 14px;background:#446;color:#fff;border:none;border-radius:5px;cursor:pointer;font-size:16px;">Time</button>'
        '<button type="button" id="btnAnnotateSave" '
        'style="padding:6px 18px;background:#c75;color:#fff;border:none;border-radius:5px;cursor:pointer;font-size:16px;font-weight:bold;">Save time</button>'
    )
    patterns = (
        r'(<div id="ctrl"[^>]*>)\s*(<button id="btnPlay")',
        r'(<div id="ctrl"[^>]*>)\s*(<button type="button" id="btnPlay")',
        r'(<div id="ctrl"[^>]*>)\s*(<button type="button"\s+id="btnPlay")',
        r'(<div id="ctrl"[^>]*>)\s*(<button[^>]*\bid="btnPlay"\b[^>]*>)',
    )
    for pat in patterns:
        new_h, n = re.subn(pat, r"\1" + block + r"\2", html, count=1)
        if n and _TOOLBAR_STATIC_MARK in new_h:
            return new_h
    # Last resort: insert immediately before the first <button id="btnPlay" …> inside #ctrl
    m = re.search(r'<div\s+id="ctrl"[^>]*>', html)
    if not m:
        return html
    start = m.end()
    idx = html.find('<button id="btnPlay"', start)
    if idx < 0:
        return html
    if _TOOLBAR_STATIC_MARK in html[start:idx]:
        return html
    return html[:idx] + block + html[idx:]


def _video_src_cache_bust_param(video_path: str | None) -> str:
    """Unique query segment so <video src="/video?..."> never hits a stale browser cache of another file."""
    try:
        vp = os.path.normcase(os.path.abspath(video_path or ""))
    except OSError:
        vp = str(video_path or "")
    return hashlib.sha256(
        (vp + "|" + str(time.time_ns())).encode("utf-8", errors="ignore")
    ).hexdigest()[:22]


def _patch_html_playback_speed(
    html_path, default_speed=0.5, video_path: str | None = None, stream_token: str | None = None
):
    """Patch generated calibration HTML:
    1. Default playback speed
    2. Speed toggle buttons
    3. Completely replace mouse handlers with clean annotation behavior:
       - Left-click on waveform → seek + play from that time
       - Shift+left-click blank → FP audit magenta lines → persisted *fp.txt (not *cali)
       - Alt+left-click → FN (miss): orange line **and** calibration shot/black line (*cali* + *fn.txt*)
       - Save time posts *cali.txt, *beep.txt, *fp.txt, *fn.txt*
       - Left-drag empty waveform → pan
       - Right-click = add/delete shot/beep; removes nearby FP/FN markers when deleting a shot line
       - Scroll = pan; Ctrl+scroll = zoom
    """
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Session-only视频 URL：/v/<token>，路径每次换视频都变，浏览器无法把 ken0102 的缓存当成别的文件。
    _src = rf"/v/{stream_token}" if stream_token else rf"/video?c={_video_src_cache_bust_param(video_path)}"
    html, _n_vid = re.subn(
        r'(<video\s+id="vid"\s+src=")([^"]*)(")',
        r"\1" + _src + r'"',
        html,
        count=1,
    )
    if not _n_vid:
        html, _n_vid = re.subn(
            r'(<video\b[^>]*\bid\s*=\s*"vid"[^>]*\ssrc=")([^"]*)(")',
            r"\1" + _src + r"\3",
            html,
            count=1,
        )
    if not _n_vid:
        html, _n_vid = re.subn(
            r'(<video\b[^>]*\ssrc=")([^"]*)("[^>]*\bid\s*=\s*"vid"[^>]*)',
            r"\1" + _src + r"\3",
            html,
            count=1,
        )
    if not _n_vid:
        print(
            "Warning: could not set <video id=vid> src to /v/<token>; "
            "browser may load a relative .mp4 from serve_dir (wrong clip). "
            f"file={html_path}",
            flush=True,
        )

    # ── 0. Hide top hint bar ─────────────────────────────────────────
    html = html.replace(
        '<div id="hint">',
        '<div id="hint" style="display:none">',
        1,
    )

    # ── 0b. Disable dragSeek in original handler (force pan-only, no seek on drag) ──
    # Even if cloneNode fails to remove the original mousemove, dragSeek=0 prevents seeking.
    html = html.replace(
        'dragSeek=(typeof vid !== \'undefined\' && vid)?1:0;',
        'dragSeek=0; // disabled: pan-only mode',
        1,
    )

    # ── 1. Default playback rate ──────────────────────────────────────
    old_init = "var vid=document.getElementById('vid');"
    new_init = (
        "var vid=document.getElementById('vid');\n"
        f"if(vid){{ vid.playbackRate={default_speed}; }}"
    )
    html = html.replace(old_init, new_init, 1)

    # ── 2. Speed buttons ─────────────────────────────────────────────
    old_ctrl = '<div id="ctrl"><button id="btnPlay" type="button">Play</button>'
    # Time / Save time / Stage are in the static HTML so they appear even if later script errors.
    new_ctrl = (
        '<div id="ctrl" style="font-size:16px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;">'
        '<span id="stageTimeDisplay" style="color:#ccc;font-size:13px;font-weight:bold;min-width:3em;margin-right:6px;"></span>'
        '<button type="button" id="btnAnnotateTime" title="Last shot − earliest beep (stage time)" '
        'style="padding:6px 14px;background:#446;color:#fff;border:none;border-radius:5px;cursor:pointer;font-size:16px;">Time</button>'
        '<button type="button" id="btnAnnotateSave" '
        'style="padding:6px 18px;background:#c75;color:#fff;border:none;border-radius:5px;cursor:pointer;font-size:16px;font-weight:bold;">Save time</button>'
        '<button id="btnPlay" type="button" style="padding:6px 18px;font-size:16px;">Play</button>'
        '<button id="spd025" onclick="_setSpeed(0.25)" style="padding:6px 14px;font-size:16px;">0.25x</button>'
        f'<button id="spd050" onclick="_setSpeed(0.5)" style="padding:6px 14px;font-size:16px;font-weight:bold;outline:2px solid #2a7;">0.5x</button>'
        '<button id="spd100" onclick="_setSpeed(1.0)" style="padding:6px 14px;font-size:16px;">1x</button>'
        '<button id="btnOpenVideo" style="padding:7px 22px;background:#36a;color:#fff;border:none;border-radius:5px;'
        'cursor:pointer;font-size:17px;font-weight:bold;margin-left:12px;">Open Video</button>'
        '<span id="loadStatus" style="color:#555;font-size:14px;"></span>'
    )
    html = html.replace(old_ctrl, new_ctrl, 1)
    html = _inject_static_toolbar_html(html)
    if _TOOLBAR_STATIC_MARK not in html:
        print(
            f"Warning: could not inject annotate toolbar (Time / Save time) into HTML; "
            f"check #ctrl markup: {html_path}",
            flush=True,
        )

    # ── 2b. FP/FN audit toolbar (playback-linked overlay; never persisted to cali/beep txt)
    if 'id="btnAuditClear"' not in html:
        html = html.replace(
            '<div id="waveformToolbar"><button id="btnSave" type="button">保存校准</button>'
            '<button id="btnSaveBeep" type="button">保存 Beep</button>',
            '<div id="waveformToolbar">'
            '<button type="button" id="btnAuditClear" '
            'style="padding:4px 8px;font-size:12px;cursor:pointer;background:#eee;border:1px solid #ccc;'
            'border-radius:4px;">清空审计标记</button>'
            '<button type="button" id="btnAuditCopy" '
            'style="padding:4px 8px;font-size:12px;cursor:pointer;background:#eee;border:1px solid #ccc;'
            'border-radius:4px;">复制审计 JSON</button>'
            '<button id="btnSave" type="button">保存校准</button>'
            '<button id="btnSaveBeep" type="button">保存 Beep</button>',
            1,
        )

    # ── 3. Toast style (no JS yet — Save All button added after cloneNode in patch 5) ──
    html = html.replace(
        "</head>",
        """<meta http-equiv="Cache-Control" content="no-store, must-revalidate" />
<style>
#saveToast{position:fixed;top:18px;right:18px;padding:12px 22px;border-radius:8px;
  font-size:14px;font-weight:bold;color:#fff;z-index:9999;display:none;
  box-shadow:0 3px 12px rgba(0,0,0,0.25);}
/* Whole toolbar uses full width so Time / Save time on the left are not clipped when #videoBox centers a wide row. */
#videoBox{overflow:visible!important;}
#ctrl{position:relative;z-index:6;align-self:stretch;width:100%;max-width:100%;box-sizing:border-box;justify-content:flex-start;padding:4px 8px;overflow-x:auto;flex-wrap:wrap;}
</style>
<div id="saveToast"></div>
</head>""",
        1,
    )

    # ── 4. Open Video JS (button already injected into #ctrl in step 2) ────
    _spd = str(default_speed)
    open_js = (
        '\n<script>'
        '\n(function(){'
        '\n  var btn=document.getElementById("btnOpenVideo");'
        '\n  var st=document.getElementById("loadStatus");'
        '\n  if(!btn) return;'
        '\n  btn.addEventListener("click",function(e){'
        '\n    e.stopPropagation();'
        '\n    btn.disabled=true; st.textContent="Opening...";'
        '\n    fetch("/open_file_dialog")'
        '\n    .then(function(r){return r.json();})'
        '\n    .then(function(res){'
        '\n      var p=res.path; if(!p){btn.disabled=false;st.textContent="Cancelled.";return;}'
        '\n      st.textContent="Loading "+p.split(/[\\\\/]/).pop()+"...";'
        '\n      return fetch("/load_video",{method:"POST",'
        '\n        headers:{"Content-Type":"application/json"},'
        '\n        body:JSON.stringify({path:p})})'
        '\n      .then(function(r){return r.json();})'
        '\n      .then(function(resp){'
        '\n        btn.disabled=false;'
        '\n        if(!resp.ok){st.textContent="Error: "+(resp.error||"?");return;}'
        '\n        var d=resp.data;'
        '\n        D=d; t0=0; t1=D.duration;'
        '\n        calibrationShots=(D.calibration_shots||[]).slice();'
        '\n        calibrationBeepTimes=(D.beep_times||[]).slice();'
        '\n        var v=document.getElementById("vid");'
        '\n        if(resp.stream_token && v){'
        '\n          v.pause();'
        '\n          try{v.removeAttribute("src");v.load();}catch(_e0){}'
        '\n          v.src="/v/"+resp.stream_token+"?bust="+Date.now();'
        '\n          try{v.load();}catch(_e1){}'
        f'\n          v.playbackRate={_spd};playheadT=0;'
        '\n          if(typeof vid!=="undefined"){ try{ vid=v; }catch(_v){} }'
        '\n        }else if(v){'
        '\n          st.textContent="Error: server did not send stream_token — check console [annotate_html]";'
        '\n        }'
        '\n        if(window._reloadAuditMarksFromData) window._reloadAuditMarksFromData(d);'
        '\n        draw();'
        '\n        if(window._updateSavePathDisplay) window._updateSavePathDisplay();'
        '\n        st.textContent="Loaded: "+(d.video_base_name||p.split(/[\\\\/]/).pop());'
        '\n      });'
        '\n    })'
        '\n    .catch(function(e){btn.disabled=false;st.textContent="Error: "+e;});'
        '\n  });'
        '\n})();'
        '\n</script>\n'
    )
    html = html.replace("</body>", open_js + "</body>", 1)

    # ── 5. Mouse handlers patch ───────────────────────────────────────
    patch_js = """
<script>
(function(){
  // If <video> still uses a bare filename (e.g. clip.mp4), the browser requests /clip.mp4 from serve_dir
  // (startup folder) — wrong clip. Force the session stream.
  var _vfix=document.getElementById('vid');
  if(_vfix){
    try{
      var _raw=_vfix.getAttribute('src')||'';
      var _pp=_raw;
      try{ if(/^https?:\\/\\//.test(_raw)||_raw.indexOf('//')===0){ _pp=new URL(_raw,location.href).pathname; } }catch(_e){}
      var sessOk=(_pp.indexOf('/v/')===0||_pp.indexOf('/video')===0);
      if(!sessOk){
        // Never assign /video here — same-path media cache can replay the first opened clip (ken0102).
        try{
          location.replace(location.pathname.split('?')[0]+'?rb='+Date.now());
        }catch(_e2){ _vfix.src='/video?enforce='+Date.now().toString(36)+Math.random().toString(36).slice(2,10); _vfix.load(); }
      }
    }catch(_e){}
  }
  // ---- Speed control ----
  window._setSpeed = function(s){
    if(vid){ vid.playbackRate=s; }
    ['spd025','spd050','spd100'].forEach(function(id){
      var b=document.getElementById(id);
      if(!b) return;
      var v=parseFloat(b.id.replace('spd',''))/100;
      // spd025=0.25, spd050=0.5, spd100=1.0
      var map={'spd025':0.25,'spd050':0.5,'spd100':1.0};
      if(map[id]===s){ b.style.outline='2px solid #2a7'; b.style.fontWeight='bold'; }
      else { b.style.outline=''; b.style.fontWeight=''; }
    });
  };

  // ---- Clone wrap to drop all existing mouse listeners ----
  var oldWrap = document.getElementById('wrap');
  var newWrap = oldWrap.cloneNode(true);
  oldWrap.parentNode.replaceChild(newWrap, oldWrap);
  // Re-bind globals that pointed at the old element
  wrap = newWrap;
  c    = document.getElementById('c');

  // ---- FP/FN audit vertical marks (listen-only overlay; Ctrl+右键 beep unaffected) ----
  var auditMarksFp = [];
  var auditMarksFn = [];
  (function initAuditFromD(){
    var a=(D&&D.audit_marks_fp)||[];
    var b=(D&&D.audit_marks_fn)||[];
    auditMarksFp = a.slice?a.slice().map(Number):[];
    auditMarksFn = b.slice?b.slice().map(Number):[];
    auditMarksFp.sort(function(x,y){return x-y;});
    auditMarksFn.sort(function(x,y){return x-y;});
  })();
  function reloadAuditMarksFromD(dd){
    var ap=(dd&&dd.audit_marks_fp)||[];
    var an=(dd&&dd.audit_marks_fn)||[];
    auditMarksFp = ap.slice?ap.slice().map(Number):[];
    auditMarksFn = an.slice?an.slice().map(Number):[];
    auditMarksFp.sort(function(x,y){return x-y;});
    auditMarksFn.sort(function(x,y){return x-y;});
  }
  window._reloadAuditMarksFromData = reloadAuditMarksFromD;

  var AUD_TOL=0.03;
  function dedupeCalibShotsTol(){
    calibrationShots.sort(function(a,b){return a-b;});
    var out=[];
    var i,x;
    for(i=0;i<calibrationShots.length;i++){
      x=calibrationShots[i];
      if(out.length && Math.abs(x-out[out.length-1])<AUD_TOL) continue;
      out.push(x);
    }
    calibrationShots=out;
  }
  function removeCalibNearest(tRm){
    for(var ci=calibrationShots.length-1;ci>=0;ci--){
      if(Math.abs(calibrationShots[ci]-tRm)<AUD_TOL){
        calibrationShots.splice(ci,1);
        return true;
      }
    }
    return false;
  }
  function syncFnAuditAfterShotDrag(prevT,newT){
    for(var ai=0;ai<auditMarksFn.length;ai++){
      if(Math.abs(auditMarksFn[ai]-prevT)<AUD_TOL){
        auditMarksFn[ai]=newT;
        auditMarksFn.sort(function(a,b){return a-b;});
        return;
      }
    }
  }
  function toggleFpAuditSeconds(ts){
    for(var i=0;i<auditMarksFp.length;i++){
      if(Math.abs(auditMarksFp[i]-ts)<AUD_TOL){
        auditMarksFp.splice(i,1);
        return;
      }
    }
    auditMarksFp.push(ts);
    auditMarksFp.sort(function(a,b){return a-b;});
  }
  function toggleFnAuditSeconds(ts){
    var i, oldFn;
    for(i=0;i<auditMarksFn.length;i++){
      oldFn=auditMarksFn[i];
      if(Math.abs(oldFn-ts)<AUD_TOL){
        auditMarksFn.splice(i,1);
        removeCalibNearest(oldFn);
        return;
      }
    }
    auditMarksFn.push(ts);
    calibrationShots.push(ts);
    auditMarksFn.sort(function(a,b){return a-b;});
    dedupeCalibShotsTol();
  }
  function purgeAuditMarkersNearRemovedShot(tRm){
    for(var i=auditMarksFn.length-1;i>=0;i--){ if(Math.abs(auditMarksFn[i]-tRm)<AUD_TOL) auditMarksFn.splice(i,1); }
    for(var j=auditMarksFp.length-1;j>=0;j--){ if(Math.abs(auditMarksFp[j]-tRm)<AUD_TOL) auditMarksFp.splice(j,1); }
  }

  var __drawWaveBeforeAudit = draw;
  function drawAuditMarksOverlay(){
    var ctx = c.getContext('2d');
    var W=c.width, H=c.height;
    var plotW = W-2*pad;
    if(plotW<1 || t1<=t0) return;
    function vlineList(arr,color){
      ctx.save();
      ctx.strokeStyle=color;
      ctx.lineWidth=2;
      ctx.setLineDash([5,6]);
      for(var i=0;i<arr.length;i++){
        var tm=arr[i];
        if(tm>=t0&&tm<=t1){
          var px=pad+(tm-t0)/(t1-t0)*plotW;
          ctx.beginPath();
          ctx.moveTo(px,pad);
          ctx.lineTo(px,H-pad);
          ctx.stroke();
        }
      }
      ctx.restore();
    }
    vlineList(auditMarksFp,'#aa00aa');
    vlineList(auditMarksFn,'#e07020');
    var lx=W-pad-266, ly2=pad+6+54;
    ctx.save();
    ctx.fillStyle='rgba(255,255,255,0.93)';
    ctx.strokeStyle='#888';
    ctx.setLineDash([]);
    ctx.lineWidth=1;
    ctx.beginPath();
    ctx.rect(lx,ly2,268,48);
    ctx.fill();
    ctx.stroke();
    ctx.font='13px sans-serif';
    ctx.fillStyle='#703070';
    ctx.fillText('Shift+单击 紫虚=FP审计→*fp.txt',lx+10,ly2+16);
    ctx.fillStyle='#a05010';
    ctx.fillText('Alt+单击 橙=FN→*fn.txt并进cali',lx+10,ly2+32);
    ctx.fillStyle='#444';
    ctx.fillText('再点±30ms可撤',lx+10,ly2+44);
    ctx.restore();
  }
  draw = function(){ __drawWaveBeforeAudit(); drawAuditMarksOverlay(); };

  // ---- Re-attach wheel (zoom/pan) — same logic as original ----
  wrap.addEventListener('wheel', function(e){
    e.preventDefault();
    var rect=wrap.getBoundingClientRect();
    var plotW=c.width-2*pad;
    if(plotW<1) return;
    if(e.ctrlKey){
      var k=e.deltaY>0?0.85:1/0.85;
      var mx=e.clientX-rect.left;
      var frac=((mx-pad)/plotW)||0.5;
      frac=Math.max(0,Math.min(1,frac));
      var tmid=t0+(t1-t0)*frac;
      var dt=t1-t0;
      var dtNew=Math.max(D.duration/5000, Math.min(D.duration, dt*k));
      t0=Math.max(0,Math.min(D.duration-dtNew, tmid-dtNew/2));
      t1=t0+dtNew;
    } else {
      var dx=(e.deltaY||e.deltaX)*0.15*(t1-t0)/10;
      var dt=t1-t0;
      t0=Math.max(0,Math.min(D.duration-dt, t0+dx));
      t1=t0+dt;
    }
    draw();
  }, {passive:false});

  // Playhead is display-only; dragging it is disabled
  function xToT(rx){
    var plotW=c.width-2*pad;
    if(rx<pad||rx>pad+plotW) return null;
    var t=t0+(rx-pad)/plotW*(t1-t0);
    return Math.max(0,Math.min(D.duration,t));
  }
  function seekAndPlay(t){
    if(!vid) return;
    var vst=D.video_start_time||0;
    vid.currentTime=t+vst;
    playheadT=t+vst;
    if(timeStr) timeStr.textContent=vid.currentTime.toFixed(2)+' / '+(vid.duration||0).toFixed(2)+' s';
    var p=vid.play();
    if(p && typeof p.catch==='function') p.catch(function(){
      // If interrupted (e.g. video not ready yet), retry once after seeked
      var onSeeked=function(){
        vid.removeEventListener('seeked',onSeeked);
        var p2=vid.play();
        if(p2 && typeof p2.catch==='function') p2.catch(function(){});
      };
      vid.addEventListener('seeked',onSeeked);
    });
    if(btnPlay) btnPlay.textContent='Pause';
    draw();
  }
  function seekOnly(t){
    var vst=D.video_start_time||0;
    vid.currentTime=t+vst;
    playheadT=t+vst;
    if(timeStr) timeStr.textContent=vid.currentTime.toFixed(2)+' / '+(vid.duration||0).toFixed(2)+' s';
    draw();
  }

  // ---- Mouse state ----
  var _mode=null;   // 'playhead'|'line'|'beep'|'pan'|null
  var _lineIdx=-1, _beepIdx=-1;
  var _startX=0, _moved=false;
  var THRESH=2;

  // ---- mousedown ----
  wrap.addEventListener('mousedown', function(e){
    if(e.button!==0) return;
    // Ignore clicks that originate inside toolbars (video #ctrl or waveform toolbar)
    var vc=document.getElementById('ctrl');
    if(vc && vc.contains(e.target)) return;
    var tb=document.getElementById('waveformToolbar');
    if(tb && tb.contains(e.target)) return;
    _startX=e.clientX; _moved=false;
    var rect=c.getBoundingClientRect();
    var rx=e.clientX-rect.left;
    // hit test shot line
    var plotW=c.width-2*pad;
    if(plotW>0 && rx>=pad && rx<=pad+plotW){
      for(var i=0;i<calibrationShots.length;i++){
        var px=pad+(calibrationShots[i]-t0)/(t1-t0)*plotW;
        if(Math.abs(rx-px)<=6){ _mode='line'; _lineIdx=i; return; }
      }
      for(var i=0;i<calibrationBeepTimes.length;i++){
        var px=pad+(calibrationBeepTimes[i]-t0)/(t1-t0)*plotW;
        if(Math.abs(rx-px)<=6){ _mode='beep'; _beepIdx=i; return; }
      }
    }
    _mode='pan';
  });

  // ---- mousemove ----
  document.addEventListener('mousemove', function(e){
    if(!_mode) return;
    if(Math.abs(e.clientX-_startX)>THRESH) _moved=true;
    var rect=c.getBoundingClientRect();
    var rx=e.clientX-rect.left;

    if(_mode==='playhead' && vid){
      var t=xToT(rx); if(t!==null) seekOnly(t);
      return;
    }
    if(_mode==='line'){
      if(_moved){
        var tln=xToT(rx);
        if(tln!==null){
          var prevT=calibrationShots[_lineIdx];
          calibrationShots[_lineIdx]=tln;
          syncFnAuditAfterShotDrag(prevT,tln);
          dedupeCalibShotsTol();
          draw();
        }
      }
      return;
    }
    if(_mode==='beep'){
      if(_moved){ var t=xToT(rx); if(t!==null){ calibrationBeepTimes[_beepIdx]=t; calibrationBeepTimes.sort(function(a,b){return a-b;}); draw(); } }
      return;
    }
    if(_mode==='pan' && _moved){
      var dt=t1-t0;
        var dx=(e.clientX-_startX)/Math.max(1,c.width)*dt;
        _startX=e.clientX;
        t0=Math.max(0,Math.min(D.duration-dt, t0-dx));
      t1=t0+dt;
      draw();
    }
  });

  // ---- mouseup ----
  document.addEventListener('mouseup', function(e){
    if(e.button!==0){ _mode=null; return; }
    var rect=c.getBoundingClientRect();
    var rx=e.clientX-rect.left;

    if(_mode==='line'){
      _lineIdx=-1;
    } else if(_mode==='beep'){
      _beepIdx=-1;
    } else if(_mode==='pan' && !_moved && vid){
      var t=xToT(rx);
      if(t!==null){
        if(e.shiftKey){
          toggleFpAuditSeconds(t);
          draw();
        }else if(e.altKey){
          toggleFnAuditSeconds(t);
          draw();
        }else seekAndPlay(t);
      }
    }
    _mode=null;
  });

  // ---- mouseleave ----
  wrap.addEventListener('mouseleave', function(){
    if(_mode==='line') _lineIdx=-1;
    if(_mode==='beep') _beepIdx=-1;
    _mode=null;
  });

  // ---- Right-click: add/delete shot or beep (unchanged logic) ----
  wrap.addEventListener('contextmenu', function(e){
    e.preventDefault();
    var rect=c.getBoundingClientRect();
    var rx=e.clientX-rect.left;
    var plotW=c.width-2*pad;
    if(plotW<1||rx<pad||rx>pad+plotW) return;
    if(e.ctrlKey){
      // beep add/delete
      for(var i=0;i<calibrationBeepTimes.length;i++){
        var px=pad+(calibrationBeepTimes[i]-t0)/(t1-t0)*plotW;
        if(Math.abs(rx-px)<=10){ calibrationBeepTimes.splice(i,1); draw(); return; }
      }
      var t=xToT(rx); if(t!==null){ calibrationBeepTimes.push(t); calibrationBeepTimes.sort(function(a,b){return a-b;}); draw(); }
      return;
    }
    // shot add/delete
    for(var i=0;i<calibrationShots.length;i++){
      var px=pad+(calibrationShots[i]-t0)/(t1-t0)*plotW;
      if(Math.abs(rx-px)<=10){
        var gone=calibrationShots[i];
        calibrationShots.splice(i,1);
        purgeAuditMarkersNearRemovedShot(gone);
        draw();
        return;
      }
    }
    var t=xToT(rx); if(t!==null){ calibrationShots.push(t); calibrationShots.sort(function(a,b){return a-b;}); draw(); }
  });

  // ---- Save / Time (buttons are in patched HTML #ctrl; wire handlers here) ----
  (function(){
    function showToast(msg, ok){
      var t=document.getElementById('saveToast');
      if(!t) return;
      t.textContent=msg;
      t.style.background=ok?'#2a7':'#c44';
      t.style.display='block';
      clearTimeout(t._tid);
      t._tid=setTimeout(function(){ t.style.display='none'; }, 3000);
    }
    window._showToast = showToast;

    var btnClr=document.getElementById('btnAuditClear');
    if(btnClr){
      btnClr.addEventListener('mousedown',function(ev){ev.stopPropagation();});
      btnClr.addEventListener('click',function(ev){
        ev.preventDefault(); ev.stopPropagation();
        var j, fnSnap=auditMarksFn.slice();
        for(j=0;j<fnSnap.length;j++) removeCalibNearest(fnSnap[j]);
        auditMarksFn.length=0;
        auditMarksFp.length=0;
        dedupeCalibShotsTol();
        draw(); showToast('已清空审计标记 (+FN 并进 cali)', true);
      });
    }
    var btnAj=document.getElementById('btnAuditCopy');
    if(btnAj){
      btnAj.addEventListener('mousedown',function(ev){ev.stopPropagation();});
      btnAj.addEventListener('click',function(ev){
        ev.preventDefault(); ev.stopPropagation();
        var s=JSON.stringify({fp:auditMarksFp.slice(),fn:auditMarksFn.slice()});
        if(navigator.clipboard && navigator.clipboard.writeText){
          navigator.clipboard.writeText(s).then(function(){ showToast('已复制审计 JSON', true); })
            .catch(function(){ window.prompt('审计 JSON:', s); });
        }else window.prompt('审计 JSON:', s);
      });
    }

    function doSave(){
      var caliPath=D.calibration_save_path;
      var beepPath=D.calibration_beep_path;
      var fpPath=D.calibration_fp_path;
      var fnPath=D.calibration_fn_path;
      var shots=calibrationShots.slice().sort(function(a,b){return a-b;});
      var beeps=calibrationBeepTimes.slice().sort(function(a,b){return a-b;});
      var fpa=auditMarksFp.slice().sort(function(a,b){return a-b;});
      var fna=auditMarksFn.slice().sort(function(a,b){return a-b;});
      var caliText=shots.map(function(t){return t.toFixed(4);}).join('\\n');
      var beepText=beeps.map(function(t){return t.toFixed(4);}).join('\\n');
      var fpTxt=fpa.map(function(t){return t.toFixed(4);}).join('\\n');
      var fnTxt=fna.map(function(t){return t.toFixed(4);}).join('\\n');
      var saved=0, total=0;
      function done(ok,msg){ saved++; if(saved===total){ showToast(ok?'Saved!':('Error: '+msg), ok); } }
      function postTxt(p, txt){
        if(!p){ return; }
        total++;
        fetch('/save_calibration',{method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({path:p,content:txt})})
        .then(function(r){return r.json();})
        .then(function(o){ done(o.ok, o.error||''); })
        .catch(function(e){ done(false,''+e); });
      }
      postTxt(caliPath, caliText);
      postTxt(beepPath, beepText);
      postTxt(fpPath, fpTxt);
      postTxt(fnPath, fnTxt);
      if(total===0){ showToast('No save path set.', false); }
    }
    window._doSave = doSave;

    // Hide the original two save buttons
    var bs=document.getElementById('btnSave');
    var bb=document.getElementById('btnSaveBeep');
    if(bs) bs.style.display='none';
    if(bb) bb.style.display='none';

    // Stage / Time / Save time: elements are already in patched #ctrl HTML; only wire handlers here.
    function computeStageSeconds(){
      var shots=(typeof calibrationShots!=='undefined'?calibrationShots:[]).slice().sort(function(a,b){return a-b;});
      var beeps=(typeof calibrationBeepTimes!=='undefined'?calibrationBeepTimes:[]).slice().sort(function(a,b){return a-b;});
      if(!shots.length || !beeps.length) return null;
      return shots[shots.length-1] - beeps[0];
    }
    window._refreshStageTimeDisplay=function(){
      var el=document.getElementById('stageTimeDisplay');
      if(!el) return;
      var s=computeStageSeconds();
      if(s===null){
        el.textContent='Stage: —';
        return;
      }
      el.textContent='Stage: '+s.toFixed(3)+' s';
    };

    var btnTime=document.getElementById('btnAnnotateTime');
    if(btnTime){
      btnTime.addEventListener('mousedown',function(e){e.stopPropagation();});
      btnTime.addEventListener('mouseup',  function(e){e.stopPropagation();});
      btnTime.addEventListener('click',function(e){
        e.preventDefault(); e.stopPropagation();
        window._refreshStageTimeDisplay();
      });
    }
    var btnAll=document.getElementById('btnAnnotateSave');
    if(btnAll){
      btnAll.addEventListener('mousedown',function(e){e.stopPropagation();});
      btnAll.addEventListener('mouseup',  function(e){e.stopPropagation();});
      btnAll.addEventListener('click',function(e){
        e.preventDefault(); e.stopPropagation();
        var msg='Save annotations?\\n'+
          (D.calibration_save_path?'  shots -> '+D.calibration_save_path+'\\n':'')+
          (D.calibration_beep_path?'  beep -> '+D.calibration_beep_path+'\\n':'')+
          (D.calibration_fp_path?'  fp   -> '+D.calibration_fp_path+'\\n':'')+
          (D.calibration_fn_path?'  fn   -> '+D.calibration_fn_path:'');
        if(!confirm(msg)) return;
        doSave();
      });
    }

    // Show current save paths (call this after D is updated too)
    window._updateSavePathDisplay = function(){
      function splitPath(p){
        if(!p) return [];
        var i=Math.max(p.lastIndexOf('/'), p.lastIndexOf(String.fromCharCode(92)));
        return i<0 ? ['',p] : [p.slice(0,i+1), p.slice(i+1)];
      }
      var sp=splitPath(D.calibration_save_path||D.calibration_beep_path||D.calibration_fp_path||D.calibration_fn_path||'');
      var dir=sp[0]||'';
      var el=document.getElementById('savePathDisplay');
      if(el){
        var parts=[];
        if(D.calibration_save_path) parts.push('shots: '+splitPath(D.calibration_save_path)[1]);
        if(D.calibration_beep_path) parts.push('beep: '+splitPath(D.calibration_beep_path)[1]);
        if(D.calibration_fp_path) parts.push('fp: '+splitPath(D.calibration_fp_path)[1]);
        if(D.calibration_fn_path) parts.push('fn: '+splitPath(D.calibration_fn_path)[1]);
        el.textContent=parts.join('   |   ');
      }
      var hint=document.getElementById('saveDirHint');
      if(hint){
        var rowV=D.annotation_video_path?('Video: '+D.annotation_video_path):'';
        var row2=(D.calibration_save_path?'shots: '+splitPath(D.calibration_save_path)[1]:'')+
                 (D.calibration_beep_path?'   beep: '+splitPath(D.calibration_beep_path)[1]:'')+
                 (D.calibration_fp_path?'   fp: '+splitPath(D.calibration_fp_path)[1]:'')+
                 (D.calibration_fn_path?'   fn: '+splitPath(D.calibration_fn_path)[1]:'');
        var rowSave=dir?('Save dir: '+dir+'  |  '+row2):row2;
        hint.textContent=(rowV?(rowV+'  |  '):'')+(rowSave||rowV||'');
      }
    };
    window._updateSavePathDisplay();
  })();

  draw();
})();
</script>
"""
    html = html.replace("</body>", patch_js + "\n</body>", 1)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def _find_annotation_dir(video_base, start_dir):
    """Search start_dir tree for existing annotation files matching video_base.
    Returns the directory containing them, or start_dir if not found.
    """
    beep_key = video_base.split("-")[0] if "-" in video_base else video_base
    project_root = os.path.dirname(os.path.abspath(__file__))
    search_roots = [start_dir, project_root]
    visited = set()
    for root_dir in search_roots:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Skip tmp and hidden directories
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "tmp"]
            key = os.path.normcase(os.path.abspath(dirpath))
            if key in visited:
                continue
            visited.add(key)
            if (video_base + "cali.txt") in filenames or (beep_key + "beep.txt") in filenames:
                return dirpath
    return start_dir


def _load_audit_markers_seed(path):
    """Load optional JSON {\"fp\":[sec,...], \"fn\":[...]} for annotate waveform overlays."""
    p = os.path.abspath(os.path.normpath(path))
    if not os.path.isfile(p):
        raise FileNotFoundError(f"audit JSON not found: {p}")
    with open(p, encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("audit JSON must be a JSON object")

    def _flt_arr(raw):
        if raw is None:
            return []
        if not isinstance(raw, (list, tuple)):
            raise ValueError("audit times must be a JSON array")
        return [float(x) for x in raw]

    fp = _flt_arr(obj.get("fp") or obj.get("audit_marks_fp"))
    fn = _flt_arr(obj.get("fn") or obj.get("audit_marks_fn"))
    if not fp and not fn and isinstance(obj.get("times"), list):
        fp = [float(x) for x in obj["times"]]
    return {"fp": sorted(fp), "fn": sorted(fn)}


def _prepare_video(
    video_path,
    ffmpeg,
    force_detect=False,
    session_shot_backend="auto",
    annotate_loose_min_conf=False,
    annotate_shot_nms_sec=0.1,
    audit_marker_seed=None,
):
    """Extract audio, load or detect annotations, build waveform data.
    Returns (cal_data_dict, video_dir, video_base, anno_dir).
    Always looks for annotation files in the video's own folder only.
    """
    video = os.path.abspath(os.path.normpath(video_path))
    if not os.path.isfile(video):
        raise FileNotFoundError(f"Video not found: {video}")
    video_dir  = os.path.dirname(video)
    video_base = os.path.splitext(os.path.basename(video))[0]

    print(f"Processing: {video}")
    print("Extracting audio...")
    audio_mono   = extract_audio_ch(video, ffmpeg, "tmp/audio_ann_mono.wav",   channels=1)
    print("  mono done")
    audio_stereo = extract_audio_ch(video, ffmpeg, "tmp/audio_ann_stereo.wav", channels=2)
    print("  stereo done")
    print("Running ffprobe...")
    fps, _ = ffprobe_info(video)
    print(f"  fps={fps}")

    beep_times, shot_times = [], []
    # Always look for annotations in the video's own folder only.
    anno_dir  = video_dir
    cali_path = os.path.join(anno_dir, video_base + "cali.txt")
    # Save target always uses full video name (e.g. S8-mainbeep.txt)
    beep_path = os.path.join(anno_dir, video_base + "beep.txt")
    # For loading, also check legacy prefix name (e.g. S8beep.txt) as fallback
    beep_key        = video_base.split("-")[0] if "-" in video_base else video_base
    beep_load_path  = beep_path  # default: full name
    beep_legacy     = os.path.join(anno_dir, beep_key + "beep.txt")
    if not os.path.isfile(beep_path) and os.path.isfile(beep_legacy):
        beep_load_path = beep_legacy  # load from legacy, but still save to full name

    print(f"  cali_path: {cali_path}  exists={os.path.isfile(cali_path)}")
    print(f"  beep_path (save): {beep_path}")
    print(f"  beep_load_path:   {beep_load_path}  exists={os.path.isfile(beep_load_path)}")

    has_cali = os.path.isfile(cali_path) and not force_detect
    has_beep = os.path.isfile(beep_load_path) and not force_detect

    # Load whatever exists; detect whatever is missing
    if has_cali:
        try:
            shot_times, _ = _parse_cali_txt_file(cali_path)
        except ValueError as e:
            print(f"  Error parsing *cali.txt (expected floats after optional # header): {e}")
            shot_times = []
        print(f"  Loaded *cali.txt: {len(shot_times)} shot(s)")
    if has_beep:
        with open(beep_load_path, encoding="utf-8") as f:
            beep_times = [float(l.strip()) for l in f if l.strip()]
        print(f"  Loaded *beep.txt: {beep_times}")

    if not has_beep:
        print("Detecting beep(s)...")
        try:
            beeps = detect_all_beeps(audio_mono, fps)
            beep_times = [float(b["t"]) for b in beeps]
            print(f"  Detected {len(beep_times)} beep(s): {beep_times}")
        except Exception as e:
            print(f"  Beep detection failed: {e}")

    if not has_cali:
        print("Detecting gunshots...")
        try:
            back, src = _resolve_shot_detector_backend(
                anno_dir, video_base, cali_path, session_shot_backend
            )
            print(f"  Shot detector backend: {back} ({src})")
            shots = _detect_shots_with_backend(
                audio_mono, fps, back, loose_min_confidence=annotate_loose_min_conf
            )
            if annotate_shot_nms_sec and annotate_shot_nms_sec > 0 and shots:
                n_nms_before = len(shots)
                shots = non_maximum_suppression(shots, time_threshold_s=float(annotate_shot_nms_sec))
                print(f"  Shot NMS ({annotate_shot_nms_sec:.3f}s): {len(shots)}/{n_nms_before}")
            shot_times = [float(s["t"]) for s in shots]
            print(f"  Detected {len(shot_times)} shot(s)")
        except Exception as e:
            print(f"  Shot detection failed: {e}")

    fp_txt_path = os.path.join(anno_dir, video_base + "fp.txt")
    fn_txt_path = os.path.join(anno_dir, video_base + "fn.txt")
    audit_fp_disk = _read_float_txt_lines(fp_txt_path)
    audit_fn_disk = _read_float_txt_lines(fn_txt_path)
    if audit_fp_disk:
        print(f"  Loaded *fp.txt ({fp_txt_path}): {audit_fp_disk}")
    if audit_fn_disk:
        print(f"  Loaded *fn.txt ({fn_txt_path}): {audit_fn_disk} (merge into calibration shots)")
    seed = audit_marker_seed or {}
    audit_fp_seed = list(seed.get("fp") or seed.get("audit_marks_fp") or [])
    audit_fn_seed = list(seed.get("fn") or seed.get("audit_marks_fn") or [])
    shot_times.extend(audit_fn_disk)
    shot_times.extend(audit_fn_seed)
    shot_times = _near_dedupe_sorted(shot_times, tol_s=0.03)
    audit_marks_fp = _near_dedupe_sorted(audit_fp_disk + audit_fp_seed, tol_s=0.03)
    audit_marks_fn = _near_dedupe_sorted(audit_fn_disk + audit_fn_seed, tol_s=0.03)

    print("Building waveform data...")
    wdata = get_waveform_data(
        audio_stereo, beep_times=beep_times, shot_times=shot_times, ref_shot_times=[]
    )
    print("Waveform done.")
    cal_data = dict(wdata)
    cal_data["calibration_shots"]     = list(shot_times)
    cal_data["video_base_name"]       = video_base
    cal_data["calibration_save_path"] = cali_path
    cal_data["calibration_beep_path"] = beep_path
    cal_data["calibration_fp_path"] = fp_txt_path
    cal_data["calibration_fn_path"] = fn_txt_path
    cal_data["annotation_video_path"] = video
    cal_data["audit_marks_fp"] = audit_marks_fp
    cal_data["audit_marks_fn"] = audit_marks_fn
    print(
        "[_prepare_video] embedding cal_data: "
        f"annotation_video_path={video!r} | D.duration={cal_data.get('duration')!r} | len(t)={len(cal_data.get('t') or [])}",
        flush=True,
    )
    return cal_data, video_dir, video_base, anno_dir


def _materialize_annotation_html(cal_data, video_path, default_speed=0.5, stream_token: str | None = None):
    """Build full annotate viewer HTML for current cal_data + video (avoid stale disk HTML on refresh)."""
    if not stream_token:
        raise ValueError("stream_token required (use /v/<token> URL so the browser cannot reuse a cached clip)")
    v_cal = cal_data.get("annotation_video_path")
    v_arg = os.path.abspath(os.path.normpath(video_path)) if video_path else None
    dur = cal_data.get("duration")
    n_t = len(cal_data.get("t") or [])
    match = (
        os.path.normcase(v_cal or "") == os.path.normcase(v_arg or "")
        if (v_cal and v_arg)
        else None
    )
    print(
        "[materialize_annotation_html] "
        f"waveform D: duration={dur!r} len(t)={n_t} | "
        f"D.annotation_video_path={v_cal!r} | "
        f"patch video_path arg={v_arg!r} | "
        f"paths_consistent={match} | "
        f"stream_token[..8]={stream_token[:8]}...",
        flush=True,
    )
    if v_cal and v_arg and match is False:
        print(
            "[materialize_annotation_html] WARNING: embedded cal_data does not match video_path argument — "
            "HTML waveform D may be for a different file than <video> after patch.",
            flush=True,
        )
    fd, tmp_path = tempfile.mkstemp(suffix=".html")
    os.close(fd)
    try:
        write_calibration_viewer_html(tmp_path, cal_data, video_path=video_path)
        _patch_html_playback_speed(
            tmp_path,
            default_speed=default_speed,
            video_path=video_path,
            stream_token=stream_token,
        )
        with open(tmp_path, encoding="utf-8") as f:
            return f.read()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def run_annotation_server(serve_dir, port, html_basename, ffmpeg_cmd,
                          initial_video_path=None, initial_save_dir=None,
                          initial_cal_data=None,
                          initial_annotate_html: str | None = None,
                          initial_stream_token: str | None = None,
                          picker_root: str | None = None,
                          default_speed=0.5, session_shot_backend="auto",
                          annotate_loose_min_conf=False, annotate_shot_nms_sec=0.1):
    """HTTP server for the annotation tool.

    Endpoints:
      GET  /<file>          — serve static files with Range support
      GET  /v/<token>       — serve the current session video (path-only URL avoids browser media cache mixups)
      GET  /video           — legacy stream for the currently loaded video with Range support
      POST /save_calibration — write annotation files
      POST /load_video      — process a new video and return updated waveform JSON + stream_token
    """
    serve_dir  = os.path.abspath(serve_dir)

    # Shared mutable state accessible by the request handler
    state = {
        "video_path": initial_video_path,
        "save_dir":   initial_save_dir or serve_dir,
        "session_shot_backend": session_shot_backend,
        "cal_data":   initial_cal_data,
        "annotate_html_bytes": None,
        "video_stream_token": None,
        # Pinned folder for Tk pickers (--picker-dir / SHOTMASK_PICKER_DIR).
        "picker_root": picker_root if (isinstance(picker_root, str) and os.path.isdir(picker_root)) else None,
        "annotate_loose_min_conf": annotate_loose_min_conf,
        "annotate_shot_nms_sec": float(annotate_shot_nms_sec),
    }

    def _rebuild_annotate_html_cache():
        """Served annotate page must match state — otherwise F5 keeps an old embedded D while /video follows session."""
        cal = state.get("cal_data")
        vp = state.get("video_path")
        if not cal or not vp or not isinstance(vp, str) or not os.path.isfile(vp):
            state["annotate_html_bytes"] = None
            state["video_stream_token"] = None
            return
        print(
            "\n--- annotate_html REBUILD --------------------------------------------------",
            flush=True,
        )
        print(
            "[annotate_html REBUILD] GET /v/<token> streams FILE:",
            vp,
            flush=True,
        )
        print(
            "[annotate_html REBUILD] embedded D.annotation_video_path (must match file used for waveform):",
            cal.get("annotation_video_path"),
            flush=True,
        )
        print(
            "[annotate_html REBUILD] D.duration, len(D.t):",
            repr(cal.get("duration")),
            len(cal.get("t") or []),
            flush=True,
        )
        # Token must survive HTML materialize failures: client needs GET /v/<token> ↔ state["video_path"].
        # If we clear the token, Open Video falls back to /video — some browsers still key media cache on
        # /video and replay the first clip (e.g. ken0102) even when the server sends a different file.
        state["video_stream_token"] = secrets.token_hex(16)
        try:
            html = _materialize_annotation_html(
                cal,
                vp,
                default_speed=default_speed,
                stream_token=state["video_stream_token"],
            )
            state["annotate_html_bytes"] = html.encode("utf-8")
        except Exception as e:
            print(f"[annotate_html] Rebuild failed: {e}")
            state["annotate_html_bytes"] = None

    if (
        initial_annotate_html is not None
        and initial_stream_token
        and isinstance(initial_stream_token, str)
    ):
        state["annotate_html_bytes"] = initial_annotate_html.encode("utf-8")
        state["video_stream_token"] = initial_stream_token
        print(
            "\n--- annotate_html initial (from main, no rebuild) --------------------------------",
            flush=True,
        )
        print(
            "[annotate_html] serving RAM page built in main(); stream token (prefix):",
            initial_stream_token[:16] + "...",
            flush=True,
        )
        print(
            "[annotate_html] session video (GET /v/… serves this):",
            initial_video_path,
            flush=True,
        )
        if initial_cal_data:
            print(
                "[annotate_html] D.annotation_video_path:",
                initial_cal_data.get("annotation_video_path"),
                flush=True,
            )
            print(
                "[annotate_html] D.duration, len(D.t):",
                repr(initial_cal_data.get("duration")),
                len(initial_cal_data.get("t") or []),
                flush=True,
            )
    else:
        _rebuild_annotate_html_cache()

    def _norm(p):
        return os.path.normcase(os.path.normpath(p))

    def _is_client_disconnect(err):
        """Browser closed tab / cancelled download — do not reply with HTTP 500 (socket already dead)."""
        if isinstance(err, (BrokenPipeError, ConnectionAbortedError, ConnectionResetError)):
            return True
        if isinstance(err, OSError):
            winerr = getattr(err, "winerror", None)
            if winerr in (10053, 10054):  # Windows: aborted / reset by peer
                return True
        return False

    def _serve_file(handler, file_path, cache_control=None):
        """Send file with HTTP Range support (needed for HTML5 video seeking)."""
        ext = os.path.splitext(file_path)[1].lower()
        ctype_map = {
            ".html": "text/html", ".htm": "text/html",
            ".txt": "text/plain",
            ".mp4": "video/mp4", ".webm": "video/webm",
            ".wav": "audio/wav", ".mp3": "audio/mpeg",
        }
        ctype = ctype_map.get(ext, "application/octet-stream")
        try:
            file_size = os.path.getsize(file_path)
            range_hdr = handler.headers.get("Range")
            if range_hdr:
                parts = range_hdr.replace("bytes=", "").split("-")
                start = int(parts[0]) if parts[0] else 0
                end   = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
                end   = min(end, file_size - 1)
                length = end - start + 1
                with open(file_path, "rb") as f:
                    f.seek(start)
                    data = f.read(length)
                handler.send_response(206)
                handler.send_header("Content-Type", ctype)
                handler.send_header("Content-Length", length)
                handler.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                handler.send_header("Accept-Ranges", "bytes")
                if cache_control:
                    handler.send_header("Cache-Control", cache_control)
                handler.end_headers()
                handler.wfile.write(data)
            else:
                with open(file_path, "rb") as f:
                    data = f.read()
                handler.send_response(200)
                handler.send_header("Content-Type", ctype)
                handler.send_header("Content-Length", file_size)
                handler.send_header("Accept-Ranges", "bytes")
                if cache_control:
                    handler.send_header("Cache-Control", cache_control)
                handler.end_headers()
                handler.wfile.write(data)
        except Exception as e:
            if _is_client_disconnect(e):
                return
            try:
                handler.send_error(500)
            except Exception:
                pass

    class AnnotationHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]

            # Special: open native file picker and return chosen path as JSON
            if path == "/open_file_dialog":
                def _pick_initialdir():
                    # Prefer the folder of the video already in this session so "Open Video" stays local
                    # (e.g. sibling clips). Picker pin (--picker-dir / SHOTMASK_PICKER_DIR) applies when there
                    # is no usable video dir yet — not on every reopen, otherwise dialogs always jump to training data.
                    v = state.get("video_path")
                    if isinstance(v, str) and os.path.isfile(v):
                        vd = os.path.dirname(os.path.abspath(v))
                        if os.path.isdir(vd):
                            return os.path.normpath(vd)
                    sd = state.get("save_dir")
                    if isinstance(sd, str) and os.path.isdir(sd):
                        return os.path.normpath(sd)
                    pr = state.get("picker_root")
                    if isinstance(pr, str) and os.path.isdir(pr):
                        return os.path.normpath(pr)
                    # Without initialdir, Windows often reuses the last browse path for python.exe/py.exe
                    # (feels like "it remembers my last Open Video folder" across runs).
                    return serve_dir if os.path.isdir(serve_dir) else os.getcwd()

                def _pick():
                    root = tk.Tk()
                    root.withdraw()
                    root.attributes("-topmost", True)
                    _tk_sync_before_dialog(root)
                    chosen = filedialog.askopenfilename(
                        title="Select video file",
                        filetypes=[
                            ("Video files", "*.mp4 *.mov *.avi *.mkv *.webm"),
                            ("All files", "*.*"),
                        ],
                        initialdir=os.path.normpath(_pick_initialdir()),
                    )
                    root.destroy()
                    return chosen
                chosen = _pick()
                data = json.dumps({"path": chosen or ""}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", len(data))
                self.end_headers()
                self.wfile.write(data)
                return

            # Session video stream: unique path per clip so the browser does not reuse another file's cached bytes.
            if path.startswith("/v/"):
                tok = path[len("/v/") :].split("/")[0]
                if not tok or ".." in tok:
                    self.send_error(400)
                    return
                if tok != state.get("video_stream_token"):
                    self.send_error(404)
                    return
                vp = state.get("video_path")
                if vp and os.path.isfile(vp):
                    # Browser <video> issues many GETs (Range chunks, buffering, seeks); do not log each one.
                    _serve_file(
                        self,
                        vp,
                        cache_control="no-store, max-age=0, must-revalidate",
                    )
                else:
                    self.send_error(404)
                return

            # Special: serve the currently loaded video
            if path == "/video":
                vp = state.get("video_path")
                if vp and os.path.isfile(vp):
                    _serve_file(
                        self,
                        vp,
                        cache_control="no-store, max-age=0, must-revalidate",
                    )
                else:
                    self.send_error(404)
                return

            path = path.replace("\\", "/").lstrip("/")
            if not path:
                path = html_basename or "index.html"

            if ".." in path:
                self.send_error(403)
                return

            # Do not serve raw .mp4 from serve_dir: relative <video src="file.mp4"> would pull the wrong file
            # (startup directory) instead of session state["video_path"] served at /video.
            if path.lower().endswith(".mp4"):
                msg = (
                    "Direct .mp4 URLs are disabled for this annotate server. "
                    "The <video> element must use src starting with /v/<token> (session) or /video."
                )
                b = msg.encode("utf-8")
                self.send_response(403)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
                return

            # Always serve annotate page from RAM so refresh matches current session (/video path + embedded D).
            if path == html_basename:
                blob = state.get("annotate_html_bytes")
                if not blob:
                    _rebuild_annotate_html_cache()
                    blob = state.get("annotate_html_bytes")
                if blob:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(blob)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(blob)
                    return
                # Do not fall back to serve_dir/annotate.html: that file is only written at process start
                # for the *first* video; reading it after Open Video would show the wrong embedded D while
                # /video still streams the current session file (path mismatch).
                msg = (
                    "<!DOCTYPE html><meta charset=utf-8><title>Annotate page unavailable</title>"
                    "<p>The annotate HTML could not be built for the current video (see server console for "
                    "<code>[annotate_html] Rebuild failed</code>).</p>"
                    "<p>Video in session: "
                    + html.escape(str(state.get("video_path") or "?"))
                    + "</p>"
                )
                body = msg.encode("utf-8")
                self.send_response(503)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            file_path = os.path.abspath(os.path.join(serve_dir, path))
            if not _norm(file_path).startswith(_norm(serve_dir)) or not os.path.isfile(file_path):
                self.send_error(404)
                return
            # Annotate page may fall back to disk when cache rebuild failed; never use a stale cached copy.
            cc = (
                "no-store, max-age=0, must-revalidate"
                if path == html_basename
                else None
            )
            _serve_file(self, file_path, cache_control=cc)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)

            def _json_resp(obj):
                data = json.dumps(obj).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", len(data))
                self.end_headers()
                self.wfile.write(data)

            if self.path == "/save_calibration":
                try:
                    req  = json.loads(body.decode("utf-8"))
                    path = os.path.abspath(os.path.normpath(req.get("path", "")))
                    content = req.get("content", "")
                    if path.lower().endswith("cali.txt"):
                        content = merge_cali_preamble_on_save(path, content)
                    # Allow saving to current video's dir or serve_dir
                    save_dir_n = _norm(state.get("save_dir", serve_dir))
                    if not _norm(path).startswith(save_dir_n):
                        _json_resp({"ok": False, "error": "path not under video dir"})
                        return
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(content)
                    _json_resp({"ok": True, "path": path})
                except Exception as e:
                    _json_resp({"ok": False, "error": str(e)})

            elif self.path == "/load_video":
                try:
                    req   = json.loads(body.decode("utf-8"))
                    vpath = req.get("path", "").strip()
                    if not vpath:
                        _json_resp({"ok": False, "error": "empty path"})
                        return
                    # Look for existing annotations in the video's own folder first;
                    # run detection only if none are found.
                    cal_data, video_dir, video_base, anno_dir = _prepare_video(
                        vpath,
                        ffmpeg_cmd,
                        force_detect=False,
                        session_shot_backend=state.get("session_shot_backend", "auto"),
                        annotate_loose_min_conf=bool(state.get("annotate_loose_min_conf", False)),
                        annotate_shot_nms_sec=float(state.get("annotate_shot_nms_sec", 0.1)),
                    )
                    state["video_path"] = os.path.abspath(os.path.normpath(vpath))
                    state["save_dir"]   = anno_dir  # save to annotation dir, not tmp/
                    state["cal_data"]   = cal_data
                    _rebuild_annotate_html_cache()
                    print(
                        "[load_video] response stream_token prefix:",
                        (state.get("video_stream_token") or "")[:16] + "..."
                        if state.get("video_stream_token")
                        else "(none — F5 page may 503)",
                        flush=True,
                    )
                    print(f"[load_video] Sending response for {video_base} ...")
                    try:
                        payload = json.dumps(
                            {
                                "ok": True,
                                "data": cal_data,
                                "stream_token": state.get("video_stream_token"),
                            }
                        ).encode("utf-8")
                    except Exception as je:
                        print(f"[load_video] JSON encode error: {je}")
                        raise
                    print(f"[load_video] Payload size: {len(payload)} bytes")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", len(payload))
                    self.end_headers()
                    self.wfile.write(payload)
                    print(f"[load_video] Done.")
                except Exception as e:
                    import traceback; traceback.print_exc()
                    try:
                        _json_resp({"ok": False, "error": str(e)})
                    except Exception:
                        pass

            else:
                self.send_error(404)

        def log_message(self, format, *args):
            pass  # suppress request logs

    session_nonce = secrets.token_hex(8)
    ts = int(time.time())
    url = f"http://127.0.0.1:{port}/{html_basename}?v={ts}&s={session_nonce}"
    print(f"Annotation server: {url}")
    print("  (opens in a NEW browser window; do not reuse an old tab / file:// annotate.html)")
    print("  (?v&s= reset each launch — stale pages were another annotate server or cached tab)")
    if initial_video_path:
        print(
            "Session video (F5 + /v/<token> + Open Video use this file):",
            initial_video_path,
        )

    def _open_browser_delayed():
        time.sleep(0.35)
        try:
            webbrowser.open(url, new=1, autoraise=True)
        except Exception as err:
            print(f"Could not auto-open browser: {err}")

    threading.Thread(target=_open_browser_delayed, daemon=True).start()

    try:
        server = HTTPServer(("127.0.0.1", port), AnnotationHandler)
    except OSError as err:
        print(
            "\n*** Cannot listen on http://127.0.0.1:{} — {}\n*** "
            "Another annotate server (older run) may still hold this port. "
            "Close its CMD/Python window or rerun start_annotate.bat "
            "(it tries to kill the port first).\n".format(port, err),
            flush=True,
        )
        raise SystemExit(1) from None
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def main():
    ap = argparse.ArgumentParser(
        description="Manual annotation tool: 0.5x playback, drag lines for shots/beep, save to *cali.txt"
    )
    ap.add_argument("--video", default=None, help="Video file path (.mp4). If omitted, a file picker opens.")
    ap.add_argument(
        "--pick",
        action="store_true",
        help="Always show the opening file picker, even if --video is set (shortcut/drag-drop). "
             "Env SHOTMASK_ALWAYS_PICK=1 does the same.",
    )
    ap.add_argument("--speed", type=float, default=0.5,
                    help="Default playback speed (default 0.5)")
    ap.add_argument("--port", type=int, default=8765,
                    help="Local server port (default 8765)")
    ap.add_argument("--no-detect", action="store_true",
                    help="Skip machine detection; load existing *cali.txt if available")
    ap.add_argument(
        "--shot-backend",
        default="auto",
        choices=["auto", "ast", "cnn"],
        help="Machine gunshot detect (no *cali.txt): ast|cnn or auto "
             "(follow calibrated_detector_params.json). "
             "Override per folder: VIDEO.shot_detector.txt or # shot_detector=cnn atop *cali.txt",
    )
    ap.add_argument("--out-dir", default=None,
                    help="HTML output directory (default: same as video)")
    ap.add_argument(
        "--picker-dir",
        default=None,
        metavar="DIR",
        help="Pin file dialogs (startup + Open Video) to this folder. Env SHOTMASK_PICKER_DIR if unset; CLI overrides env.",
    )
    ap.add_argument(
        "--annotate-loose",
        action="store_true",
        help="Gunshot hints: skip calibrated confidence threshold → many extras (legacy; use rarely).",
    )
    ap.add_argument(
        "--shot-nms",
        type=float,
        default=0.1,
        metavar="SEC",
        help="Merge detections within SEC s, keep highest confidence (default 0.1; 0=off).",
    )
    ap.add_argument(
        "--audit-json",
        default=None,
        metavar="PATH",
        help='Optional audit times JSON {\"fp\":[sec,...],\"fn\":[…]} merge with *fp.txt/*fn.txt ; fn-times also merged into calibration (*cali).',
    )
    args = ap.parse_args()

    if args.pick or os.environ.get("SHOTMASK_ALWAYS_PICK", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        args.video = None

    picker_root = _resolve_picker_root(args.picker_dir)
    if picker_root:
        print(f"File dialogs pinned to: {picker_root}", flush=True)

    print(f"Using annotate_shots.py: {os.path.abspath(__file__)}", flush=True)

    ffmpeg = get_ffmpeg_cmd()
    if not ffmpeg:
        print("Error: ffmpeg not found.")
        return 1

    # If no video given, open native file picker
    if not args.video:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        _start = picker_root or (
            _ANNOTATE_SHOTS_DIR if os.path.isdir(_ANNOTATE_SHOTS_DIR) else os.getcwd()
        )
        _tk_sync_before_dialog(root)
        chosen = filedialog.askopenfilename(
            title="Select video file to annotate",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.avi *.mkv *.webm"),
                ("All files", "*.*"),
            ],
            initialdir=os.path.normpath(_start),
        )
        root.destroy()
        if not chosen:
            print("No video selected, exiting.")
            return 0
        args.video = chosen

    video = os.path.abspath(os.path.normpath(args.video))
    if not os.path.isfile(video):
        print(f"Error: video not found: {video}")
        return 1

    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(video)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("tmp", exist_ok=True)

    audit_seed = None
    if args.audit_json:
        try:
            audit_seed = _load_audit_markers_seed(args.audit_json)
            print(f"Loaded audit markers from JSON: fp={len(audit_seed['fp'])} fn={len(audit_seed['fn'])}", flush=True)
        except Exception as ex:
            print(f"Error reading --audit-json: {ex}", flush=True)
            return 1

    # ── 1. Process initial video ──────────────────────────────────────
    cal_data, video_dir, video_base, anno_dir = _prepare_video(
        video,
        ffmpeg,
        force_detect=args.no_detect,
        session_shot_backend=args.shot_backend,
        annotate_loose_min_conf=args.annotate_loose,
        annotate_shot_nms_sec=args.shot_nms,
        audit_marker_seed=audit_seed,
    )

    # ── 2. Generate HTML ──────────────────────────────────────────────
    html_name = "annotate.html"
    html_path = os.path.join(out_dir, html_name)

    # Write HTML with <video src="/v/<token>"> so each clip has a unique URL (no stale ken0102 cache).
    stream_token = secrets.token_hex(16)
    html_page = _materialize_annotation_html(
        cal_data, video, default_speed=args.speed, stream_token=stream_token
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_page)
    # annotate.html lives next to this mp4 unless --out-dir (not a separate saved folder)
    print("\nVideo file (this path decides the output folder):", flush=True)
    print(f"  {video}", flush=True)
    print(f"\nHTML ready: {html_path}")
    print(f"Default playback speed: {args.speed}x")

    print(f"\nStarting annotation server on port {args.port}...")
    print("Press Ctrl+C to stop\n")
    run_annotation_server(
        serve_dir=out_dir,
        port=args.port,
        html_basename=html_name,
        ffmpeg_cmd=ffmpeg,
        initial_video_path=video,
        initial_save_dir=anno_dir,
        initial_cal_data=cal_data,
        initial_annotate_html=html_page,
        initial_stream_token=stream_token,
        default_speed=args.speed,
        session_shot_backend=args.shot_backend,
        annotate_loose_min_conf=args.annotate_loose,
        annotate_shot_nms_sec=args.shot_nms,
        picker_root=picker_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
