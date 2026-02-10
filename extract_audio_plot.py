"""
Subprogram: Extract audio from video and generate waveform plot
Usage: python extract_audio_plot.py --video "video_path.mp4" [--output "output_image_path.png"]
Calibrate detector on training set: python extract_audio_plot.py --video 1.mp4 --calibrate
"""
import html
import json
import os
import shutil
import subprocess
import argparse
import base64
import threading
import webbrowser
import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.optimize import linear_sum_assignment

from detectors.beep import detect_beeps
from detectors.shot_audio import (
    detect_shots,
    CALIBRATED_PARAMS_FILENAME,
    load_calibrated_params,
    compute_feature_at_time,
    detect_shots_improved,
    cluster_peaks_by_time,
)
from detectors.shot_logreg import train_logreg
from detectors.shot_motion import detect_shots_from_motion_improved, detect_shots_from_motion_roi_auto
from detectors.shot_energy import detect_shots_energy
from reference_splits import ref_shot_times
from ref_from_image import get_ref_times_for_video, get_ref_times_and_source


def run_calibration(audio_path, fps, ref_times, max_match_s=0.04):
    """Tune detector params on training set; save best to calibrated_detector_params.json.
    max_match_s: 匹配容差(s)。split 可达 0.09s，故用 0.04 (<0.09/2) 避免匹配过宽。
    枪声间隔不会小于 0.08s，故 min_dist_sec 只试 0.08 及以上。
    """
    params_list = []
    for d in [0.08, 0.10]:  # 间隔不小于 0.08s，不再用 0.06
        for p in [0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
            for k in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
                params_list.append({"threshold_coef": k, "min_dist_sec": d, "prominence_frac": p})
            for pct in [88, 90, 92, 95, 97]:
                params_list.append({"threshold_percentile": pct, "min_dist_sec": d, "prominence_frac": p})
    for smooth_ms in [4.0, 6.0, 8.0]:
        for d in [0.08]:  # 间隔≥0.08s
            for p in [0.2, 0.3]:
                for k in [2.5, 3.0, 3.5]:
                    params_list.append({"threshold_coef": k, "min_dist_sec": d, "prominence_frac": p, "smooth_ms": smooth_ms})
    n_ref = len(ref_times)
    # 综合得分：匹配数 - 过检惩罚，使检测数尽量接近 n_ref（避免 80+ 多检）
    over_penalty = 0.10  # 每多检 1 个扣 0.1，多 10 个则少算 1 个“有效匹配”，抑制 80+ 多检
    best_score, best_n, best_mae, best_ndet = -1.0, 0, float("inf"), 0
    best_params = None
    print(f"Calibrating with {len(params_list)} parameter combinations...")
    for i, params in enumerate(params_list):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(params_list)}")
        shots = detect_shots(audio_path, fps, use_calibrated=False, **params)
        det_t = [s["t"] for s in shots]
        n, mae = calibration_metrics(ref_times, det_t, max_match_s)
        ndet = len(det_t)
        # Penalty for over-detection (more strict)
        penalty = over_penalty * max(0, ndet - n_ref)
        # Bonus for under-detection (but less than over-detection penalty)
        bonus = 0.05 * max(0, n_ref - ndet)
        score = n - penalty + bonus
        # Prefer parameters that get close to n_ref detections
        if score > best_score or (abs(score - best_score) < 1e-9 and mae < best_mae):
            best_score, best_n, best_mae, best_ndet = score, n, mae, ndet
            best_params = dict(params)
    if best_params is None:
        print("Calibration: no valid params found.")
        return
    # Ensure post_filter is enabled if not explicitly set
    if "post_filter" not in best_params:
        best_params["post_filter"] = True
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CALIBRATED_PARAMS_FILENAME)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nCalibration done: best matched {best_n}/{n_ref} ref shots, detections = {best_ndet}, MAE = {best_mae:.3f}s")
    print(f"Saved params to {out_path}: {best_params}")
    print("New videos will use these params automatically.")


def calibration_metrics(ref_times, detected_times, max_match_s=0.12):
    """Match ref to detected in order (greedy). Returns (n_matched, mae). max_match_s=0.10–0.15 for tight localization."""
    ref = np.asarray(ref_times)
    det = np.asarray(detected_times)
    if len(det) == 0:
        return 0, float("inf")
    j_start = 0
    errs = []
    for i, rt in enumerate(ref):
        best_j, best_d = None, np.inf
        for j in range(j_start, len(det)):
            d = abs(det[j] - rt)
            if d < best_d:
                best_d, best_j = d, j
        if best_j is not None and best_d <= max_match_s:
            errs.append(det[best_j] - rt)
            j_start = best_j + 1
    if not errs:
        return 0, float("inf")
    return len(errs), float(np.mean(np.abs(np.array(errs))))


def calibration_metrics_hungarian(ref_times, detected_times, max_match_s=0.12):
    """Global optimal bipartite matching: maximize pairs with |t_det - t_ref| <= max_match_s. Returns (n_matched, mae)."""
    ref = np.asarray(ref_times)
    det = np.asarray(detected_times)
    n_ref, n_det = len(ref), len(det)
    if n_det == 0:
        return 0, float("inf")
    n = max(n_ref, n_det)
    BIG = 1e9
    C = np.zeros((n, n))
    for i in range(n_ref):
        for j in range(n_det):
            d = abs(det[j] - ref[i])
            C[i, j] = d if d <= max_match_s else BIG
    row_ind, col_ind = linear_sum_assignment(C)
    errs = []
    for i in range(n_ref):
        j = col_ind[i]
        if j < n_det and C[i, j] < BIG:
            errs.append(det[j] - ref[i])
    if not errs:
        return 0, float("inf")
    return len(errs), float(np.mean(np.abs(np.array(errs))))


def calibration_report(ref_times, detected_times, max_match_s=0.12):
    """Match ref to detected in order, print per-shot errors and summary. max_match_s=0.10–0.15 for tight localization."""
    ref = np.asarray(ref_times)
    det = np.asarray(detected_times)
    if len(det) == 0:
        print("Calibration: no detected shots to compare.")
        return
    j_start = 0
    paired_ref, paired_det, errs = [], [], []
    unmatched_info = []  # (ref_idx, ref_t, best_d, best_det_t or None)
    for i, rt in enumerate(ref):
        best_j, best_d = None, np.inf
        for j in range(j_start, len(det)):
            d = abs(det[j] - rt)
            if d < best_d:
                best_d, best_j = d, j
        if best_j is not None and best_d <= max_match_s:
            paired_ref.append(rt)
            paired_det.append(det[best_j])
            errs.append(det[best_j] - rt)
            j_start = best_j + 1
        else:
            unmatched_info.append((i, float(rt), float(best_d) if best_j is not None else np.nan, float(det[best_j]) if best_j is not None else None))
    n = len(errs)
    if n == 0:
        print("Calibration: no ref–detected pairs within {:.2f}s.".format(max_match_s))
        return
    err_arr = np.array(errs)
    mae = np.mean(np.abs(err_arr))
    rmse = np.sqrt(np.mean(err_arr ** 2))
    n_h, mae_h = calibration_metrics_hungarian(ref_times, detected_times, max_match_s)
    print("Calibration vs reference (ref = beep + splits):")
    print("  Greedy:  Matched {}/{} ref shots (max_match = {:.2f}s)".format(n, len(ref), max_match_s))
    print("  Hungarian: Matched {}/{} ref shots (global optimal bipartite)".format(n_h, len(ref)))
    print("  MAE = {:.3f}s  RMSE = {:.3f}s".format(mae, rmse))
    if n <= 12:
        for i in range(n):
            print("    shot {:2d}: ref {:.2f}s  det {:.2f}s  err {:+.3f}s".format(
                i + 1, paired_ref[i], paired_det[i], errs[i]))
    else:
        for i in [0, 1, 2]:
            print("    shot {:2d}: ref {:.2f}s  det {:.2f}s  err {:+.3f}s".format(
                i + 1, paired_ref[i], paired_det[i], errs[i]))
        print("    ...")
        for i in range(n - 2, n):
            print("    shot {:2d}: ref {:.2f}s  det {:.2f}s  err {:+.3f}s".format(
                i + 1, paired_ref[i], paired_det[i], errs[i]))
    if unmatched_info:
        print("  Unmatched refs (need detection within {:.2f}s):".format(max_match_s))
        for idx, rt, bd, bdet in unmatched_info:
            nearest = "nearest det {:.3f}s (dist {:.3f}s)".format(bdet, bd) if bdet is not None else "no det"
            print("    ref[{}] {:.3f}s -> {}".format(idx, rt, nearest))


def get_waveform_data(audio_path, beep_times=None, shot_times=None, ref_shot_times=None, max_points=5000):
    """Load audio, compute envelopes, downsample. shot_times=算法探测, ref_shot_times=参考数据."""
    data, sr = sf.read(audio_path)
    duration = len(data) / sr
    mono = np.mean(data, axis=1) if len(data.shape) > 1 else np.asarray(data, dtype=np.float64)
    env_win = max(1, int(sr * 0.003))
    upper = maximum_filter1d(mono.astype(np.float64), size=env_win, mode="nearest")
    lower = minimum_filter1d(mono.astype(np.float64), size=env_win, mode="nearest")
    energy = np.abs(mono)
    win = max(1, int(sr * 0.01))
    energy = np.convolve(energy, np.ones(win) / win, mode="same")
    n = len(mono)
    step = max(1, n // max_points)
    t = np.arange(n)[::step].astype(np.float64) / sr
    out = {
        "duration": float(duration),
        "beep_times": list(beep_times) if beep_times else [],
        "t": t.tolist(),
        "mono": mono[::step].tolist(),
        "upper": upper[::step].tolist(),
        "lower": lower[::step].tolist(),
        "energy": energy[::step].tolist(),
        "shot_times": [float(x) for x in shot_times] if shot_times else [],
        "ref_shot_times": [float(x) for x in ref_shot_times] if ref_shot_times else [],
    }
    return out


def write_data_zoom_viewer_envelope_only_html(html_path, data, video_path=None):
    """Write HTML data-zoom viewer with only Vegas-style envelope. If video_path, add video player on top and sync with envelope time.
    Uses audio stream start_time so waveform time 0 (= first audio sample) aligns with video playback (playhead/seek use this offset)."""
    import json
    data = dict(data)
    data.setdefault("video_start_time", get_audio_start_time(video_path) if video_path and os.path.isfile(video_path) else 0.0)
    data_json = json.dumps(data)
    video_src = ""
    video_ui = ""
    video_js_init = ""
    video_js_draw = ""
    video_js_events = ""
    if video_path and os.path.isfile(video_path):
        out_dir = os.path.abspath(os.path.dirname(html_path))
        vid_abs = os.path.abspath(os.path.normpath(video_path))
        data.setdefault("video_abs_path", vid_abs)
        data.setdefault("waveform_source_abs_path", vid_abs)
        try:
            rel = os.path.relpath(vid_abs, out_dir).replace("\\", "/")
        except ValueError:
            rel = os.path.basename(video_path)
        video_src = rel
        video_ui = """<div id="videoBox"><video id="vid" src=\"__VIDEO_SRC__"></video>
  <div id="ctrl"><button id="btnPlay" type="button">Play</button> <span id="timeStr">0.00 / 0.00 s</span></div></div>""".replace(
            "__VIDEO_SRC__", video_src.replace("\\", "/")
        )
        video_js_init = """
var vid=document.getElementById('vid');
var btnPlay=document.getElementById('btnPlay');
var timeStr=document.getElementById('timeStr');
var playheadT=0;
function fmt(t){{ return (t>=0&&t<1e6)?t.toFixed(2):'-'; }}
function updateTimeStr(){{ if(timeStr&&vid) timeStr.textContent=fmt(vid.currentTime)+' / '+fmt(vid.duration||0)+' s'; }}
if(vid){{
 vid.addEventListener('timeupdate',function(){{ playheadT=vid.currentTime; updateTimeStr(); draw(); }});
 vid.addEventListener('loadeddata',function(){{ playheadT=vid.currentTime; updateTimeStr(); draw(); }});
 vid.addEventListener('durationchange',updateTimeStr);
 vid.addEventListener('ended',function(){{ if(btnPlay) btnPlay.textContent='Play'; }});
}}
if(btnPlay&&vid){{ btnPlay.addEventListener('click',function(){{ if(vid.paused){{ vid.play(); btnPlay.textContent='Pause'; }}else{{ vid.pause(); btnPlay.textContent='Play'; }} }}); }}
"""
        video_js_draw = """
 if(typeof playheadT !== 'undefined'&&vid){ var vst=(D.video_start_time!=null?D.video_start_time:0); var pt=playheadT-vst; if(pt>=t0&&pt<=t1){ var px=pad+(pt-t0)/(t1-t0)*plotW; ctx.strokeStyle='#0066cc'; ctx.lineWidth=2; ctx.setLineDash([4,4]); ctx.beginPath(); ctx.moveTo(px,pad); ctx.lineTo(px,H-pad); ctx.stroke(); ctx.setLineDash([]); }}
"""
        video_js_events = """
var wasDrag=drag; drag=0; if(wasDrag) return;
if(vid&&c){{ var rect=c.getBoundingClientRect(); var rx=e.clientX-rect.left; var plotW=c.width-2*pad; if(rx>=pad&&rx<=pad+plotW){{ var vst=(D.video_start_time!=null?D.video_start_time:0); var seekT=t0+(rx-pad)/plotW*(t1-t0); seekT=Math.max(0,Math.min(D.duration,seekT)); vid.currentTime=seekT+vst; playheadT=seekT+vst; if(vid.paused){{ vid.play(); if(btnPlay) btnPlay.textContent='Pause'; }} draw(); }} }}
"""

    top_offset = "24px"
    wrap_top = "24px"
    video_box_top = "24px"
    source_paths_html = ""
    if video_src:
        top_offset = "0"
        wrap_top = "calc(24px + 58vh)"
    if video_path and os.path.isfile(video_path):
        vid_abs_display = os.path.abspath(os.path.normpath(video_path))
        source_paths_html = (
            '<div id="sourcePaths" style="position:fixed;top:24px;left:0;right:0;background:rgba(0,0,0,0.08);'
            'color:#333;font-size:11px;line-height:18px;padding:4px 10px;z-index:1;word-break:break-all;">'
            'Video: ' + html.escape(vid_abs_display) + '<br>Waveform from: ' + html.escape(vid_abs_display) + '</div>'
        )
        video_box_top = "60px"
        wrap_top = "calc(60px + 58vh)"
    html_content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Vegas-style envelope + video</title>
<style>
* {{ margin:0; padding:0; }}
html, body {{ width:100%; height:100%; overflow:hidden; background:#ebebeb; font-family:sans-serif; }}
#hint {{ position:fixed; top:0; left:0; right:0; height:24px; background:rgba(0,0,0,0.06); color:#444; font-size:12px; line-height:24px; padding:0 10px; z-index:1; }}
#sourcePaths {{ }}
#videoBox {{ position:fixed; top:{video_box_top}; left:0; right:0; height:58vh; background:#222; display:flex; flex-direction:column; align-items:center; justify-content:center; z-index:1; }}
#videoBox video {{ max-height:54vh; max-width:100%; object-fit:contain; }}
#ctrl {{ margin-top:6px; }}
#ctrl button {{ padding:4px 12px; cursor:pointer; }}
#wrap {{ position:absolute; top:{wrap_top}; left:0; right:0; bottom:0; cursor:grab; }}
#wrap.dragging {{ cursor:grabbing; }}
#c {{ display:block; width:100%; height:100%; }}
</style></head>
<body><div id="hint">波形：绿=Beep 橙=算法枪声 蓝虚线=参考 &nbsp;|&nbsp; Ctrl+滚轮=缩放 &nbsp; 滚轮=平移 &nbsp; 点击/拖拽波形=跳转播放</div>
{source_paths_html}
{video_ui}
<div id="wrap"><canvas id="c"></canvas></div>
<script>
var D = {data_json};
var c=document.getElementById('c'), wrap=document.getElementById('wrap');
var pad=50, t0=0, t1=D.duration, lastX=0, drag=0, dragSeek=0, dragPlayhead=0;
{video_js_init}
function clamp(v,a,b){{ return Math.max(a,Math.min(b,v)); }}
function hitTestPlayhead(rx){{
 var plotW=c.width-2*pad;
 if(plotW<1 || typeof playheadT === 'undefined' || !vid) return false;
 var vst=(D.video_start_time!=null?D.video_start_time:0);
 var pt=playheadT-vst;
 if(pt<t0||pt>t1) return false;
 var px=pad+(pt-t0)/(t1-t0)*plotW;
 return Math.abs(rx-px)<=12;
}}
function draw(){{
 var W=c.width, H=c.height;
 var ctx=c.getContext('2d');
 ctx.fillStyle='#ebebeb';
 ctx.fillRect(0,0,W,H);
 var n=D.t.length;
 if(n<2 || t1<=t0) return;
 var i0=0, i1=n-1;
 for(var i=0;i<n;i++){{ if(D.t[i]>=t0){{ i0=i; break; }} }}
 for(var i=n-1;i>=0;i--){{ if(D.t[i]<=t1){{ i1=i; break; }} }}
 var seg=Math.max(1,i1-i0+1);
 var plotW=W-2*pad, plotH=H-2*pad;
 var u=D.upper, l=D.lower;
 var mx=Math.max.apply(null, u.slice(i0,i1+1).map(Math.abs).concat(l.slice(i0,i1+1).map(Math.abs)));
 if(mx<1e-9) mx=1;
 var midY=pad+plotH/2, half=plotH/2;
 ctx.strokeStyle='#333';
 ctx.lineWidth=1;
 ctx.beginPath();
 ctx.moveTo(pad,midY);
 ctx.lineTo(pad+plotW,midY);
 ctx.stroke();
 ctx.fillStyle='#782d2d';
 ctx.beginPath();
 for(var i=i0;i<=i1;i++){{
  var x=pad+(i-i0)/(seg-1||1)*plotW;
  var vy0=midY-(u[i]||0)/mx*half, vy1=midY-(l[i]||0)/mx*half;
  if(i===i0) ctx.moveTo(x,vy0);
  else ctx.lineTo(x,vy0);
 }}
 for(var i=i1;i>=i0;i--){{
  var x=pad+(i-i0)/(seg-1||1)*plotW;
  ctx.lineTo(x,midY-(l[i]||0)/mx*half);
 }}
 ctx.closePath();
 ctx.fill();
 if(D.beep_times&&D.beep_times.length){{
  ctx.strokeStyle='rgba(0,150,0,0.8)';
  ctx.lineWidth=2;
  D.beep_times.forEach(function(bt){{
   if(bt>=t0&&bt<=t1){{ var x=pad+(bt-t0)/(t1-t0)*plotW; ctx.beginPath(); ctx.moveTo(x,pad); ctx.lineTo(x,H-pad); ctx.stroke(); }}
  }});
 }}
 if(D.ref_shot_times&&D.ref_shot_times.length){{
  ctx.strokeStyle='rgba(0,100,220,0.85)';
  ctx.lineWidth=2;
  ctx.setLineDash([4,4]);
  D.ref_shot_times.forEach(function(st){{
   if(st>=t0&&st<=t1){{ var x=pad+(st-t0)/(t1-t0)*plotW; ctx.beginPath(); ctx.moveTo(x,pad); ctx.lineTo(x,H-pad); ctx.stroke(); }}
  }});
  ctx.setLineDash([]);
 }}
 if(D.shot_times&&D.shot_times.length){{
  ctx.strokeStyle='rgba(220,80,0,0.9)';
  ctx.lineWidth=2;
  D.shot_times.forEach(function(st){{
   if(st>=t0&&st<=t1){{ var x=pad+(st-t0)/(t1-t0)*plotW; ctx.beginPath(); ctx.moveTo(x,pad); ctx.lineTo(x,H-pad); ctx.stroke(); }}
  }});
 }}
{video_js_draw}
 // Legend (top-right)
 var L=140, lx=W-pad-L-8, ly=pad+6, lh=16;
 ctx.fillStyle='rgba(255,255,255,0.92)';
 ctx.strokeStyle='#888';
 ctx.lineWidth=1;
 ctx.beginPath();
 ctx.rect(lx,ly,L,(typeof vid !== 'undefined' && vid ? 4 : 3)*lh+8);
 ctx.fill();
 ctx.stroke();
 ctx.font='12px sans-serif';
 ctx.textBaseline='middle';
 ctx.fillStyle='rgba(0,120,0,0.9)';
 ctx.fillRect(lx+6,ly+4+lh*0.5-4,20,2);
 ctx.fillStyle='#333';
 ctx.fillText('绿 = Beep（起跑哔声）',lx+32,ly+4+lh*0.5);
 ctx.fillStyle='rgba(220,80,0,0.9)';
 ctx.fillRect(lx+6,ly+4+lh*1.5-4,20,2);
 ctx.fillStyle='#333';
 ctx.fillText('橙 = 算法探测（枪声）',lx+32,ly+4+lh*1.5);
 ctx.fillStyle='rgba(0,100,220,0.85)';
 ctx.setLineDash([3,3]);
 ctx.beginPath();
 ctx.moveTo(lx+6,ly+4+lh*2.5); ctx.lineTo(lx+26,ly+4+lh*2.5);
 ctx.stroke();
 ctx.setLineDash([]);
 ctx.fillStyle='#333';
 ctx.fillText('蓝虚线 = 参考（标注）',lx+32,ly+4+lh*2.5);
 if(typeof vid !== 'undefined' && vid){{
  ctx.strokeStyle='#0066cc';
  ctx.setLineDash([3,3]);
  ctx.beginPath();
  ctx.moveTo(lx+6,ly+4+lh*3.5); ctx.lineTo(lx+26,ly+4+lh*3.5);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle='#333';
  ctx.fillText('浅蓝虚线 = 播放头',lx+32,ly+4+lh*3.5);
 }}
}}
function resize(){{
 c.width=wrap.clientWidth;
 c.height=wrap.clientHeight;
 draw();
}}
function xToTime(rx){{
 var plotW=c.width-2*pad;
 if(rx<pad||rx>pad+plotW) return null;
 var vst=(D.video_start_time!=null?D.video_start_time:0);
 var seekT=t0+(rx-pad)/plotW*(t1-t0);
 seekT=clamp(seekT,0,D.duration);
 return {{ t: seekT, vst: vst }};
}}
wrap.addEventListener('wheel',function(e){{
 e.preventDefault();
 var rect=wrap.getBoundingClientRect();
 var plotW=c.width-2*pad;
 if(plotW<1) return;
 if(e.ctrlKey){{
  var k=e.deltaY>0?0.85:1/0.85;
  var mx=e.clientX-rect.left;
  var frac=((mx-pad)/plotW)||0.5;
  frac=Math.max(0,Math.min(1,frac));
  var tmid=t0+(t1-t0)*frac;
  var dt=t1-t0;
  var dtNew=clamp(dt*k, D.duration/5000, D.duration);
  t0=clamp(tmid-dtNew/2, 0, D.duration-dtNew);
  t1=t0+dtNew;
 }} else {{
  var dx=(e.deltaY||e.deltaX)*0.15*(t1-t0)/10;
  t0=clamp(t0+dx, 0, D.duration-(t1-t0));
  t1=t0+(t1-t0);
 }}
 draw();
}}, {{ passive:false }});
wrap.addEventListener('mousedown',function(e){{
 if(e.button!==0) return;
 var rect=c.getBoundingClientRect();
 var rx=e.clientX-rect.left;
 if(typeof vid !== 'undefined' && vid && hitTestPlayhead(rx)){{ dragPlayhead=1; lastX=e.clientX; return; }}
 drag=1;
 dragSeek=(typeof vid !== 'undefined' && vid)?1:0;
 lastX=e.clientX;
}});
wrap.addEventListener('mousemove',function(e){{
 if(dragPlayhead && typeof vid !== 'undefined' && vid){{
  var rect=c.getBoundingClientRect();
  var rx=e.clientX-rect.left;
  var o=xToTime(rx);
  if(o){{ vid.currentTime=o.t+o.vst; if(typeof playheadT !== 'undefined') playheadT=o.t+o.vst; if(timeStr) timeStr.textContent=vid.currentTime.toFixed(2)+' / '+(vid.duration||0).toFixed(2)+' s'; draw(); }}
  return;
 }}
 if(!drag) return;
 var rect=c.getBoundingClientRect();
 var rx=e.clientX-rect.left;
 if(dragSeek && typeof vid !== 'undefined' && vid){{
  var o=xToTime(rx);
  if(o){{ vid.currentTime=o.t+o.vst; if(typeof playheadT !== 'undefined') playheadT=o.t+o.vst; if(timeStr) timeStr.textContent=vid.currentTime.toFixed(2)+' / '+(vid.duration||0).toFixed(2)+' s'; draw(); }}
 }} else {{
  var dt=t1-t0, dx=(e.clientX-lastX)/c.width*dt;
  lastX=e.clientX;
  t0=clamp(t0-dx,0,D.duration-dt);
  t1=t0+dt;
  draw();
 }}
}});
wrap.addEventListener('mouseup',function(e){{
 if(e.button!==0) return;
 if(dragPlayhead && typeof vid !== 'undefined' && vid && c){{
  var rect=c.getBoundingClientRect();
  var rx=e.clientX-rect.left;
  var o=xToTime(rx);
  if(o){{ vid.currentTime=o.t+o.vst; if(typeof playheadT !== 'undefined') playheadT=o.t+o.vst; if(timeStr) timeStr.textContent=vid.currentTime.toFixed(2)+' / '+(vid.duration||0).toFixed(2)+' s'; vid.play(); if(btnPlay) btnPlay.textContent='Pause'; draw(); }}
  dragPlayhead=0;
  return;
 }}
 if(drag && typeof vid !== 'undefined' && vid && c){{
  var rect=c.getBoundingClientRect();
  var rx=e.clientX-rect.left;
  var o=xToTime(rx);
  if(o){{ vid.currentTime=o.t+o.vst; if(typeof playheadT !== 'undefined') playheadT=o.t+o.vst; if(timeStr) timeStr.textContent=vid.currentTime.toFixed(2)+' / '+(vid.duration||0).toFixed(2)+' s'; if(btnPlay&&vid.paused){{ vid.play(); btnPlay.textContent='Pause'; }} draw(); }}
 }}
 drag=0; dragSeek=0;
}});
wrap.addEventListener('mouseleave',function(){{ drag=0; dragSeek=0; dragPlayhead=0; }});
window.addEventListener('resize',resize);
resize();
</script></body></html>"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Envelope-only zoom viewer saved to: {html_path}" + (" (with video)" if video_src else ""))
    return html_path


def write_calibration_viewer_html(html_path, data, video_path=None):
    """Write HTML for manual shot calibration: envelope + video, black solid draggable lines for shots.
    Initial shots from data['calibration_shots'] (ref or detected). Drag lines to correct time, Save -> *cali.txt (splits)."""
    import json
    data = dict(data)
    data.setdefault("video_start_time", get_audio_start_time(video_path) if video_path and os.path.isfile(video_path) else 0.0)
    calibration_shots = data.get("calibration_shots")
    if not calibration_shots:
        calibration_shots = list(data.get("ref_shot_times") or data.get("shot_times") or [])
    data["calibration_shots"] = sorted(calibration_shots)
    if data.get("beep_times"):
        data["beep_t"] = float(data["beep_times"][0])
    else:
        data.setdefault("beep_t", 0.0)
    if video_path and os.path.isfile(video_path):
        video_dir = os.path.dirname(os.path.abspath(os.path.normpath(video_path)))
        video_base = os.path.splitext(os.path.basename(video_path))[0]
        data["video_base_name"] = video_base
        data["calibration_save_dir"] = video_dir
        data["calibration_save_path"] = os.path.join(video_dir, video_base + "cali.txt")
        data["calibration_http_url"] = "http://127.0.0.1:8765/" + os.path.basename(html_path).replace("\\", "/")
    else:
        if "video_base_name" not in data:
            data["video_base_name"] = os.path.splitext(os.path.basename(html_path))[0].replace("_calibration_viewer", "").replace("_waveform_calibration", "")
        data.setdefault("calibration_save_dir", "")
        data.setdefault("calibration_save_path", "")
        data.setdefault("calibration_http_url", "")
    data_json = json.dumps(data)
    video_src = ""
    video_ui = ""
    video_js_init = ""
    video_js_draw = ""
    video_js_events = ""
    if video_path and os.path.isfile(video_path):
        out_dir = os.path.abspath(os.path.dirname(html_path))
        vid_abs = os.path.abspath(os.path.normpath(video_path))
        try:
            rel = os.path.relpath(vid_abs, out_dir).replace("\\", "/")
        except ValueError:
            rel = os.path.basename(video_path)
        video_src = rel
        video_ui = """<div id="videoBox"><video id="vid" src=\"__VIDEO_SRC__"></video>
  <div id="ctrl"><button id="btnPlay" type="button">Play</button> <span id="timeStr">0.00 / 0.00 s</span></div></div>""".replace(
            "__VIDEO_SRC__", video_src.replace("\\", "/")
        )
        video_js_init = """
var vid=document.getElementById('vid');
var btnPlay=document.getElementById('btnPlay');
var timeStr=document.getElementById('timeStr');
var playheadT=0;
function fmt(t){ return (t>=0&&t<1e6)?t.toFixed(2):'-'; }
function updateTimeStr(){ if(timeStr&&vid) timeStr.textContent=fmt(vid.currentTime)+' / '+fmt(vid.duration||0)+' s'; }
if(vid){
 vid.addEventListener('timeupdate',function(){ playheadT=vid.currentTime; updateTimeStr(); draw(); });
 vid.addEventListener('loadeddata',function(){ playheadT=vid.currentTime; updateTimeStr(); draw(); });
 vid.addEventListener('durationchange',updateTimeStr);
 vid.addEventListener('ended',function(){ if(btnPlay) btnPlay.textContent='Play'; });
}
if(btnPlay&&vid){ btnPlay.addEventListener('click',function(){ if(vid.paused){ vid.play(); btnPlay.textContent='Pause'; }else{ vid.pause(); btnPlay.textContent='Play'; } }); }
"""
        video_js_draw = """
 if(typeof playheadT !== 'undefined'&&vid){ var vst=(D.video_start_time!=null?D.video_start_time:0); var pt=playheadT-vst; if(pt>=t0&&pt<=t1){ var px=pad+(pt-t0)/(t1-t0)*plotW; ctx.strokeStyle='#0066cc'; ctx.lineWidth=2; ctx.setLineDash([4,4]); ctx.beginPath(); ctx.moveTo(px,pad); ctx.lineTo(px,H-pad); ctx.stroke(); ctx.setLineDash([]); }}
"""
        video_js_events = """
var wasDrag=drag; drag=0; if(wasDrag) return;
if(dragLineIndex>=0) return;
if(vid&&c){ var rect=c.getBoundingClientRect(); var rx=e.clientX-rect.left; var plotW=c.width-2*pad; if(rx>=pad&&rx<=pad+plotW){ var vst=(D.video_start_time!=null?D.video_start_time:0); var seekT=t0+(rx-pad)/plotW*(t1-t0); seekT=clamp(seekT,0,D.duration); vid.currentTime=seekT+vst; playheadT=seekT+vst; if(vid.paused){ vid.play(); if(btnPlay) btnPlay.textContent='Pause'; } draw(); } }
"""
    wrap_top = "calc(24px + 58vh)"
    video_box_top = "24px"
    if video_src:
        wrap_top = "calc(60px + 58vh)"
        video_box_top = "60px"
    video_base = data.get("video_base_name", "video")
    html_content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Shot calibration - {html.escape(video_base)}</title>
<style>
* {{ margin:0; padding:0; }}
html, body {{ width:100%; height:100%; overflow:hidden; background:#ebebeb; font-family:sans-serif; }}
#hint {{ position:fixed; top:0; left:0; right:0; height:24px; background:rgba(0,0,0,0.08); color:#444; font-size:12px; line-height:24px; padding:0 10px; z-index:2; display:flex; align-items:center; gap:12px; }}
#videoBox {{ position:fixed; top:24px; left:0; right:0; height:58vh; background:#222; z-index:1; display:flex; flex-direction:column; align-items:center; justify-content:center; }}
#videoBox video {{ max-height:54vh; max-width:100%; object-fit:contain; }}
#ctrl {{ margin-top:6px; }}
#ctrl button {{ padding:4px 12px; cursor:pointer; }}
#wrap {{ position:absolute; top:{wrap_top}; left:0; right:0; bottom:0; cursor:grab; z-index:1; }}
#wrap.dragging {{ cursor:grabbing; }}
#c {{ display:block; width:100%; height:100%; }}
#waveformToolbar {{ position:absolute; top:8px; left:8px; z-index:2; display:flex; flex-direction:column; gap:6px; }}
#btnSave {{ padding:12px 24px; background:#2a7; color:#fff; border:none; border-radius:6px; cursor:pointer; font-size:16px; font-weight:bold; }}
#btnSave:hover {{ background:#3b8; }}
#saveDirHint {{ font-size:11px; color:#555; max-width:320px; word-break:break-all; }}
</style></head>
<body>
<div id="hint">手动校准：黑线=枪声可拖；右键黑线=删除该条，右键空白=添加一条枪声 <span id="fileHint" style="display:none"></span></div>
{video_ui}
<div id="wrap"><canvas id="c"></canvas><div id="waveformToolbar"><button id="btnSave" type="button">保存校准</button><span id="saveDirHint"></span></div></div>
<script>
var D = {data_json};
var calibrationShots = (D.calibration_shots || []).slice();
var dragLineIndex = -1;
var dragPlayhead = 0;
var c=document.getElementById('c'), wrap=document.getElementById('wrap');
var pad=50, t0=0, t1=D.duration, lastX=0, drag=0, dragSeek=0;
{video_js_init}
function clamp(v,a,b){{ return Math.max(a,Math.min(b,v)); }}
function hitTestPlayhead(rx){{
 var plotW=c.width-2*pad;
 if(plotW<1 || typeof playheadT === 'undefined' || !vid) return false;
 var vst=(D.video_start_time!=null?D.video_start_time:0);
 var pt=playheadT-vst;
 if(pt<t0||pt>t1) return false;
 var px=pad+(pt-t0)/(t1-t0)*plotW;
 return Math.abs(rx-px)<=12;
}}
function draw(){{
 var W=c.width, H=c.height;
 var ctx=c.getContext('2d');
 ctx.fillStyle='#ebebeb';
 ctx.fillRect(0,0,W,H);
 var n=D.t.length;
 if(n<2 || t1<=t0) return;
 var i0=0, i1=n-1;
 for(var i=0;i<n;i++){{ if(D.t[i]>=t0){{ i0=i; break; }} }}
 for(var i=n-1;i>=0;i--){{ if(D.t[i]<=t1){{ i1=i; break; }} }}
 var seg=Math.max(1,i1-i0+1);
 var plotW=W-2*pad, plotH=H-2*pad;
 var u=D.upper, l=D.lower;
 var mx=Math.max.apply(null, u.slice(i0,i1+1).map(Math.abs).concat(l.slice(i0,i1+1).map(Math.abs)));
 if(mx<1e-9) mx=1;
 var midY=pad+plotH/2, half=plotH/2;
 ctx.strokeStyle='#333';
 ctx.lineWidth=1;
 ctx.beginPath();
 ctx.moveTo(pad,midY);
 ctx.lineTo(pad+plotW,midY);
 ctx.stroke();
 ctx.fillStyle='#782d2d';
 ctx.beginPath();
 for(var i=i0;i<=i1;i++){{
  var x=pad+(i-i0)/(seg-1||1)*plotW;
  var vy0=midY-(u[i]||0)/mx*half, vy1=midY-(l[i]||0)/mx*half;
  if(i===i0) ctx.moveTo(x,vy0);
  else ctx.lineTo(x,vy0);
 }}
 for(var i=i1;i>=i0;i--){{
  var x=pad+(i-i0)/(seg-1||1)*plotW;
  ctx.lineTo(x,midY-(l[i]||0)/mx*half);
 }}
 ctx.closePath();
 ctx.fill();
 if(D.beep_times&&D.beep_times.length){{
  ctx.strokeStyle='rgba(0,150,0,0.8)';
  ctx.lineWidth=2;
  D.beep_times.forEach(function(bt){{
   if(bt>=t0&&bt<=t1){{ var x=pad+(bt-t0)/(t1-t0)*plotW; ctx.beginPath(); ctx.moveTo(x,pad); ctx.lineTo(x,H-pad); ctx.stroke(); }}
  }});
 }}
 ctx.strokeStyle='#000';
 ctx.lineWidth=3;
 calibrationShots.forEach(function(st){{
  if(st>=t0&&st<=t1){{ var x=pad+(st-t0)/(t1-t0)*plotW; ctx.beginPath(); ctx.moveTo(x,pad); ctx.lineTo(x,H-pad); ctx.stroke(); }}
 }});
{video_js_draw}
 var lx=W-pad-180, ly=pad+6;
 ctx.fillStyle='rgba(255,255,255,0.92)';
 ctx.strokeStyle='#888';
 ctx.beginPath();
 ctx.rect(lx,ly,172,52);
 ctx.fill();
 ctx.stroke();
 ctx.font='16px sans-serif';
 ctx.fillStyle='#000';
 ctx.fillText('黑实线 = 枪声（可拖）',lx+10,ly+20);
 ctx.fillStyle='rgba(0,120,0,0.9)';
 ctx.fillText('绿 = Beep',lx+10,ly+42);
}}
function resize(){{
 c.width=wrap.clientWidth;
 c.height=wrap.clientHeight;
 draw();
}}
function xToTime(rx){{
 var plotW=c.width-2*pad;
 if(rx<pad||rx>pad+plotW) return null;
 var seekT=t0+(rx-pad)/plotW*(t1-t0);
 return clamp(seekT,0,D.duration);
}}
function hitTestLine(rx){{
 var plotW=c.width-2*pad;
 if(plotW<1 || rx<pad || rx>pad+plotW) return -1;
 var threshold=10;
 for(var i=0;i<calibrationShots.length;i++){{
  var x=pad+(calibrationShots[i]-t0)/(t1-t0)*plotW;
  if(Math.abs(rx-x)<=threshold) return i;
 }}
 return -1;
}}
wrap.addEventListener('wheel',function(e){{
 e.preventDefault();
 var rect=wrap.getBoundingClientRect();
 var plotW=c.width-2*pad;
 if(plotW<1) return;
 if(e.ctrlKey){{
  var k=e.deltaY>0?0.85:1/0.85;
  var mx=e.clientX-rect.left;
  var frac=((mx-pad)/plotW)||0.5;
  frac=Math.max(0,Math.min(1,frac));
  var tmid=t0+(t1-t0)*frac;
  var dt=t1-t0;
  var dtNew=clamp(dt*k, D.duration/5000, D.duration);
  t0=clamp(tmid-dtNew/2, 0, D.duration-dtNew);
  t1=t0+dtNew;
 }} else {{
  var dx=(e.deltaY||e.deltaX)*0.15*(t1-t0)/10;
  t0=clamp(t0+dx, 0, D.duration-(t1-t0));
  t1=t0+(t1-t0);
 }}
 draw();
}}, {{ passive:false }});
wrap.addEventListener('mousedown',function(e){{
 if(e.button!==0) return;
 var rect=c.getBoundingClientRect();
 var rx=e.clientX-rect.left;
 if(vid && hitTestPlayhead(rx)){{ dragPlayhead=1; lastX=e.clientX; return; }}
 var idx=hitTestLine(rx);
 if(idx>=0){{ dragLineIndex=idx; drag=1; lastX=e.clientX; return; }}
 drag=1;
 dragSeek=(typeof vid !== 'undefined' && vid)?1:0;
 lastX=e.clientX;
}});
wrap.addEventListener('mousemove',function(e){{
 if(dragPlayhead && vid){{
  var rect=c.getBoundingClientRect();
  var rx=e.clientX-rect.left;
  var o=xToTime(rx);
  if(o!==null){{ var vst=(D.video_start_time!=null?D.video_start_time:0); vid.currentTime=o+vst; if(typeof playheadT !== 'undefined') playheadT=o+vst; if(timeStr) timeStr.textContent=vid.currentTime.toFixed(2)+' / '+(vid.duration||0).toFixed(2)+' s'; draw(); }}
  return;
 }}
 if(dragLineIndex>=0){{
  var rect=c.getBoundingClientRect();
  var rx=e.clientX-rect.left;
  var t=xToTime(rx);
  if(t!==null){{ calibrationShots[dragLineIndex]=t; draw(); }}
  return;
 }}
 if(!drag) return;
 var rect=c.getBoundingClientRect();
 var rx=e.clientX-rect.left;
 if(dragSeek && typeof vid !== 'undefined' && vid){{
  var o=xToTime(rx); if(o!==null){{ var vst=(D.video_start_time!=null?D.video_start_time:0); vid.currentTime=o+vst; if(typeof playheadT !== 'undefined') playheadT=o+vst; if(timeStr) timeStr.textContent=vid.currentTime.toFixed(2)+' / '+(vid.duration||0).toFixed(2)+' s'; draw(); }}
 }} else {{
  var dt=t1-t0, dx=(e.clientX-lastX)/c.width*dt;
  lastX=e.clientX;
  t0=clamp(t0-dx,0,D.duration-dt);
  t1=t0+dt;
  draw();
 }}
}});
wrap.addEventListener('mouseup',function(e){{
 if(e.button!==0) return;
 if(dragPlayhead && vid && c){{
  var rect=c.getBoundingClientRect();
  var rx=e.clientX-rect.left;
  var o=xToTime(rx);
  if(o!==null){{ var vst=(D.video_start_time!=null?D.video_start_time:0); vid.currentTime=o+vst; if(typeof playheadT !== 'undefined') playheadT=o+vst; if(timeStr) timeStr.textContent=vid.currentTime.toFixed(2)+' / '+(vid.duration||0).toFixed(2)+' s'; vid.play(); if(btnPlay) btnPlay.textContent='Pause'; draw(); }}
  dragPlayhead=0;
  return;
 }}
 if(dragLineIndex>=0){{ dragLineIndex=-1; drag=0; return; }}
 if(drag && typeof vid !== 'undefined' && vid && c){{
  var rect=c.getBoundingClientRect();
  var rx=e.clientX-rect.left;
  var o=xToTime(rx);
  if(o!==null){{ var vst=(D.video_start_time!=null?D.video_start_time:0); vid.currentTime=o+vst; if(typeof playheadT !== 'undefined') playheadT=o+vst; if(timeStr) timeStr.textContent=vid.currentTime.toFixed(2)+' / '+(vid.duration||0).toFixed(2)+' s'; if(btnPlay&&vid.paused){{ vid.play(); btnPlay.textContent='Pause'; }} draw(); }}
 }}
 drag=0; dragSeek=0;
}});
wrap.addEventListener('mouseleave',function(){{ dragLineIndex=-1; dragPlayhead=0; drag=0; dragSeek=0; }});
wrap.addEventListener('contextmenu',function(e){{
 e.preventDefault();
 var rect=c.getBoundingClientRect();
 var rx=e.clientX-rect.left;
 var plotW=c.width-2*pad;
 if(plotW<1 || rx<pad || rx>pad+plotW) return;
 var idx=hitTestLine(rx);
 if(idx>=0){{ calibrationShots.splice(idx,1); draw(); return; }}
 var t=xToTime(rx);
 if(t!==null){{ calibrationShots.push(t); calibrationShots.sort(function(a,b){{ return a-b; }}); draw(); }}
}});
var btnSaveEl = document.getElementById('btnSave');
btnSaveEl.addEventListener('mousedown',function(e){{ e.stopPropagation(); }});
btnSaveEl.addEventListener('mouseup',function(e){{ e.stopPropagation(); }});
btnSaveEl.addEventListener('click',function(e){{
 e.preventDefault();
 e.stopPropagation();
 var beep_t = (D.beep_t != null) ? D.beep_t : (D.beep_times && D.beep_times[0] != null ? D.beep_times[0] : 0);
 var sorted = calibrationShots.slice().sort(function(a,b){{ return a-b; }});
 if(sorted.length===0){{ alert('没有枪声时刻'); return; }}
 var splits = [];
 splits.push(parseFloat((sorted[0]-beep_t).toFixed(4)));
 for(var i=1;i<sorted.length;i++) splits.push(parseFloat((sorted[i]-sorted[i-1]).toFixed(4)));
 var text = splits.map(function(x){{ return x.toFixed(4); }}).join('\\n');
 if(D.calibration_save_path && location.protocol==='http:'){{
  fetch('/save_calibration', {{ method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify({{ path: D.calibration_save_path, content: text }}) }})
   .then(function(r){{ return r.json(); }})
   .then(function(o){{ if(o.ok) document.getElementById('saveDirHint').textContent='已保存到: '+o.path; else alert('保存失败: '+(o.error||'')); }})
   .catch(function(e){{ alert('保存失败: '+e); }});
  return;
 }}
 if(D.calibration_http_url){{ window.open(D.calibration_http_url); alert('当前为直接打开文件，无法写入视频目录。已在新标签页打开校准页，请在新页面中点击「保存校准」即可写入视频目录。本页将同时下载一份到您的下载文件夹。'); }}
 var blob = new Blob([text], {{ type: 'text/plain;charset=utf-8' }});
 var a = document.createElement('a');
 a.href = URL.createObjectURL(blob);
 a.download = (D.video_base_name || 'video') + 'cali.txt';
 a.click();
 URL.revokeObjectURL(a.href);
}});
window.addEventListener('resize',resize);
resize();
if(D.calibration_save_dir && !D.calibration_save_path) document.getElementById('saveDirHint').textContent='请保存到: '+D.calibration_save_dir;
if(D.calibration_save_path && location.protocol!=='http:') document.getElementById('saveDirHint').textContent='请用脚本加 --calibration 打开本页，保存将写入视频目录';
if(location.protocol!=='http:' && D.calibration_http_url){{ var el=document.getElementById('fileHint'); if(el){{ el.style.display='inline'; el.innerHTML='| 当前为直接打开，保存会下载到本机。要写入视频目录请 <a href=\"'+D.calibration_http_url+'\" target=\"_blank\">通过此链接打开</a>（需先运行脚本 --calibration）'; }} }}
if(D.calibration_save_path && location.protocol==='http:') document.getElementById('saveDirHint').textContent='保存将写入视频目录: '+D.calibration_save_path;
</script></body></html>"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Calibration viewer saved to: {html_path}")
    return html_path


def run_calibration_server(serve_dir, port=8765, html_basename=None, allowed_save_dir=None):
    """Run HTTP server that serves from serve_dir and accepts POST /save_calibration to write calibration file.
    allowed_save_dir: directory where Save is allowed (default: serve_dir). Use video's dir so Save goes to video folder."""
    try:
        from http.server import HTTPServer, BaseHTTPRequestHandler
    except ImportError:
        from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler  # py2
    allowed_dir = os.path.abspath(os.path.normpath(serve_dir))
    if not os.path.isdir(allowed_dir):
        print(f"Calibration serve: not a directory: {allowed_dir}")
        return
    save_dir = os.path.abspath(os.path.normpath(allowed_save_dir)) if allowed_save_dir else allowed_dir

    class CalibrationHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0].replace("\\", "/")
            if path == "/" or path == "":
                path = "/" + (html_basename or "index.html")
            rel = path.lstrip("/")
            file_path = os.path.normpath(os.path.join(allowed_dir, rel))
            file_path = os.path.abspath(file_path)
            if not file_path.startswith(allowed_dir) or not os.path.isfile(file_path):
                self.send_error(404)
                return
            try:
                with open(file_path, "rb") as f:
                    data = f.read()
            except Exception:
                self.send_error(500)
                return
            ext = os.path.splitext(file_path)[1].lower()
            ctype = "text/html" if ext in (".html", ".htm") else "application/octet-stream"
            if ext == ".txt":
                ctype = "text/plain"
            elif ext == ".mp4":
                ctype = "video/mp4"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", len(data))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path != "/save_calibration":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                data = json.loads(body.decode("utf-8"))
                path = data.get("path", "")
                content = data.get("content", "")
            except Exception as e:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
                return
            req_path = os.path.abspath(os.path.normpath(path))
            if not req_path.startswith(save_dir):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": "path not under video dir"}).encode("utf-8"))
                return
            try:
                os.makedirs(os.path.dirname(req_path), exist_ok=True)
                with open(req_path, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception as e:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "path": req_path}).encode("utf-8"))

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", port), CalibrationHandler)
    url = f"http://127.0.0.1:{port}/{html_basename}" if html_basename else f"http://127.0.0.1:{port}/"
    print(f"Calibration server: {url}  (Save -> auto to video folder, Ctrl+C to stop)")
    threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.shutdown()


def write_image_zoom_viewer_html(image_path):
    """Write HTML that zooms by scaling the PNG (image zoom only)."""
    if not image_path or not os.path.isfile(image_path):
        return None
    out_dir = os.path.dirname(image_path)
    base = os.path.basename(image_path)
    name = os.path.splitext(base)[0]
    html_path = os.path.join(out_dir, name + "_image_zoom.html")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    mime = "image/png"
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Image zoom</title>
<style>
* {{ margin:0; padding:0; }}
html, body {{ width:100%; height:100%; overflow:hidden; background:#1a1a1a; }}
#wrap {{ width:100%; height:100%; overflow:hidden; cursor:grab; position:relative; }}
#wrap.dragging {{ cursor:grabbing; }}
#img {{ position:absolute; top:50%; left:50%; transform-origin: 0 0; }}
</style></head>
<body><div id="wrap"><img id="img" src="data:{mime};base64,{b64}" alt="waveform"></div>
<script>
(function(){{
 var wrap=document.getElementById('wrap'), img=document.getElementById('img');
 var scale=1, tx=0, ty=0, lastX=0, lastY=0, drag=0, initCX=0, initCY=0;
 function clamp(v,a,b){{ return Math.max(a,Math.min(b,v)); }}
 function apply(){{
  img.style.transform = 'translate('+(tx+initCX)+'px,'+(ty+initCY)+'px) scale('+scale+') translate(-50%,-50%)';
 }}
 img.onload = function(){{
  initCX=wrap.clientWidth/2; initCY=wrap.clientHeight/2;
  apply();
 }};
 if(img.complete) img.onload();
 wrap.addEventListener('wheel', function(e){{
  e.preventDefault();
  var k = e.deltaY > 0 ? 0.88 : 1/0.88;
  var rect=wrap.getBoundingClientRect();
  var mx=e.clientX-rect.left, my=e.clientY-rect.top;
  var cx=rect.width/2, cy=rect.height/2;
  var dx=mx-cx-tx, dy=my-cy-ty;
  scale = clamp(scale*k, 0.05, 100);
  tx = mx - cx - dx*k; ty = my - cy - dy*k;
  apply();
 }}, {{ passive:false }});
 wrap.addEventListener('mousedown', function(e){{ drag=1; lastX=e.clientX; lastY=e.clientY; wrap.classList.add('dragging'); }});
 wrap.addEventListener('mousemove', function(e){{ if(drag){{ tx+=e.clientX-lastX; ty+=e.clientY-lastY; lastX=e.clientX; lastY=e.clientY; apply(); }} }});
 wrap.addEventListener('mouseup', function(){{ drag=0; wrap.classList.remove('dragging'); }});
 wrap.addEventListener('mouseleave', function(){{ drag=0; wrap.classList.remove('dragging'); }});
}})();
</script></body></html>"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path

# Try importing matplotlib, if not available use Pillow fallback
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    from PIL import Image, ImageDraw, ImageFont

def get_ffmpeg_cmd():
    """Get ffmpeg command (prefer PATH, otherwise use full path)"""
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    # Common Windows paths
    common_paths = [
        r"C:\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path
    return None

def get_ffprobe_cmd():
    """Get ffprobe command (prefer PATH, otherwise use full path)"""
    if shutil.which("ffprobe"):
        return "ffprobe"
    common_paths = [
        r"C:\ffmpeg-8.0.1-essentials_build\bin\ffprobe.exe",
        r"C:\ffmpeg\bin\ffprobe.exe",
        r"C:\Program Files\ffmpeg\bin\ffprobe.exe",
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path
    return None

def get_fps_from_video(video):
    """Get fps from video via ffprobe"""
    ffprobe = get_ffprobe_cmd()
    if not ffprobe:
        return None
    cmd = (
        f'"{ffprobe}" -v error -select_streams v:0 '
        f'-show_entries stream=r_frame_rate -of csv=p=0 "{video}"'
    )
    out = subprocess.check_output(cmd, shell=True).decode().strip()
    if "/" in out:
        num, den = map(int, out.split("/"))
        return num / den if den else 30.0
    return float(out) if out else 30.0


def get_video_start_time(video):
    """Get start_time of video stream (seconds). Many MP4 have small offset e.g. 0.012."""
    ffprobe = get_ffprobe_cmd()
    if not ffprobe:
        return 0.0
    try:
        cmd = (
            f'"{ffprobe}" -v error -select_streams v:0 '
            f'-show_entries stream=start_time -of csv=p=0 "{video}"'
        )
        out = subprocess.check_output(cmd, shell=True).decode().strip()
        if out and out != "N/A":
            return float(out)
    except Exception:
        pass
    return 0.0


def get_audio_start_time(video):
    """Get start_time of audio stream (seconds). Waveform is from extracted audio, so time 0 = first audio sample; use this so playhead/seek align waveform with video by audio timeline."""
    ffprobe = get_ffprobe_cmd()
    if not ffprobe:
        return 0.0
    try:
        cmd = (
            f'"{ffprobe}" -v error -select_streams a:0 '
            f'-show_entries stream=start_time -of csv=p=0 "{video}"'
        )
        out = subprocess.check_output(cmd, shell=True).decode().strip()
        if out and out != "N/A":
            return float(out)
    except Exception:
        pass
    return 0.0


def get_stream_durations(video):
    """Return (video_duration_sec, audio_duration_sec) from container. Either can be None if stream missing or probe fails."""
    ffprobe = get_ffprobe_cmd()
    if not ffprobe:
        return None, None
    v_dur, a_dur = None, None
    for stream, key in [("v:0", "v_dur"), ("a:0", "a_dur")]:
        try:
            cmd = (
                f'"{ffprobe}" -v error -select_streams {stream} '
                f'-show_entries stream=duration -of csv=p=0 "{video}"'
            )
            out = subprocess.check_output(cmd, shell=True).decode().strip()
            if out and out != "N/A":
                val = float(out)
                if stream == "v:0":
                    v_dur = val
                else:
                    a_dur = val
        except Exception:
            pass
    return v_dur, a_dur


def extract_audio(video, ffmpeg, output_audio="tmp/audio.wav", channels=1):
    """Extract audio from video. channels=1 for mono (beep/energy), channels=2 for stereo (Vegas-style plot).
    Source is always the given video file; waveform duration = audio stream duration in that file."""
    os.makedirs(os.path.dirname(output_audio), exist_ok=True)
    print(f"Extracting audio ({channels}ch) from: {video}")
    print(f"  -> {output_audio}")
    cmd = f'"{ffmpeg}" -y -i "{video}" -ac {channels} -ar 48000 -vn "{output_audio}"'
    subprocess.run(cmd, shell=True, check=True)
    return output_audio

def plot_waveform_pillow(audio_path, output_image, beep_times=None,
                         ref_shot_times=None, audio_shot_times=None, energy_shot_times=None,
                         video_path=None, waveform_source_path=None):
    """Generate simplified waveform plot using Pillow (fallback). 
    beep_times: list of seconds to mark on x-axis.
    ref_shot_times: reference shot times for row 0.
    audio_shot_times: audio-detected shot times for row 1.
    energy_shot_times: energy-threshold detected shots for row 3 (Energy Envelope with detections).
    Row 2 is pure energy envelope with no detection markers.
    video_path, waveform_source_path: optional absolute paths to show at bottom (Video / Waveform from).
    """
    # Read audio (keep stereo for Vegas-style)
    data, sr = sf.read(audio_path)
    duration = len(data) / sr
    
    # Stereo: left & right for Vegas; mono mix for Amplitude/Energy
    if len(data.shape) > 1:
        left_ch = data[:, 0]
        right_ch = data[:, 1]
        data_mono = np.mean(data, axis=1)
    else:
        left_ch = data
        right_ch = data
        data_mono = data
    
    # Upper/lower envelope: sliding-window max/min (~3 ms)
    env_win = max(1, int(sr * 0.003))
    upper_env = maximum_filter1d(data_mono.astype(np.float64), size=env_win, mode='nearest')
    lower_env = minimum_filter1d(data_mono.astype(np.float64), size=env_win, mode='nearest')
    
    # Downsample for display
    max_samples = 2000
    if len(data_mono) > max_samples:
        step = len(data_mono) // max_samples
        left_display = left_ch[::step]
        right_display = right_ch[::step]
        data_display = data_mono[::step]
        upper_display = upper_env[::step]
        lower_display = lower_env[::step]
    else:
        left_display = left_ch
        right_display = right_ch
        data_display = data_mono
        upper_display = upper_env
        lower_display = lower_env
    
    # Energy from mono
    energy = np.abs(data_mono)
    win_size = max(1, int(sr * 0.01))
    smooth_energy = np.convolve(energy, np.ones(win_size) / win_size, mode='same')
    if len(smooth_energy) > max_samples:
        energy_display = smooth_energy[::step]
    else:
        energy_display = smooth_energy
    
    # Create image: 4 rows = Vegas + Amplitude + Energy (pure) + Energy (with detections); light gray bg
    width, height = 1400, 1600
    bg_light_gray = (235, 235, 235)
    img = Image.new('RGB', (width, height), bg_light_gray)
    draw = ImageDraw.Draw(img)
    
    padding = 60
    vegas_row_height = (height - 5 * padding) // 4
    plot_height = (height - 5 * padding) // 4
    plot_width = width - 2 * padding
    left, right = padding, padding + plot_width
    
    max_val = np.max(np.abs(data_display))
    normalized = (data_display / max_val) if max_val > 0 else data_display
    max_energy = np.max(energy_display) if len(energy_display) > 0 else 1
    normalized_energy = (energy_display / max_energy) if max_energy > 0 else energy_display
    
    vegas_color = (120, 45, 45)  # dark reddish-brown
    center_color = (80, 80, 80)
    
    # Row 0: fill between upper/lower envelope, zero at center, fit-to-strip
    def draw_envelope_strip(dr, top, bottom, raw_upper, raw_lower, color, center_line_color):
        mid_y = (top + bottom) // 2
        h = (bottom - top) // 2
        range_max = float(max(np.max(np.abs(raw_upper)), np.max(np.abs(raw_lower)), 1e-9))
        scale = (h / range_max) if range_max > 1e-9 else h
        n = len(raw_upper)
        dr.line([(left, mid_y), (right, mid_y)], fill=center_line_color, width=1)
        pts_upper = []
        pts_lower = []
        for i in range(n):
            x = left + int(i * plot_width / max(1, n - 1)) if n > 1 else left
            y_upper = mid_y - int(float(raw_upper[i]) * scale)
            y_lower = mid_y - int(float(raw_lower[i]) * scale)
            y_upper = max(top, min(bottom, y_upper))
            y_lower = max(top, min(bottom, y_lower))
            pts_upper.append((x, y_upper))
            pts_lower.append((x, y_lower))
        poly = pts_upper + list(reversed(pts_lower))
        if len(poly) >= 3:
            dr.polygon(poly, fill=color, outline=color)
    
    top0, bottom0 = padding, padding + vegas_row_height
    draw_envelope_strip(draw, top0, bottom0, upper_display, lower_display, vegas_color, center_color)
    
    # --- Row 1: Amplitude (line waveform) ---
    top1, bottom1 = padding * 2 + plot_height, padding * 2 + 2 * plot_height
    # Row 2: pure energy (no markers); Row 3: energy + detection markers
    top2_pure, bottom2_pure = padding * 3 + 2 * plot_height, padding * 3 + 3 * plot_height
    top2, bottom2 = padding * 4 + 3 * plot_height, padding * 4 + 4 * plot_height
    mid_y1 = (top1 + bottom1) // 2
    for i in range(len(normalized) - 1):
        x1 = left + int(i * plot_width / len(normalized))
        y1 = mid_y1 - int(normalized[i] * plot_height // 2)
        x2 = left + int((i + 1) * plot_width / len(normalized))
        y2 = mid_y1 - int(normalized[i + 1] * plot_height // 2)
        draw.line([(x1, y1), (x2, y2)], fill='blue', width=1)
    
    # --- Row 2: Pure energy envelope (no detection markers) ---
    for i in range(len(normalized_energy) - 1):
        x1 = left + int(i * plot_width / len(normalized_energy))
        y1 = top2_pure + plot_height - int(normalized_energy[i] * plot_height)
        x2 = left + int((i + 1) * plot_width / len(normalized_energy))
        y2 = top2_pure + plot_height - int(normalized_energy[i + 1] * plot_height)
        draw.line([(x1, y1), (x2, y2)], fill='red', width=2)
    # --- Row 3: Energy envelope with detection markers ---
    for i in range(len(normalized_energy) - 1):
        x1 = left + int(i * plot_width / len(normalized_energy))
        y1 = top2 + plot_height - int(normalized_energy[i] * plot_height)
        x2 = left + int((i + 1) * plot_width / len(normalized_energy))
        y2 = top2 + plot_height - int(normalized_energy[i + 1] * plot_height)
        draw.line([(x1, y1), (x2, y2)], fill='red', width=2)
    
    # Fonts
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    axis_color = (0, 0, 0)
    axis_width = 2
    
    # --- Axes Row 0 (Vegas-style: mono) ---
    draw.line([(left, bottom0), (right, bottom0)], fill=axis_color, width=axis_width)
    draw.line([(left, top0), (left, bottom0)], fill=axis_color, width=axis_width)
    
    # --- Axes Row 1 (Amplitude line) ---
    draw.line([(left, bottom1), (right, bottom1)], fill=axis_color, width=axis_width)
    draw.line([(left, top1), (left, bottom1)], fill=axis_color, width=axis_width)
    for val, label in [(-1, '-1'), (0, '0'), (1, '1')]:
        y = mid_y1 - int(val * plot_height // 2)
        if top1 <= y <= bottom1:
            draw.line([(left - 4, y), (left + 4, y)], fill=axis_color, width=1)
            draw.text((left - 28, y - 6), label, fill='black', font=font_small)
    
    # --- Axes Row 2 (Pure energy) ---
    draw.line([(left, bottom2_pure), (right, bottom2_pure)], fill=axis_color, width=axis_width)
    draw.line([(left, top2_pure), (left, bottom2_pure)], fill=axis_color, width=axis_width)
    for val, label in [(0, '0'), (0.5, '0.5'), (1, '1')]:
        y = bottom2_pure - int(val * plot_height)
        if top2_pure <= y <= bottom2_pure:
            draw.line([(left - 4, y), (left + 4, y)], fill=axis_color, width=1)
            draw.text((left - 28, y - 6), label, fill='black', font=font_small)
    draw.text((8, top2_pure + plot_height // 2 - 40), 'Energy', fill='black', font=font_small)
    # --- Axes Row 3 (Energy + detections) ---
    draw.line([(left, bottom2), (right, bottom2)], fill=axis_color, width=axis_width)
    draw.line([(left, top2), (left, bottom2)], fill=axis_color, width=axis_width)
    for val, label in [(0, '0'), (0.5, '0.5'), (1, '1')]:
        y = bottom2 - int(val * plot_height)
        if top2 <= y <= bottom2:
            draw.line([(left - 4, y), (left + 4, y)], fill=axis_color, width=1)
            draw.text((left - 28, y - 6), label, fill='black', font=font_small)
    draw.text((8, top2 + plot_height // 2 - 40), 'Energy', fill='black', font=font_small)
    
    # Titles
    draw.text((width // 2 - 180, 10), f'Audio Waveform - {os.path.basename(audio_path)}', fill='black', font=font)
    draw.text((left, top0 - 18), 'Vegas-style (envelope) - Reference Shots', fill='black', font=font_small)
    draw.text((left, top1 - 18), 'Amplitude (line) - Audio Detected Shots', fill='black', font=font_small)
    draw.text((left, top2_pure - 18), 'Energy Envelope (no detection)', fill='black', font=font_small)
    draw.text((left, top2 - 18), 'Energy Envelope - Energy Detected Shots (reference only, many false positives)', fill='black', font=font_small)
    
    # X-axis label and time ticks (Time in s) for all four rows
    def draw_time_axis(dr, left_x, plot_w, duration, y_baseline, fnt):
        dr.text((left_x + plot_w // 2 - 30, y_baseline + 2), 'Time (s)', fill='black', font=fnt)
        step = 5.0 if duration >= 15 else (2.0 if duration >= 6 else 1.0)
        t = 0.0
        while t <= duration:
            x = left_x + int((t / duration) * plot_w)
            dr.line([(x, y_baseline - 4), (x, y_baseline + 4)], fill='black', width=1)
            label = f'{t:.0f}' if t == int(t) else f'{t:.1f}'
            dr.text((max(0, x - 8), y_baseline + 6), label, fill='black', font=fnt)
            t += step
    
    draw_time_axis(draw, left, plot_width, duration, bottom0 - 5, font_small)
    draw_time_axis(draw, left, plot_width, duration, bottom1 - 5, font_small)
    draw_time_axis(draw, left, plot_width, duration, bottom2_pure - 5, font_small)
    draw_time_axis(draw, left, plot_width, duration, bottom2 - 5, font_small)
    
    # Mark beep time(s) on x-axis: vertical line across all four plots
    if beep_times and duration > 0:
        beep_color = (0, 128, 0)  # green
        for t in beep_times:
            if 0 <= t <= duration:
                x = left + int((t / duration) * plot_width)
                draw.line([(x, top0), (x, bottom2)], fill=beep_color, width=2)
                label = f"beep {t:.2f}s"
                draw.text((max(0, x - 24), bottom2 + 4), label, fill=beep_color, font=font_small)
    
    # Mark reference shot times on row 0 (bright blue dashed, thicker)
    if ref_shot_times and duration > 0:
        ref_color = (0, 100, 255)  # bright blue
        for t in ref_shot_times:
            if 0 <= t <= duration:
                x = left + int((t / duration) * plot_width)
                # Draw dashed line (draw multiple short segments, thicker and more visible)
                dash_len = 8
                gap_len = 2
                y = top0
                while y < bottom0:
                    # Draw thicker dashed segments
                    draw.line([(x, y), (x, min(y + dash_len, bottom0))], fill=ref_color, width=4)
                    y += dash_len + gap_len
                # Also draw a solid line overlay for better visibility
                draw.line([(x, top0), (x, bottom0)], fill=ref_color, width=1)
    
    # Mark audio-detected shot times on row 1 (bright orange/red solid, thicker)
    if audio_shot_times and duration > 0:
        audio_color = (255, 100, 0)  # bright orange-red for visibility
        for t in audio_shot_times:
            if 0 <= t <= duration:
                x = left + int((t / duration) * plot_width)
                # Draw very thick line (5 pixels wide)
                draw.line([(x, top1), (x, bottom1)], fill=audio_color, width=5)
                # Also draw surrounding lines for even better visibility
                if x > 1 and x < width - 2:
                    draw.line([(x-1, top1), (x-1, bottom1)], fill=audio_color, width=2)
                    draw.line([(x+1, top1), (x+1, bottom1)], fill=audio_color, width=2)
    
    # Mark energy-detected shot times on row 3 only (bright purple/magenta solid, thicker)
    if energy_shot_times and duration > 0:
        energy_color = (255, 0, 255)  # bright magenta for maximum visibility
        for t in energy_shot_times:
            if 0 <= t <= duration:
                x = left + int((t / duration) * plot_width)
                draw.line([(x, top2), (x, bottom2)], fill=energy_color, width=5)
                if x > 1 and x < width - 2:
                    draw.line([(x-1, top2), (x-1, bottom2)], fill=energy_color, width=2)
                    draw.line([(x+1, top2), (x+1, bottom2)], fill=energy_color, width=2)
    
    # Source paths at bottom (absolute paths for verification)
    path_color = (60, 60, 60)
    y1, y2 = height - 36, height - 20
    if video_path:
        draw.text((left, y1), "Video: " + video_path, fill=path_color, font=font_small)
    if waveform_source_path:
        draw.text((left, y2), "Waveform from: " + waveform_source_path, fill=path_color, font=font_small)
    
    # Save
    img.save(output_image)
    print(f"Waveform plot saved to: {output_image} (generated using Pillow)")

def plot_waveform(audio_path, output_image=None, show_plot=True, beep_times=None, 
                  ref_shot_times=None, audio_shot_times=None, energy_shot_times=None,
                  video_path=None):
    """Generate waveform plot. 
    beep_times: list of seconds to mark on x-axis.
    ref_shot_times: reference shot times for ax0 (waveform 1).
    audio_shot_times: audio-detected shot times for ax1 (waveform 2).
    energy_shot_times: energy-threshold detected shots for ax3 (waveform 4). ax2 = pure energy, no markers.
    video_path: optional absolute path to source video (shown on PNG and used for waveform source label).
    """
    if not HAS_MATPLOTLIB:
        if output_image:
            plot_waveform_pillow(audio_path, output_image, beep_times=beep_times,
                                ref_shot_times=ref_shot_times, audio_shot_times=audio_shot_times,
                                energy_shot_times=energy_shot_times,
                                video_path=video_path, waveform_source_path=video_path)
        else:
            print("Warning: matplotlib not installed, use --output to specify output path")
        return None
    
    # Read audio (keep stereo for Vegas-style)
    data, sr = sf.read(audio_path)
    duration = len(data) / sr
    if len(data.shape) > 1:
        left_ch, right_ch = data[:, 0], data[:, 1]
        data_mono = np.mean(data, axis=1)
    else:
        left_ch = right_ch = data
        data_mono = data
    
    # Upper/lower envelope (~3 ms window)
    env_win = max(1, int(sr * 0.003))
    upper_env = maximum_filter1d(data_mono.astype(np.float64), size=env_win, mode='nearest')
    lower_env = minimum_filter1d(data_mono.astype(np.float64), size=env_win, mode='nearest')
    
    time_axis = np.linspace(0, duration, len(data_mono))
    max_samples = 50000
    step = len(data_mono) // max_samples if len(data_mono) > max_samples else 1
    time_display = time_axis[::step]
    left_display = left_ch[::step]
    right_display = right_ch[::step]
    data_display = data_mono[::step]
    upper_display = upper_env[::step]
    lower_display = lower_env[::step]
    
    energy = np.abs(data_mono)
    win_size = max(1, int(sr * 0.01))
    smooth_energy = np.convolve(energy, np.ones(win_size) / win_size, mode='same')
    energy_display = smooth_energy[::step]
    time_energy = time_axis[::step]
    
    # Create figure: 4 rows = Vegas + Amplitude + Energy (pure) + Energy (with detections); light gray bg
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(14, 16))
    bg_light_gray = '#ebebeb'
    fig.patch.set_facecolor(bg_light_gray)
    vegas_color = '#782d2d'  # dark reddish-brown
    
    ax0.set_facecolor(bg_light_gray)
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')
    ax3.set_facecolor('white')
    # Row 0: fill between upper/lower envelope, zero at center
    ax0.fill_between(time_display, lower_display, upper_display, color=vegas_color, alpha=0.9)
    ax0.axhline(0, color='gray', linewidth=0.8)
    if beep_times:
        for t in beep_times:
            if 0 <= t <= duration:
                ax0.axvline(t, color='green', linestyle='--', linewidth=1.5, alpha=0.8)
    # Add reference shot times (blue dashed lines)
    if ref_shot_times:
        for t in ref_shot_times:
            if 0 <= t <= duration:
                ax0.axvline(t, color='blue', linestyle='--', linewidth=1.2, alpha=0.7)
    ax0.set_ylabel('Amplitude', fontsize=11)
    ax0.set_title('Vegas-style (envelope) - Reference Shots', fontsize=12, fontweight='bold')
    ax0.grid(True, alpha=0.25)
    ax0.set_xlim(0, duration)
    lim_env = max(np.max(np.abs(upper_display)), np.max(np.abs(lower_display))) * 1.1
    lim_env = max(lim_env, 1e-6)
    ax0.set_ylim(-lim_env, lim_env)
    ax0.set_xticklabels([])
    
    # Row 1: Amplitude (line waveform, mono)
    ax1.plot(time_display, data_display, linewidth=0.5, color='blue', alpha=0.7)
    if beep_times:
        for t in beep_times:
            if 0 <= t <= duration:
                ax1.axvline(t, color='green', linestyle='--', linewidth=1.5, alpha=0.8)
                ax1.annotate(f'beep {t:.2f}s', xy=(t, ax1.get_ylim()[1]), xytext=(t, ax1.get_ylim()[1] * 1.02),
                             fontsize=9, color='green', ha='center')
    if audio_shot_times:
        for t in audio_shot_times:
            if 0 <= t <= duration:
                ax1.axvline(t, color='orange', linestyle='-', linewidth=1.2, alpha=0.7)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.set_title('Amplitude (line) - Audio Detected Shots', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, duration)
    ax1.set_xticklabels([])
    
    # Row 2: Pure energy envelope (no detection markers)
    ax2.plot(time_energy, energy_display, linewidth=0.8, color='red', alpha=0.7)
    if beep_times:
        for t in beep_times:
            if 0 <= t <= duration:
                ax2.axvline(t, color='green', linestyle='--', linewidth=1.5, alpha=0.8)
    ax2.set_ylabel('Energy Envelope', fontsize=12)
    ax2.set_title('Energy Envelope (no detection)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, duration)
    ax2.set_xticklabels([])
    
    # Row 3: Energy envelope with detection markers
    ax3.plot(time_energy, energy_display, linewidth=0.8, color='red', alpha=0.7)
    if beep_times:
        for t in beep_times:
            if 0 <= t <= duration:
                ax3.axvline(t, color='green', linestyle='--', linewidth=1.5, alpha=0.8)
    if energy_shot_times:
        for t in energy_shot_times:
            if 0 <= t <= duration:
                ax3.axvline(t, color='purple', linestyle='-', linewidth=1.2, alpha=0.7)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Energy Envelope', fontsize=12)
    ax3.set_title('Energy Envelope - Energy Detected Shots (reference only, many FPs)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, duration)
    
    fig.suptitle(f'Audio Waveform - {os.path.basename(audio_path)}', fontsize=14, fontweight='bold', y=1.00)
    info_text = (
        f"Sample Rate: {sr} Hz  |  Duration: {duration:.2f} s  |  Max Amp: {np.max(np.abs(data)):.4f}  |  Mean Energy: {np.mean(energy):.4f}"
    )
    fig.text(0.5, 0.01, info_text, fontsize=9, ha='center', va='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])

    # Mouse-wheel zoom when showing (Windows: button 4=up/zoom-in, 5=down/zoom-out)
    def on_scroll(event):
        if event.inaxes is None:
            return
        ax = event.inaxes
        zoom_in = (getattr(event, 'button', None) == 4 or
                   (getattr(event, 'step', 0) or 0) > 0)
        k = 1.25 if zoom_in else 1.0 / 1.25
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        xc = (event.xdata if event.xdata is not None else (xlim[0] + xlim[1]) / 2)
        yc = (event.ydata if event.ydata is not None else (ylim[0] + ylim[1]) / 2)
        ax.set_xlim(xc - (xc - xlim[0]) / k, xc + (xlim[1] - xc) / k)
        ax.set_ylim(yc - (yc - ylim[0]) / k, yc + (ylim[1] - yc) / k)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    # Save image (light gray bg for Vegas row)
    if output_image:
        plt.savefig(output_image, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        print(f"Waveform plot saved to: {output_image}")
    
    # Show image (scroll over an axis to zoom in/out)
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

def main(video, output_image=None, show_plot=True, use_ref=False, use_calibrate=False, use_train_logreg=False, no_viewers=True, use_calibration_viewer=False):
    # Resolve video to absolute path once so ref/beep lookup and ffmpeg use the same file regardless of cwd
    video = os.path.abspath(os.path.normpath(video))
    if not os.path.isfile(video):
        raise FileNotFoundError("Video file not found: {}".format(video))
    ffmpeg = get_ffmpeg_cmd()
    if not ffmpeg:
        print("ffmpeg not detected, please install first.")
        print("  Windows: Download ffmpeg-release-essentials.zip from https://www.gyan.dev/ffmpeg/builds/")
        raise SystemExit(1)
    
    # Extract mono for beep detection
    audio_mono = extract_audio(video, ffmpeg, "tmp/audio.wav", channels=1)
    # Extract stereo for Vegas-style plot (real L/R)
    audio_stereo = extract_audio(video, ffmpeg, "tmp/audio_stereo.wav", channels=2)
    # Waveform is from this video's audio stream only; duration = length of extracted audio
    v_dur, a_dur = get_stream_durations(video)
    try:
        data_check, sr_check = sf.read(audio_stereo)
        extracted_dur = len(data_check) / sr_check
    except Exception:
        extracted_dur = a_dur if a_dur is not None else 0.0
    if v_dur is not None or a_dur is not None:
        print(f"  Video stream duration: {v_dur:.2f}s" if v_dur is not None else "  Video stream duration: (unknown)")
        print(f"  Audio stream duration: {a_dur:.2f}s" if a_dur is not None else "  Audio stream duration: (unknown)")
    print(f"  Extracted waveform length: {extracted_dur:.2f}s (from same file: {os.path.basename(video)})")
    if v_dur is not None and a_dur is not None and abs(v_dur - a_dur) > 0.05:
        print(f"  Note: video and audio duration differ; waveform follows audio stream. Sync may look off at the end.")
    
    # Beep: prefer same-folder *beep.txt (e.g. 1beep.txt for 1.mp4), else detect
    beep_times = None
    video_dir = os.path.dirname(os.path.abspath(video))
    video_base = os.path.splitext(os.path.basename(video))[0]
    beep_txt = os.path.join(video_dir, video_base + "beep.txt")
    if os.path.isfile(beep_txt):
        try:
            with open(beep_txt, "r", encoding="utf-8") as f:
                line = f.readline().strip()
                if line:
                    beep_times = [float(line)]
                    print(f"Beep time from {os.path.basename(beep_txt)}: {beep_times[0]:.4f}s")
        except Exception:
            pass
    shot_times = []
    ref_times = None
    ref_times_for_plot = None  # only set when video has its own .txt/.jpg (no ref for test data)
    energy_shot_times = []
    try:
        fps = get_fps_from_video(video)
        if fps is not None and beep_times is None:
            beeps = detect_beeps(audio_mono, fps)
            beep_times = [b["t"] for b in beeps]
            if beep_times:
                print(f"Beep time(s) for x-axis: {beep_times}")
        # When we have beep (from file or detect): get ref_times and run detector (row0=ref, row1=audio/CNN shots)
        if fps is not None and beep_times:
            ref_times, ref_source = get_ref_times_and_source(video, float(beep_times[0]))
            if not ref_times:
                ref_times = ref_shot_times(beep_times[0])
                ref_source = "reference_splits(beep)"
            beep_src = os.path.basename(beep_txt) if (beep_txt and os.path.isfile(beep_txt)) else "detect_beeps"
            try:
                cal = load_calibrated_params()
                use_ast = cal.get("use_ast_gunshot", bool(cal.get("ast_gunshot_path") and not cal.get("cnn_gunshot_path")))
                detector_name = "AST" if use_ast else "CNN"
            except Exception:
                detector_name = "CNN"
            print(f"[Ref] {len(ref_times)} shots from {ref_source}, beep from {beep_src}")
            print(f"[Detector] {detector_name} (use_cnn_only=True in config)")
            # Plot reference whenever ref came from same-base .txt (e.g. 2.txt); hide only for global fallback.
            ref_times_for_plot = ref_times if (ref_times and ref_source != "reference_splits(beep)") else None
            if use_calibrate:
                run_calibration(audio_mono, fps, ref_times)
                return
            if use_train_logreg:
                    # Train logistic regression on candidate features using ref times
                    # Build candidate features and feature context
                    MISS_WEIGHT = 4.5  # FN/miss weight for single-video high-recall training
                    GT_COVER_TOL = 0.04
                    data_lr, sr_lr = sf.read(audio_mono)
                    if len(data_lr.shape) > 1:
                        data_lr = np.mean(data_lr, axis=1)
                    data_lr = np.asarray(data_lr, dtype=np.float64)

                    # If we already have a logreg model saved, use it to compute FN times
                    cal_existing = load_calibrated_params() or {}
                    existing_model = cal_existing.get("logreg_model", None)

                    # Use use_mfcc=True so candidates have mfcc_band for 9-dim LogReg training
                    cal_cfg = cal_existing or {}
                    shots_result = detect_shots_improved(
                        data_lr, sr_lr, fps,
                        cluster_window_sec=cal_cfg.get("cluster_window_sec", 0.25),
                        mad_k=cal_cfg.get("mad_k", 6.0),
                        candidate_min_dist_ms=cal_cfg.get("candidate_min_dist_ms", 50),
                        score_weights=cal_cfg.get("score_weights"),
                        min_confidence_threshold=None,
                        logreg_model=existing_model,
                        return_candidates=True,
                        return_feature_context=True,
                        use_mfcc=True,
                    )
                    if not isinstance(shots_result, tuple) or len(shots_result) != 3:
                        print("Train logreg: no candidates/context returned.")
                        return
                    shots_cur, candidates, context = shots_result
                    if not candidates:
                        print("Train logreg: no candidates to train.")
                        return

                    # ---- Diagnostic: GT coverage by any candidate (Stage1 health) ----
                    cand_times = [float(c.get("t", 0.0)) for c in candidates]
                    gt_covered = 0
                    for rt in ref_times:
                        if any(abs(t - rt) <= GT_COVER_TOL for t in cand_times):
                            gt_covered += 1
                    # Use ASCII-only print for Windows console compatibility
                    print(f"GT coverage = {gt_covered}/{len(ref_times)} (any candidate within +/-{GT_COVER_TOL:.2f}s)")

                    # Label candidates and misses (binary: TP+FN=1, FP=0)
                    window_before = 0.02
                    window_after = 0.08
                    X = []
                    y = []
                    weights = []
                    # Candidates: TP or FP
                    for c in candidates:
                        t = c["t"]
                        is_pos = any((t >= rt - window_before) and (t <= rt + window_after) for rt in ref_times)
                        y.append(1 if is_pos else 0)
                        weights.append(1.0)
                        X.append([
                            float(c.get("onset_log", 0.0)),
                            float(c.get("r1_log", 0.0)),
                            float(c.get("r2_log", 0.0)),
                            float(c.get("flatness", 0.0)),
                            float(c.get("attack_norm", 0.0)),
                            float(c.get("E_low_ratio", 0.0)),
                            1.0 if c.get("hard_neg_slam", False) else 0.0,
                            1.0 if c.get("hard_neg_metal", False) else 0.0,
                            float(c.get("mfcc_band", 0.0)),
                        ])

                    # ---- Step 1: Add nearest candidate for FN GT (most important) ----
                    # FN defined using current detector output (shots_cur) vs GT with tight tol
                    shot_times_cur = [float(s.get("t", 0.0)) for s in (shots_cur or [])]
                    fn_times = []
                    for rt in ref_times:
                        if not any(abs(st - rt) <= GT_COVER_TOL for st in shot_times_cur):
                            fn_times.append(float(rt))

                    # For each FN GT, find nearest candidate peak; if close enough, add as positive with MISS_WEIGHT
                    for t_gt in fn_times:
                        if not cand_times:
                            break
                        nearest_i = int(np.argmin(np.abs(np.asarray(cand_times) - float(t_gt))))
                        t_peak = float(cand_times[nearest_i])
                        if abs(t_peak - float(t_gt)) < window_after:
                            c = candidates[nearest_i]
                            y.append(1)
                            weights.append(MISS_WEIGHT)
                            X.append([
                                float(c.get("onset_log", 0.0)),
                                float(c.get("r1_log", 0.0)),
                                float(c.get("r2_log", 0.0)),
                                float(c.get("flatness", 0.0)),
                                float(c.get("attack_norm", 0.0)),
                                float(c.get("E_low_ratio", 0.0)),
                                1.0 if c.get("hard_neg_slam", False) else 0.0,
                                1.0 if c.get("hard_neg_metal", False) else 0.0,
                                float(c.get("mfcc_band", 0.0)),
                            ])

                    # Miss samples: ref times without nearby candidate -> synthesize positive at GT time
                    for rt in ref_times:
                        has_candidate = any(abs(c["t"] - rt) <= window_after for c in candidates)
                        if not has_candidate:
                            feat = compute_feature_at_time(context, rt, window_before, window_after)
                            y.append(1)
                            weights.append(MISS_WEIGHT)
                            X.append([
                                float(feat.get("onset_log", 0.0)),
                                float(feat.get("r1_log", 0.0)),
                                float(feat.get("r2_log", 0.0)),
                                float(feat.get("flatness", 0.0)),
                                float(feat.get("attack_norm", 0.0)),
                                float(feat.get("E_low_ratio", 0.0)),
                                1.0 if feat.get("hard_neg_slam", False) else 0.0,
                                1.0 if feat.get("hard_neg_metal", False) else 0.0,
                                0.0,
                            ])
                    if len(set(y)) < 2:
                        print("Train logreg: not enough class variety.")
                        return
                    model = train_logreg(np.asarray(X), np.asarray(y), sample_weight=np.asarray(weights))
                    # Persist model to calibrated params
                    cal = load_calibrated_params() or {}
                    cal["logreg_model"] = model
                    out_path = os.path.join(os.getcwd(), CALIBRATED_PARAMS_FILENAME)
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(cal, f, indent=2)
                    print(f"LogReg model saved to: {out_path}")
                    return
            shots = detect_shots(audio_mono, fps)
            # Shots all happen after beep: drop any detection before beep (video start noise)
            if beep_times and len(shots) > 0:
                t0_beep = float(beep_times[0])
                shots = [s for s in shots if s["t"] >= t0_beep]
            shot_times = [s["t"] for s in shots]
            if shot_times:
                print(f"Shot time(s) for envelope markers: {shot_times}")
            if beep_times and ref_times_for_plot is not None:
                calibration_report(ref_times_for_plot, shot_times)
            
            # Energy-threshold detection (filter -> envelope > 0.3 = shot) for row 2
            print("Detecting shots from energy envelope (threshold 0.3)...")
            try:
                energy_shots = detect_shots_energy(
                    audio_path=audio_mono,
                    energy_threshold=0.3,
                    min_dist_sec=0.08,
                    t0_beep=float(beep_times[0]) if beep_times else None,
                )
                energy_shot_times = [s["t"] for s in energy_shots]
                if energy_shot_times:
                    print(f"Energy-detected shot time(s): {len(energy_shot_times)} shots")
            except Exception as e:
                print(f"Energy detection skipped: {e}")
    except Exception as e:
        print(f"Beep/shot detection skipped: {e}")
    
    # Default output: same directory as the video (so all outputs stay with the video)
    if not output_image:
        video_dir = os.path.dirname(video)
        video_name = os.path.splitext(os.path.basename(video))[0]
        output_image = os.path.join(video_dir, video_name + "_waveform.png")
    
    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    
    # Generate waveform plot from stereo (Vegas-style L/R)
    n_ref_plot = len(ref_times_for_plot) if ref_times_for_plot else 0
    print(f"Generating waveform plot... (ref lines on plot: {n_ref_plot})")
    if not HAS_MATPLOTLIB:
        if not output_image:
            print("Warning: matplotlib not installed, please install matplotlib or use --output to specify output path")
            print("Install command: pip install matplotlib")
            return
    video_abs = os.path.abspath(os.path.normpath(video))
    plot_waveform(audio_stereo, output_image, show_plot, 
                  beep_times=beep_times,
                  ref_shot_times=ref_times_for_plot if ref_times_for_plot else None,
                  audio_shot_times=shot_times if shot_times else None,
                  energy_shot_times=energy_shot_times if energy_shot_times else None,
                  video_path=video_abs)
    
    viewer_envelope_path = ""
    wdata = None
    if not no_viewers and output_image:
        out_dir = os.path.dirname(output_image)
        name = os.path.splitext(os.path.basename(output_image))[0]
        viewer_envelope_path = os.path.join(out_dir, name + "_viewer_envelope.html")
        try:
            wdata = get_waveform_data(
                audio_stereo,
                beep_times=beep_times,
                shot_times=shot_times,
                ref_shot_times=ref_times_for_plot if ref_times_for_plot is not None else [],
            )
            write_data_zoom_viewer_envelope_only_html(viewer_envelope_path, wdata, video_path=video)
        except Exception as e:
            print(f"Envelope viewer skipped: {e}")
        if use_calibration_viewer and wdata is not None:
            try:
                cal_data = dict(wdata)
                cal_data["calibration_shots"] = list(ref_times_for_plot if ref_times_for_plot else (shot_times or []))
                cal_data["video_base_name"] = name
                cal_viewer_path = os.path.join(out_dir, name + "_calibration_viewer.html")
                write_calibration_viewer_html(cal_viewer_path, cal_data, video_path=video)
                video_dir_cal = os.path.dirname(os.path.abspath(os.path.normpath(video)))
                run_calibration_server(out_dir, port=8765, html_basename=os.path.basename(cal_viewer_path), allowed_save_dir=video_dir_cal)
            except Exception as e:
                print(f"Calibration viewer skipped: {e}")
    
    print(f"\nDone!")
    print(f"  Audio (mono): {audio_mono}")
    print(f"  Audio (stereo): {audio_stereo}")
    print(f"  Waveform plot: {output_image}")
    if not no_viewers and viewer_envelope_path and os.path.isfile(viewer_envelope_path):
        print(f"  Envelope viewer: {viewer_envelope_path} (open in browser)")
    if use_calibration_viewer and output_image:
        cal_path = os.path.join(os.path.dirname(output_image), os.path.splitext(os.path.basename(output_image))[0] + "_calibration_viewer.html")
        if os.path.isfile(cal_path):
            print(f"  Calibration viewer: {cal_path} (drag lines, Save -> *cali.txt)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract audio from video and generate waveform plot")
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--output", help="Output image path (default: same dir as video, e.g. <video_dir>/<name>_waveform.png)")
    parser.add_argument("--no-show", action="store_true", help="Don't show plot window (save file only)")
    parser.add_argument("--use-ref", action="store_true", help="Use reference shot times (training set only; for comparing to detector)")
    parser.add_argument("--calibrate", action="store_true", help="Tune detector on training set (1.mp4), save best params for new videos")
    parser.add_argument("--train-logreg", action="store_true", help="Train logistic regression on candidate features (requires beep/ref)")
    parser.add_argument("--viewers", action="store_true", help="Generate zoom viewer and envelope+video HTML (default: off)")
    parser.add_argument("--calibration", action="store_true", help="Generate manual shot calibration viewer (draggable lines); starts local server so Save writes to video folder")
    args = parser.parse_args()
    
    main(
        args.video, args.output, show_plot=not args.no_show,
        use_ref=args.use_ref, use_calibrate=args.calibrate, use_train_logreg=args.train_logreg,
        no_viewers=not args.viewers,
        use_calibration_viewer=args.calibration,
    )
