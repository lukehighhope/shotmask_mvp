"""
手动标注工具：0.5 倍速播放 + 可拖拽竖线标注枪声/beep。
保存结果直接写入 *cali.txt 和 *beep.txt（与 extract_audio_plot.py 格式一致）。

Usage:
  python annotate_shots.py --video "test data/v1.mp4"
  python annotate_shots.py --video "traning data/jeff 03-04/S1-main.mp4"

操作说明：
  - 右键点击波形       → 新增枪声标注线（黑线）
  - 右键点击已有黑线   → 删除该枪声
  - 拖拽黑线          → 移动枪声时间
  - Ctrl+右键         → 新增/删除 Beep（绿线）
  - 拖拽绿线          → 移动 Beep 时间
  - 点击波形空白处     → 跳转视频到该时间并播放
  - 保存校准          → 写入 *cali.txt
  - 保存 Beep         → 写入 *beep.txt
  - 速度按钮          → 0.25x / 0.5x（默认）/ 1x

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

# 把项目根目录加入 import 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_audio_plot import (
    write_calibration_viewer_html,
    get_waveform_data,
    get_audio_start_time,
    extract_audio as extract_audio_ch,
)
from detectors.beep import detect_beeps
from detectors.shot_audio import detect_shots
from main import get_ffmpeg_cmd, get_ffprobe_cmd, ffprobe_info
import subprocess
import json
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import tkinter as tk
from tkinter import filedialog


_ALLOWED_BACKENDS = frozenset({"auto", "ast", "cnn"})


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


def _detect_shots_with_backend(audio_mono, fps, backend):
    backend = (backend or "auto").lower()
    if backend == "ast":
        return detect_shots(audio_mono, fps, use_ast_gunshot=True)
    if backend == "cnn":
        return detect_shots(audio_mono, fps, use_ast_gunshot=False)
    return detect_shots(audio_mono, fps)


def _patch_html_playback_speed(html_path, default_speed=0.5):
    """Patch generated calibration HTML:
    1. Default playback speed
    2. Speed toggle buttons
    3. Completely replace mouse handlers with clean annotation behavior:
       - Left-click on waveform → seek + play from that time
       - Left-drag playhead (blue line) → drag to position, release to play from there
       - Left-drag on empty waveform → pan waveform view (no video seek)
       - Right-click = add/delete shot/beep (unchanged)
       - Scroll = pan; Ctrl+scroll = zoom (unchanged)
    """
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

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
    new_ctrl = (
        '<div id="ctrl" style="font-size:16px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;">'
        '<button id="btnPlay" type="button" style="padding:6px 18px;font-size:16px;">Play</button>'
        '<button id="spd025" onclick="_setSpeed(0.25)" style="padding:6px 14px;font-size:16px;">0.25x</button>'
        f'<button id="spd050" onclick="_setSpeed(0.5)" style="padding:6px 14px;font-size:16px;font-weight:bold;outline:2px solid #2a7;">0.5x</button>'
        '<button id="spd100" onclick="_setSpeed(1.0)" style="padding:6px 14px;font-size:16px;">1x</button>'
        '<button id="btnOpenVideo" style="padding:7px 22px;background:#36a;color:#fff;border:none;border-radius:5px;'
        'cursor:pointer;font-size:17px;font-weight:bold;margin-left:12px;">Open Video</button>'
        '<span id="loadStatus" style="color:#555;font-size:14px;"></span>'
    )
    html = html.replace(old_ctrl, new_ctrl, 1)

    # ── 3. Toast style (no JS yet — Save All button added after cloneNode in patch 5) ──
    html = html.replace(
        "</head>",
        """<style>
#saveToast{position:fixed;top:18px;right:18px;padding:12px 22px;border-radius:8px;
  font-size:14px;font-weight:bold;color:#fff;z-index:9999;display:none;
  box-shadow:0 3px 12px rgba(0,0,0,0.25);}
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
        '\n        if(vid){vid.src="/video?"+Date.now();vid.load();'
        f'\n          vid.playbackRate={_spd};playheadT=0;'
        '\n        }'
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
    // Ignore clicks that originate inside the toolbar (buttons, etc.)
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
      if(_moved){ var t=xToT(rx); if(t!==null){ calibrationShots[_lineIdx]=t; draw(); } }
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
      // Clean click on waveform → seek + play
      var t=xToT(rx); if(t!==null) seekAndPlay(t);
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
      if(Math.abs(rx-px)<=10){ calibrationShots.splice(i,1); draw(); return; }
    }
    var t=xToT(rx); if(t!==null){ calibrationShots.push(t); calibrationShots.sort(function(a,b){return a-b;}); draw(); }
  });

  // ---- Save All button (created after cloneNode so it keeps its listener) ----
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

    function doSave(){
      var caliPath=D.calibration_save_path;
      var beepPath=D.calibration_beep_path;
      var shots=calibrationShots.slice().sort(function(a,b){return a-b;});
      var beeps=calibrationBeepTimes.slice().sort(function(a,b){return a-b;});
      var caliText=shots.map(function(t){return t.toFixed(4);}).join('\\n');
      var beepText=beeps.map(function(t){return t.toFixed(4);}).join('\\n');
      var saved=0, total=0;
      function done(ok,msg){ saved++; if(saved===total){ showToast(ok?'Saved!':('Error: '+msg), ok); } }
      if(caliPath){
        total++;
        fetch('/save_calibration',{method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({path:caliPath,content:caliText})})
        .then(function(r){return r.json();})
        .then(function(o){ done(o.ok, o.error||''); })
        .catch(function(e){ done(false,''+e); });
      }
      if(beepPath){
        total++;
        fetch('/save_calibration',{method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({path:beepPath,content:beepText})})
        .then(function(r){return r.json();})
        .then(function(o){ done(o.ok, o.error||''); })
        .catch(function(e){ done(false,''+e); });
      }
      if(total===0){ showToast('No save path set.', false); }
    }
    window._doSave = doSave;

    // Hide the original two save buttons
    var bs=document.getElementById('btnSave');
    var bb=document.getElementById('btnSaveBeep');
    if(bs) bs.style.display='none';
    if(bb) bb.style.display='none';

    // Create Save All button and insert before Play button in #ctrl
    var btnAll=document.createElement('button');
    btnAll.type='button';
    btnAll.textContent='Save All';
    btnAll.style.cssText='padding:6px 18px;background:#c75;color:#fff;border:none;border-radius:5px;cursor:pointer;font-size:16px;font-weight:bold;';
    btnAll.addEventListener('mousedown',function(e){e.stopPropagation();});
    btnAll.addEventListener('mouseup',  function(e){e.stopPropagation();});
    btnAll.addEventListener('click',function(e){
      e.preventDefault(); e.stopPropagation();
      var msg='Save annotations?\\n'+
        (D.calibration_save_path?'  shots -> '+D.calibration_save_path+'\\n':'')+
        (D.calibration_beep_path?'  beep  -> '+D.calibration_beep_path:'');
      if(!confirm(msg)) return;
      doSave();
    });
    var ctrl=document.getElementById('ctrl');
    var playBtn=document.getElementById('btnPlay');
    if(ctrl && playBtn){
      ctrl.insertBefore(btnAll, playBtn);
    } else if(ctrl){
      ctrl.insertBefore(btnAll, ctrl.firstChild);
    }

    // Show current save paths (call this after D is updated too)
    window._updateSavePathDisplay = function(){
      function splitPath(p){
        if(!p) return [];
        var i=Math.max(p.lastIndexOf('/'), p.lastIndexOf(String.fromCharCode(92)));
        return i<0 ? ['',p] : [p.slice(0,i+1), p.slice(i+1)];
      }
      var sp=splitPath(D.calibration_save_path||D.calibration_beep_path||'');
      var dir=sp[0]||'';
      var el=document.getElementById('savePathDisplay');
      if(el){
        var parts=[];
        if(D.calibration_save_path) parts.push('shots: '+splitPath(D.calibration_save_path)[1]);
        if(D.calibration_beep_path) parts.push('beep: '+splitPath(D.calibration_beep_path)[1]);
        el.textContent=parts.join('   |   ');
      }
      var hint=document.getElementById('saveDirHint');
      if(hint){
        var row2=(D.calibration_save_path?'shots: '+splitPath(D.calibration_save_path)[1]:'')+
                 (D.calibration_beep_path?'   beep: '+splitPath(D.calibration_beep_path)[1]:'');
        hint.textContent=dir?('Save dir: '+dir+'  |  '+row2):row2;
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


def _prepare_video(video_path, ffmpeg, force_detect=False, session_shot_backend="auto"):
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
        print("Detecting beep...")
        try:
            beeps = detect_beeps(audio_mono, fps)
            beep_times = [float(b["t"]) for b in beeps]
            print(f"  Detected {len(beep_times)} beep(s): {beep_times}")
        except Exception as e:
            print(f"  Beep detection failed: {e}")

    if not has_cali:
        print("Detecting gunshots...")
        try:
            t0_beep = beep_times[-1] if beep_times else 0.0
            back, src = _resolve_shot_detector_backend(
                anno_dir, video_base, cali_path, session_shot_backend
            )
            print(f"  Shot detector backend: {back} ({src})")
            shots = _detect_shots_with_backend(audio_mono, fps, back)
            shots = [s for s in shots if float(s["t"]) >= t0_beep]
            shot_times = [float(s["t"]) for s in shots]
            print(f"  Detected {len(shot_times)} shot(s)")
        except Exception as e:
            print(f"  Shot detection failed: {e}")

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
    return cal_data, video_dir, video_base, anno_dir


def run_annotation_server(serve_dir, port, html_basename, ffmpeg_cmd,
                          initial_video_path=None, initial_save_dir=None,
                          default_speed=0.5, session_shot_backend="auto"):
    """HTTP server for the annotation tool.

    Endpoints:
      GET  /<file>          — serve static files with Range support
      GET  /video           — serve the currently loaded video with Range support
      POST /save_calibration — write annotation files
      POST /load_video      — process a new video and return updated waveform JSON
    """
    serve_dir  = os.path.abspath(serve_dir)

    # Shared mutable state accessible by the request handler
    state = {
        "video_path": initial_video_path,
        "save_dir":   initial_save_dir or serve_dir,
        "session_shot_backend": session_shot_backend,
    }

    def _norm(p):
        return os.path.normcase(os.path.normpath(p))

    def _serve_file(handler, file_path):
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
                handler.end_headers()
                handler.wfile.write(data)
            else:
                with open(file_path, "rb") as f:
                    data = f.read()
                handler.send_response(200)
                handler.send_header("Content-Type", ctype)
                handler.send_header("Content-Length", file_size)
                handler.send_header("Accept-Ranges", "bytes")
                handler.end_headers()
                handler.wfile.write(data)
        except Exception:
            handler.send_error(500)

    class AnnotationHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]

            # Special: open native file picker and return chosen path as JSON
            if path == "/open_file_dialog":
                def _pick():
                    root = tk.Tk()
                    root.withdraw()
                    root.attributes("-topmost", True)
                    chosen = filedialog.askopenfilename(
                        title="Select video file",
                        filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.webm"), ("All files", "*.*")],
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

            # Special: serve the currently loaded video
            if path == "/video":
                vp = state.get("video_path")
                if vp and os.path.isfile(vp):
                    _serve_file(self, vp)
                else:
                    self.send_error(404)
                return

            path = path.replace("\\", "/").lstrip("/")
            if not path:
                path = html_basename or "index.html"

            if ".." in path:
                self.send_error(403)
                return

            file_path = os.path.abspath(os.path.join(serve_dir, path))
            if not _norm(file_path).startswith(_norm(serve_dir)) or not os.path.isfile(file_path):
                self.send_error(404)
                return
            _serve_file(self, file_path)

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
                    )
                    state["video_path"] = os.path.abspath(os.path.normpath(vpath))
                    state["save_dir"]   = anno_dir  # save to annotation dir, not tmp/
                    print(f"[load_video] Sending response for {video_base} ...")
                    try:
                        payload = json.dumps({"ok": True, "data": cal_data}).encode("utf-8")
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

    server = HTTPServer(("127.0.0.1", port), AnnotationHandler)
    url = f"http://127.0.0.1:{port}/{html_basename}"
    print(f"Annotation server: {url}  (Ctrl+C to stop)")
    threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def main():
    ap = argparse.ArgumentParser(
        description="Manual annotation tool: 0.5x playback, drag lines for shots/beep, save to *cali.txt"
    )
    ap.add_argument("--video", default=None, help="Video file path (.mp4). If omitted, a file picker opens.")
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
    args = ap.parse_args()

    ffmpeg = get_ffmpeg_cmd()
    if not ffmpeg:
        print("Error: ffmpeg not found.")
        return 1

    # If no video given, open native file picker
    if not args.video:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askopenfilename(
            title="Select video file to annotate",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.webm"), ("All files", "*.*")],
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

    # ── 1. Process initial video ──────────────────────────────────────
    cal_data, video_dir, video_base, anno_dir = _prepare_video(
        video,
        ffmpeg,
        force_detect=args.no_detect,
        session_shot_backend=args.shot_backend,
    )

    # ── 2. Generate HTML ──────────────────────────────────────────────
    html_name = "annotate.html"
    html_path = os.path.join(out_dir, html_name)

    # Write HTML with video src pointing to /video (served dynamically)
    write_calibration_viewer_html(html_path, cal_data, video_path=video)

    _patch_html_playback_speed(html_path, default_speed=args.speed)
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
        default_speed=args.speed,
        session_shot_backend=args.shot_backend,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
