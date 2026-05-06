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
"""
import os
import sys
# Force unbuffered stdout so progress prints appear immediately in terminals
sys.stdout.reconfigure(line_buffering=True)
import argparse

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
        '<div id="ctrl">'
        '<button id="btnPlay" type="button">Play</button>&nbsp;'
        '<button id="spd025" onclick="_setSpeed(0.25)">0.25x</button>'
        f'<button id="spd050" onclick="_setSpeed(0.5)" style="font-weight:bold;outline:2px solid #2a7;">0.5x</button>'
        '<button id="spd100" onclick="_setSpeed(1.0)">1x</button>'
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

    # ── 4. Open Video button (calls native file picker on server) ────────
    _spd = str(default_speed)
    open_bar = (
        '\n<div id="openVideoBar" style="padding:6px 10px;background:#f5f5f5;'
        'border-bottom:1px solid #ccc;display:flex;align-items:center;gap:10px;font-size:13px;">'
        '\n  <button id="btnOpenVideo" style="padding:5px 16px;background:#36a;color:#fff;'
        'border:none;border-radius:4px;cursor:pointer;font-size:13px;font-weight:bold;">Open Video</button>'
        '\n  <span id="loadStatus" style="color:#555;font-size:12px;"></span>'
        '\n</div>'
        '\n<script>'
        '\n(function(){'
        '\n  var btn=document.getElementById("btnOpenVideo");'
        '\n  var st=document.getElementById("loadStatus");'
        '\n  if(!btn) return;'
        '\n  btn.addEventListener("click",function(e){'
        '\n    e.stopPropagation();'
        '\n    btn.disabled=true; st.textContent="Opening file picker...";'
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
    # Insert the open-video bar right after <body>
    html = html.replace("<body>", "<body>\n" + open_bar, 1)

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

  // ---- Enlarged playhead hit test (24px) ----
  function hitPlayhead(rx){
    var plotW=c.width-2*pad;
    if(plotW<1||!vid||typeof playheadT==='undefined') return false;
    var vst=D.video_start_time||0;
    var pt=playheadT-vst;
    if(pt<t0||pt>t1) return false;
    var px=pad+(pt-t0)/(t1-t0)*plotW;
    return Math.abs(rx-px)<=24;
  }
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
  var THRESH=5;

  // ---- mousedown ----
  wrap.addEventListener('mousedown', function(e){
    if(e.button!==0) return;
    // Ignore clicks that originate inside the toolbar (buttons, etc.)
    var tb=document.getElementById('waveformToolbar');
    if(tb && tb.contains(e.target)) return;
    _startX=e.clientX; _moved=false;
    var rect=c.getBoundingClientRect();
    var rx=e.clientX-rect.left;
    if(vid && hitPlayhead(rx)){
      _mode='playhead'; return;
    }
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

    if(_mode==='playhead' && vid){
      // Release playhead → play from here
      var t=xToT(rx); if(t!==null) seekAndPlay(t);
    } else if(_mode==='line'){
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

    var toolbar=document.getElementById('waveformToolbar');
    if(toolbar){
      // Hide the original two save buttons (replaced by Save All)
      var bs=document.getElementById('btnSave');
      var bb=document.getElementById('btnSaveBeep');
      if(bs) bs.style.display='none';
      if(bb) bb.style.display='none';
      // Create Save All button
      var btnAll=document.createElement('button');
      btnAll.type='button';
      btnAll.textContent='Save All';
      btnAll.style.cssText='padding:10px 20px;background:#c75;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:14px;font-weight:bold;';
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
      toolbar.insertBefore(btnAll, toolbar.firstChild);

      // Save path display
      var pathDiv = document.createElement('div');
      pathDiv.id = 'savePathDisplay';
      pathDiv.style.cssText = 'font-size:11px;color:#555;margin-top:4px;word-break:break-all;';
      toolbar.appendChild(pathDiv);
    }

    // Show current save paths (call this after D is updated too)
    window._updateSavePathDisplay = function(){
      var el = document.getElementById('savePathDisplay');
      if(!el) return;
      var lines = [];
      if(D.calibration_save_path) lines.push('shots: ' + D.calibration_save_path);
      if(D.calibration_beep_path) lines.push('beep:  ' + D.calibration_beep_path);
      el.textContent = lines.join('   |   ');
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


def _prepare_video(video_path, ffmpeg, force_detect=False):
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
    anno_dir = video_dir
    beep_key  = video_base.split("-")[0] if "-" in video_base else video_base
    cali_path = os.path.join(anno_dir, video_base + "cali.txt")
    beep_path = os.path.join(anno_dir, beep_key + "beep.txt")
    print(f"  cali_path: {cali_path}  exists={os.path.isfile(cali_path)}")
    print(f"  beep_path: {beep_path}  exists={os.path.isfile(beep_path)}")

    has_cali = os.path.isfile(cali_path) and not force_detect
    has_beep = os.path.isfile(beep_path) and not force_detect

    # Load whatever exists; detect whatever is missing
    if has_cali:
        with open(cali_path, encoding="utf-8") as f:
            shot_times = [float(l.strip()) for l in f if l.strip()]
        print(f"  Loaded *cali.txt: {len(shot_times)} shot(s)")
    if has_beep:
        with open(beep_path, encoding="utf-8") as f:
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
            shots = detect_shots(audio_mono, fps)
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
                          default_speed=0.5):
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
                        vpath, ffmpeg_cmd
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
    ap.add_argument("--video", required=True, help="Video file path (.mp4)")
    ap.add_argument("--speed", type=float, default=0.5,
                    help="Default playback speed (default 0.5)")
    ap.add_argument("--port", type=int, default=8765,
                    help="Local server port (default 8765)")
    ap.add_argument("--no-detect", action="store_true",
                    help="Skip machine detection; load existing *cali.txt if available")
    ap.add_argument("--out-dir", default=None,
                    help="HTML output directory (default: same as video)")
    args = ap.parse_args()

    ffmpeg = get_ffmpeg_cmd()
    if not ffmpeg:
        print("Error: ffmpeg not found.")
        return 1

    video = os.path.abspath(os.path.normpath(args.video))
    if not os.path.isfile(video):
        print(f"Error: video not found: {video}")
        return 1

    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(video)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("tmp", exist_ok=True)

    # ── 1. Process initial video ──────────────────────────────────────
    cal_data, video_dir, video_base, anno_dir = _prepare_video(
        video, ffmpeg, force_detect=args.no_detect
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
