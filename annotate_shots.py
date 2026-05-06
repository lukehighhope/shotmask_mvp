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
import argparse

# 把项目根目录加入 import 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_audio_plot import (
    write_calibration_viewer_html,
    run_calibration_server,
    get_waveform_data,
    get_audio_start_time,
    extract_audio as extract_audio_ch,
)
from detectors.beep import detect_beeps
from detectors.shot_audio import detect_shots
from main import get_ffmpeg_cmd, get_ffprobe_cmd, ffprobe_info
import subprocess
import json


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

    # ── 3. Replace ALL wrap mouse listeners with clean handlers ───────
    # Strategy: inject a script block that clones wrap (drops old listeners),
    # then re-attaches wheel + clean mousedown/mousemove/mouseup.
    # Also inject a "Save All" button that triggers both cali + beep saves.
    patch_js = """
<script>
(function(){
  var toolbar = document.getElementById('waveformToolbar');
  if(toolbar){
    var btnAll = document.createElement('button');
    btnAll.type = 'button';
    btnAll.textContent = 'Save All';
    btnAll.style.cssText = 'padding:10px 20px;background:#c75;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:14px;font-weight:bold;margin-left:10px;';
    btnAll.addEventListener('mousedown', function(e){ e.stopPropagation(); });
    btnAll.addEventListener('mouseup',   function(e){ e.stopPropagation(); });
    btnAll.addEventListener('click', function(e){
      e.preventDefault(); e.stopPropagation();
      var bs = document.getElementById('btnSave');
      var bb = document.getElementById('btnSaveBeep');
      if(bs) bs.click();
      if(bb) bb.click();
    });
    toolbar.appendChild(btnAll);
  }
})();
</script>
"""
    html = html.replace("</body>", patch_js + "\n</body>", 1)

    # ── 4. Mouse handlers patch ───────────────────────────────────────
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
    var vst=D.video_start_time||0;
    vid.currentTime=t+vst;
    playheadT=t+vst;
    if(timeStr) timeStr.textContent=vid.currentTime.toFixed(2)+' / '+(vid.duration||0).toFixed(2)+' s';
    vid.play();
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
        if(Math.abs(rx-px)<=10){ _mode='line'; _lineIdx=i; return; }
      }
      for(var i=0;i<calibrationBeepTimes.length;i++){
        var px=pad+(calibrationBeepTimes[i]-t0)/(t1-t0)*plotW;
        if(Math.abs(rx-px)<=10){ _mode='beep'; _beepIdx=i; return; }
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
      var t=xToT(rx); if(t!==null){ calibrationShots[_lineIdx]=t; draw(); }
      return;
    }
    if(_mode==='beep'){
      var t=xToT(rx);
      if(t!==null){ calibrationBeepTimes[_beepIdx]=t; calibrationBeepTimes.sort(function(a,b){return a-b;}); draw(); }
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

  draw();
})();
</script>
"""
    html = html.replace("</body>", patch_js + "\n</body>", 1)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


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

    video = os.path.abspath(os.path.normpath(args.video))
    if not os.path.isfile(video):
        print(f"错误：找不到视频文件: {video}")
        return 1

    video_dir = os.path.dirname(video)
    video_base = os.path.splitext(os.path.basename(video))[0]
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else video_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. 提取音频 ──────────────────────────────────────────────────
    ffmpeg = get_ffmpeg_cmd()
    if not ffmpeg:
        print("Error: ffmpeg not found. Please install ffmpeg first.")
        return 1

    print(f"Video: {video}")
    print("Extracting audio...")
    audio_mono   = extract_audio_ch(video, ffmpeg, "tmp/audio_ann_mono.wav",   channels=1)
    audio_stereo = extract_audio_ch(video, ffmpeg, "tmp/audio_ann_stereo.wav", channels=2)
    fps, duration = ffprobe_info(video)
    print(f"FPS={fps}  duration={duration:.2f}s")

    # ── 2. 载入已有标注 或 机器检测 ──────────────────────────────────
    beep_times = []
    shot_times = []

    cali_path = os.path.join(video_dir, video_base + "cali.txt")
    beep_key  = video_base.split("-")[0] if "-" in video_base else video_base
    beep_path = os.path.join(video_dir, beep_key + "beep.txt")

    has_cali = os.path.isfile(cali_path)
    has_beep = os.path.isfile(beep_path)

    if (has_cali or has_beep) and not args.no_detect:
        # Existing annotation files found — load them directly
        if has_cali:
            with open(cali_path, encoding="utf-8") as f:
                shot_times = [float(l.strip()) for l in f if l.strip()]
            print(f"  Loaded existing *cali.txt: {len(shot_times)} shot(s)")
        if has_beep:
            with open(beep_path, encoding="utf-8") as f:
                beep_times = [float(l.strip()) for l in f if l.strip()]
            print(f"  Loaded existing *beep.txt: {beep_times}")
    elif not args.no_detect:
        print("Detecting beep...")
        try:
            beeps = detect_beeps(audio_mono, fps)
            beep_times = [float(b["t"]) for b in beeps]
            print(f"  Detected {len(beep_times)} beep(s): {beep_times}")
        except Exception as e:
            print(f"  Beep detection failed: {e}")

        print("Detecting gunshots...")
        try:
            t0_beep = beep_times[-1] if beep_times else 0.0
            shots = detect_shots(audio_mono, fps)
            shots = [s for s in shots if float(s["t"]) >= t0_beep]
            shot_times = [float(s["t"]) for s in shots]
            print(f"  Detected {len(shot_times)} shot(s)")
        except Exception as e:
            print(f"  Shot detection failed: {e}")
    else:
        # --no-detect: load existing files only
        if has_cali:
            with open(cali_path, encoding="utf-8") as f:
                shot_times = [float(l.strip()) for l in f if l.strip()]
            print(f"  Loaded *cali.txt: {len(shot_times)} shot(s)")
        if has_beep:
            with open(beep_path, encoding="utf-8") as f:
                beep_times = [float(l.strip()) for l in f if l.strip()]
            print(f"  Loaded *beep.txt: {beep_times}")

    # ── 3. 构建波形数据 + 生成 HTML ───────────────────────────────────
    print("Building waveform data...")
    wdata = get_waveform_data(
        audio_stereo,
        beep_times=beep_times,
        shot_times=shot_times,
        ref_shot_times=[],
    )

    cal_data = dict(wdata)
    cal_data["calibration_shots"] = list(shot_times)
    cal_data["video_base_name"] = video_base

    html_name = video_base + "_annotate.html"
    html_path = os.path.join(out_dir, html_name)

    write_calibration_viewer_html(html_path, cal_data, video_path=video)

    # ── 4. 注入 0.5x 默认速度 + 速度按钮 ─────────────────────────────
    _patch_html_playback_speed(html_path, default_speed=args.speed)
    print(f"\nHTML ready: {html_path}")
    print(f"Default playback speed: {args.speed}x")

    print(f"\nStarting local server on port {args.port}...")
    print("When done: click [Save Calibration] -> *cali.txt  |  [Save Beep] -> *beep.txt")
    print("Press Ctrl+C to stop the server\n")
    run_calibration_server(
        out_dir,
        port=args.port,
        html_basename=html_name,
        allowed_save_dir=video_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
