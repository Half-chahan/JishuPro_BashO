#!/usr/bin/env python3
import os
import json
import subprocess
import tempfile
import random
import math
import struct
import wave
import shutil

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import uvicorn

import rospy
from haiku_msgs.msg import HaikuResult, LineArt

from openai import OpenAI  # 新しい公式SDK

# ========= OpenAI クライアント =========

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("環境変数 OPENAI_API_KEY が設定されていません。")
    return OpenAI(api_key=api_key)

client = get_openai_client()

# ========= ROS1（rospy）初期化 =========

rospy.init_node("haiku_web_gpt_node", anonymous=True)
publisher = rospy.Publisher("haiku_result", HaikuResult, queue_size=10)
lineart_publisher = rospy.Publisher("lineart", LineArt, queue_size=10)
lineart_joint_publisher = rospy.Publisher("lineart_joint", LineArt, queue_size=10)
lineart_joint_snap_publisher = rospy.Publisher("lineart_joint_snap", LineArt, queue_size=10)
rospy.loginfo("haiku_web_gpt_node started, publishing to 'haiku_result'.")

# ========= Web アプリ (FastAPI) =========

app = FastAPI()

INDEX_TEMPLATE = """
<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>モード選択</title>
<style>
 :root {
   --bg: #e9eef4;
   --ink: #132131;
   --accent: #1b3a57;
   --accent-2: #295b8a;
   --paper: #fdfcff;
   --line: #b8c8d9;
 }
 body {
   font-family: "Hiragino Mincho ProN", "Yu Mincho", "Noto Serif JP", serif;
   margin: 0;
   color: var(--ink);
   background:
     radial-gradient(1200px 600px at 80% -10%, #ffffff 0%, transparent 60%),
     radial-gradient(900px 500px at -10% 20%, #edf4fb 0%, transparent 60%),
     var(--bg);
 }
 .page {
   min-height: 100vh;
   padding: 48px 20px 80px;
 }
 .box {
   max-width: 980px;
   margin: 0 auto;
 }
 .hero {
   display: grid;
   grid-template-columns: 1.1fr 1fr;
   gap: 24px;
   align-items: center;
   background: var(--paper);
   border: 1px solid var(--line);
   padding: 28px;
   border-radius: 16px;
   box-shadow: 0 14px 40px rgba(0,0,0,0.08);
   position: relative;
   overflow: hidden;
 }
 .hero::after {
   content: "";
   position: absolute;
   right: -60px;
   top: -60px;
   width: 180px;
   height: 180px;
   border-radius: 50%;
   background: radial-gradient(circle at 30% 30%, rgba(154,47,47,0.12), transparent 65%);
 }
 h1 {
   font-size: 28px;
   margin: 0 0 8px;
   letter-spacing: 0.08em;
 }
 .lead {
   font-family: "Noto Sans JP", "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
   color: #4d433b;
   line-height: 1.7;
 }
 .modes {
   display: grid;
   grid-template-columns: repeat(2, minmax(0, 1fr));
   gap: 22px;
 }
 .card {
   display: block;
   text-decoration: none;
   color: var(--ink);
   border: 1px solid var(--line);
   border-radius: 16px;
   padding: 22px 20px;
   background: #fff;
   transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
   min-height: 120px;
   display: flex;
   align-items: center;
   justify-content: space-between;
   gap: 14px;
   text-align: center;
 }
 .card:hover {
   transform: translateY(-2px);
   border-color: #c7b7a6;
   box-shadow: 0 12px 26px rgba(0,0,0,0.08);
 }
 .card-title {
   font-size: 14px;
   letter-spacing: 0.08em;
   margin: 0;
   white-space: nowrap;
}
 .card-sub {
   font-family: "Noto Sans JP", "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
   color: #3a4d63;
   font-size: 10px;
   line-height: 1.2;
   white-space: nowrap;
   overflow: hidden;
   text-overflow: ellipsis;
 }
 .tag {
   display: inline-block;
   padding: 4px 10px;
   border-radius: 999px;
   font-size: 12px;
   background: #dfeaf7;
   color: #1b3a57;
   margin-bottom: 10px;
 }
 @keyframes fadeUp {
   from { opacity: 0; transform: translateY(8px); }
   to { opacity: 1; transform: translateY(0); }
 }
 .hero, .card { animation: fadeUp 360ms ease both; }
 @media (max-width: 860px) {
   .hero { grid-template-columns: 1fr; }
   .modes { grid-template-columns: 1fr; }
 }
</style>
</head>
<body>
<div class="page">
  <div class="box">
    <div class="hero">
      <div>
        <div class="tag">HAIKU STUDIO</div>
        <h1>俳句評価およびロボットアーム描画統合システムBash-O</h1>
        <p class="lead">俳句の評価、顔からの線画、俳句からの線画を、選んだモードごとに起動します。</p>
      </div>
      <div class="modes">
        <a class="card" href="/haiku">
          <div class="card-title">俳句の評価</div>
          <div class="card-sub">松尾芭蕉の眼で講評</div>
        </a>
        <a class="card" href="/face">
          <div class="card-title">顔認識から線画</div>
          <div class="card-sub">撮影して輪郭抽出</div>
        </a>
        <a class="card" href="/face_joint">
          <div class="card-title">顔線画（角度指定）</div>
          <div class="card-sub">関節角で直接描画</div>
        </a>
        <a class="card" href="/face_joint_snap">
          <div class="card-title">顔線画（角度指定・スナップ）</div>
          <div class="card-sub">格子点優先で描画</div>
        </a>
        <a class="card" href="/haiku_lineart">
          <div class="card-title">俳句から線画</div>
          <div class="card-sub">AIが俳句を線の集合へ変換</div>
        </a>
        <a class="card" href="/haiku_lineart_joint">
          <div class="card-title">俳句線画（角度指定）</div>
          <div class="card-sub">関節角で直接描画</div>
        </a>
        <a class="card" href="/haiku_lineart_joint_snap">
          <div class="card-title">俳句線画（角度指定・スナップ）</div>
          <div class="card-sub">格子点優先で描画</div>
        </a>
        <a class="card" href="/quiz">
          <div class="card-title">穴埋めクイズ</div>
          <div class="card-sub">名句の7文字を補う</div>
        </a>
        <a class="card" href="/lineart_quiz">
          <div class="card-title">名句線画クイズ</div>
          <div class="card-sub">3択で線画を当てる</div>
        </a>
        <a class="card" href="/music_draw">
          <div class="card-title">音列ドロー</div>
          <div class="card-sub">作曲して線を描く</div>
        </a>
      </div>
    </div>
    <div class="panel" style="margin-top:18px;padding:16px 20px;">
      <div class="badge">操作パネル</div>
      <div style="display:flex;gap:10px;flex-wrap:wrap;">
        <form method="post" action="/control/reset">
          <button class="btn" type="submit">リセット姿勢</button>
        </form>
        <form method="post" action="/control/pose">
          <button class="btn" type="submit">指定角度姿勢</button>
        </form>
      </div>
    </div>
  </div>
</div>
</body>
</html>
"""

HAIKU_TEMPLATE = """
<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>俳句評価システム</title>
<style>
 :root {{
   --bg: #e9eef4;
   --ink: #132131;
   --accent: #1b3a57;
   --accent-2: #295b8a;
   --paper: #fdfcff;
   --line: #b8c8d9;
 }}
 body {{
   font-family: "Hiragino Mincho ProN", "Yu Mincho", "Noto Serif JP", serif;
   margin: 0;
   color: var(--ink);
   background:
     linear-gradient(180deg, rgba(255,255,255,0.92), rgba(233,238,244,0.92)),
     var(--bg);
 }}
 .page {{ padding: 48px 20px 80px; }}
 .box {{ max-width: 980px; margin: 0 auto; }}
 .panel {{
   background: var(--paper);
   border: 1px solid var(--line);
   border-radius: 16px;
   box-shadow: 0 14px 40px rgba(0,0,0,0.08);
   overflow: hidden;
 }}
 .header {{
   padding: 22px 26px;
   border-bottom: 1px solid var(--line);
   display: flex;
   justify-content: space-between;
   align-items: baseline;
 }}
 h1 {{
   font-size: 24px;
   margin: 0;
   letter-spacing: 0.08em;
 }}
 .hint {{
   font-family: "Noto Sans JP", "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
   font-size: 12px;
   color: #6b6158;
 }}
 .content {{
   padding: 22px 26px 26px;
   display: grid;
   grid-template-columns: 1.1fr 1fr;
   gap: 20px;
 }}
 textarea {{
   width: 100%;
   height: 140px;
   background: #fff;
   border: 1px solid var(--line);
   border-radius: 12px;
   padding: 12px;
   font-family: "Noto Sans JP", "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
 }}
 .section {{ border: 1px solid var(--line); border-radius: 12px; padding: 16px; background: #fff; }}
 .score {{
   font-size: 1.4em;
   font-weight: bold;
   color: var(--accent);
 }}
 .badge {{
   display: inline-block;
   padding: 4px 10px;
   border-radius: 999px;
   background: #dfeaf7;
   color: #1b3a57;
   font-size: 12px;
   letter-spacing: 0.08em;
 }}
 .error {{ color: #b33a3a; }}
 .actions {{
   margin-top: 10px;
   display: flex;
   gap: 8px;
 }}
 .btn {{
   padding: 10px 16px;
   border-radius: 10px;
   border: 1px solid var(--line);
   background: #fff;
   font-family: "Noto Sans JP", "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
   cursor: pointer;
 }}
 .btn.primary {{
   background: var(--accent-2);
   border-color: var(--accent-2);
   color: #fff;
 }}
 .footer {{
   padding: 0 26px 22px;
   font-family: "Noto Sans JP", "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
 }}
 @media (max-width: 860px) {{
   .content {{ grid-template-columns: 1fr; }}
 }}
</style>
</head>
<body>
<div class="page">
  <div class="box">
    <div class="panel">
      <div class="header">
        <h1>俳句評価</h1>
        <div class="hint">松尾芭蕉の眼で講評</div>
      </div>
      <div class="content">
        <div class="section">
          <div class="badge">INPUT</div>
          <form method="post" action="/haiku">
            <p><textarea name="haiku" placeholder="古池や 蛙飛びこむ 水の音">{haiku}</textarea></p>
            <div class="actions">
              <button class="btn primary" type="submit">評価する</button>
              <a class="btn" href="/">戻る</a>
            </div>
          </form>
          {error_block}
        </div>
        <div class="section">
          <div class="badge">RESULT</div>
          {result_block}
        </div>
      </div>
      <div class="footer"></div>
    </div>
  </div>
</div>
</body>
</html>
"""

LINEART_TEMPLATE = """
<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
 :root {{
   --bg: #e9eef4;
   --ink: #132131;
   --accent: #1b3a57;
   --accent-2: #295b8a;
   --paper: #fdfcff;
   --line: #b8c8d9;
 }}
 body {{
   font-family: "Hiragino Mincho ProN", "Yu Mincho", "Noto Serif JP", serif;
   margin: 0;
   color: var(--ink);
   background:
     radial-gradient(900px 500px at 90% -10%, #ffffff 0%, transparent 60%),
     radial-gradient(700px 420px at 10% 20%, #e8f1fb 0%, transparent 60%),
     var(--bg);
 }}
 .page {{ padding: 48px 20px 80px; }}
 .box {{ max-width: 980px; margin: 0 auto; }}
 .panel {{
   background: var(--paper);
   border: 1px solid var(--line);
   border-radius: 16px;
   box-shadow: 0 14px 40px rgba(0,0,0,0.08);
   overflow: hidden;
 }}
 .header {{
   padding: 22px 26px;
   border-bottom: 1px solid var(--line);
 }}
 h1 {{
   font-size: 24px;
   margin: 0;
   letter-spacing: 0.08em;
 }}
 .section {{
   margin: 18px 22px;
   padding: 16px;
   border: 1px solid var(--line);
   border-radius: 12px;
   background: #fff;
 }}
 textarea {{
   width: 100%;
   height: 140px;
   background: #fff;
   border: 1px solid var(--line);
   border-radius: 12px;
   padding: 12px;
   font-family: "Noto Sans JP", "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
 }}
 .badge {{
   display: inline-block;
   padding: 4px 10px;
   border-radius: 999px;
   background: #dfeaf7;
   color: #1b3a57;
   font-size: 12px;
   letter-spacing: 0.08em;
   margin-bottom: 8px;
 }}
 .error {{ color: #b33a3a; }}
 .actions {{
   margin-top: 10px;
   display: flex;
   gap: 8px;
 }}
 .btn {{
   padding: 10px 16px;
   border-radius: 10px;
   border: 1px solid var(--line);
   background: #fff;
   font-family: "Noto Sans JP", "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
   cursor: pointer;
 }}
 .btn.primary {{
   background: var(--accent-2);
   border-color: var(--accent-2);
   color: #fff;
 }}
</style>
</head>
<body>
<div class="page">
  <div class="box">
    <div class="panel">
      <div class="header">
        <h1>{title}</h1>
      </div>
      <div class="section">
        <div class="badge">INPUT</div>
        {form_block}
        {error_block}
      </div>
      <div class="section">
        <div class="badge">RESULT</div>
        {result_block}
      </div>
      <div class="section">
        <div class="actions">
          <a class="btn" href="/">戻る</a>
        </div>
      </div>
    </div>
  </div>
</div>
</body>
</html>
"""

def simple_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )

def render_haiku_page(haiku: str = "", score=None, comment: str = None, revised: str = None, error: str = None):
    if error:
        error_block = f'<p class="error">{simple_escape(error)}</p>'
    else:
        error_block = ""

    if score is None:
        result_block = ""
    else:
        haiku_html = simple_escape(haiku)
        comment_html = simple_escape(comment)
        revised_html = simple_escape(revised)

        result_block = f"""
        <div class="section">
          <p><b>入力された俳句：</b><br>{haiku_html}</p>
          <p><b>点数：</b> <span class="score">{score} / 3</span></p>
          <p><b>講評：</b><br>{comment_html}</p>
          <p><b>添削後の句：</b><br>{revised_html}</p>
        </div>
        """

    return HAIKU_TEMPLATE.format(
        haiku=haiku,
        error_block=error_block,
        result_block=result_block,
    )

def render_lineart_page(title: str, form_block: str, result: str = None, error: str = None):
    if error:
        error_block = f'<p class="error">{simple_escape(error)}</p>'
    else:
        error_block = ""
    result_block = result or ""
    return LINEART_TEMPLATE.format(
        title=simple_escape(title),
        form_block=form_block,
        error_block=error_block,
        result_block=result_block,
    )

def normalize_quiz_text(s: str) -> str:
    s = (
        s.replace(" ", "")
         .replace("　", "")
         .replace("\n", "")
         .strip()
    )
    out = []
    for ch in s:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        else:
            out.append(ch)
    return "".join(out)

def normalize_note_text(s: str) -> str:
    return (
        s.replace(" ", "")
         .replace("　", "")
         .replace("\n", "")
         .replace("⇒", ">")
         .replace("→", ">")
         .strip()
         .lower()
    )

def parse_note_token(tok: str):
    tok = normalize_note_text(tok)
    if not tok:
        return None
    jp_map = {
        "ド": "do",
        "レ": "re",
        "ミ": "mi",
        "ファ": "fa",
        "ソ": "so",
        "ラ": "la",
        "シ": "ti",
    }
    for k, v in jp_map.items():
        tok = tok.replace(k, v)
    tok = tok.replace("sol", "so")
    sharp = "#" in tok or "♯" in tok or "sharp" in tok
    for ch in ["#", "♯"]:
        tok = tok.replace(ch, "")
    octave = None
    digits = "".join([c for c in tok if c.isdigit()])
    if digits:
        octave = int(digits)
        tok = "".join([c for c in tok if not c.isdigit()])
    base = tok
    base_map = {"do": 0, "re": 2, "mi": 4, "fa": 5, "so": 7, "la": 9, "ti": 11}
    if base not in base_map:
        return None
    semitone = base_map[base] + (1 if sharp else 0)
    if octave is None:
        octave = 4
    return semitone, octave

def notes_to_frequencies(notes):
    freqs = []
    for n in notes:
        parsed = parse_note_token(n)
        if not parsed:
            continue
        semitone, octave = parsed
        midi = (octave + 1) * 12 + semitone
        freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        freqs.append(freq)
    return freqs

def generate_wav(freqs, bpm, out_path):
    if not freqs:
        return None
    sr = 22050
    beat = 60.0 / max(1.0, float(bpm))
    dur = beat * 0.8
    gap = beat * 0.2
    samples = []
    for f in freqs:
        n = int(sr * dur)
        for i in range(n):
            t = i / sr
            val = 0.2 * math.sin(2 * math.pi * f * t)
            samples.append(int(val * 32767))
        gap_n = int(sr * gap)
        samples.extend([0] * gap_n)
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack("<" + "h" * len(samples), *samples))
    return out_path

def try_play_audio(path):
    player = None
    for cmd in ["aplay", "paplay", "afplay"]:
        if shutil.which(cmd):
            player = cmd
            break
    if not player:
        return False
    subprocess.run([player, path], check=False)
    return True

def build_lineart_from_notes(freqs, width=100, height=100):
    if not freqs:
        return None
    n = len(freqs)
    fmin = min(freqs)
    fmax = max(freqs)
    if fmax <= fmin:
        fmax = fmin + 1.0
    pts = []
    for i, f in enumerate(freqs):
        x = 0 if n == 1 else (width * i / (n - 1))
        y = height - ((f - fmin) / (fmax - fmin) * height)
        pts.append((x, y))
    if len(pts) == 1:
        pts.append((pts[0][0] + 1.0, pts[0][1]))
    return {
        "width": int(width),
        "height": int(height),
        "polylines": [pts],
    }

def build_preview_from_polylines(data):
    polylines = data.get("polylines", [])
    width = int(data.get("width", 0))
    height = int(data.get("height", 0))
    xy = []
    counts = []
    for pl in polylines:
        if len(pl) < 2:
            continue
        counts.append(len(pl))
        for x, y in pl:
            xy.append(float(x))
            xy.append(float(y))
    return {"width": width, "height": height, "xy": xy, "counts": counts}

def call_gpt_notes(prompt_text: str):
    system_prompt = (
        "あなたは作曲者です。"
        "単なるドレミ順列ではなく、動機（短いフレーズ）と反復、跳躍、緩急を含む"
        "“音楽らしい”8〜16音のメロディを作ってください。"
        "同じ音の連続は最大2回まで。音域は1オクターブ半以内。"
        "出力は JSON のみ。"
        "notes はドレミ表記で、例: do4,re4,mi4,fa4,so4,la4,ti4。"
        "bpm は 60-160 の整数。"
        "reason は松尾芭蕉になりきって作曲意図を説明する日本語文。"
        "{\"notes\":[\"do4\",\"so4\"],\"bpm\":120,\"reason\":\"...\"}"
    )
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ],
        response_format={"type": "json_object"},
        temperature=0.6,
    )
    content = response.choices[0].message.content
    data = json.loads(content)
    notes = data.get("notes", [])
    bpm = int(data.get("bpm", 120))
    reason = str(data.get("reason", "")).strip()
    return notes, bpm, reason

QUIZ_QUESTIONS = [
    {
        "full": "ふるいけや かわずとびこむ みずのおと",
        "masked": "ふるいけや 〇〇〇〇〇〇〇 みずのおと",
        "answer": "かわずとびこむ",
    },
    {
        "full": "しずけさや いわにしみいる せみのこえ",
        "masked": "しずけさや 〇〇〇〇〇〇〇 せみのこえ",
        "answer": "いわにしみいる",
    },
    {
        "full": "さみだれを あつめてはやし もがみがわ",
        "masked": "さみだれを 〇〇〇〇〇〇〇 もがみがわ",
        "answer": "あつめてはやし",
    },
    {
        "full": "なつくさや つわものどもが ゆめのあと",
        "masked": "なつくさや 〇〇〇〇〇〇〇 ゆめのあと",
        "answer": "つわものどもが",
    },
    {
        "full": "あらうみや さどによこたう あまのがわ",
        "masked": "あらうみや 〇〇〇〇〇〇〇 あまのがわ",
        "answer": "さどによこたう",
    },
]

def maybe_scale_polylines(polylines, width, height):
    if width <= 1 or height <= 1:
        return polylines
    all_norm = True
    for pl in polylines:
        for x, y in pl:
            if x < 0.0 or y < 0.0 or x > 1.0 or y > 1.0:
                all_norm = False
                break
        if not all_norm:
            break
    if not all_norm:
        return polylines
    return [[(x * width, y * height) for x, y in pl] for pl in polylines]

def scale_polylines_to_size(polylines, src_width, src_height, dst_width, dst_height):
    sw = float(src_width)
    sh = float(src_height)
    dw = float(dst_width)
    dh = float(dst_height)
    if sw <= 0.0 or sh <= 0.0 or dw <= 0.0 or dh <= 0.0:
        return polylines
    scale = min(dw / sw, dh / sh)
    xoff = (dw - (sw * scale)) * 0.5
    yoff = (dh - (sh * scale)) * 0.5
    out = []
    for pl in polylines:
        if not pl:
            continue
        pts = []
        for x, y in pl:
            pts.append((x * scale + xoff, y * scale + yoff))
        out.append(pts)
    return out

def publish_lineart(width, height, polylines):
    polylines = [pl for pl in polylines if pl and len(pl) >= 2]
    polylines = maybe_scale_polylines(polylines, width, height)

    xy = []
    counts = []
    for pl in polylines:
        counts.append(len(pl))
        for x, y in pl:
            xy.append(float(x))
            xy.append(float(y))

    msg = LineArt()
    msg.width = int(width)
    msg.height = int(height)
    msg.xy = xy
    msg.counts = counts
    lineart_publisher.publish(msg)
    rospy.loginfo("Published LineArt(polylines=%d, points=%d)", len(counts), int(len(xy) / 2))

def publish_lineart_joint(width, height, polylines):
    polylines = [pl for pl in polylines if pl and len(pl) >= 2]
    polylines = maybe_scale_polylines(polylines, width, height)

    xy = []
    counts = []
    for pl in polylines:
        counts.append(len(pl))
        for x, y in pl:
            xy.append(float(x))
            xy.append(float(y))

    msg = LineArt()
    msg.width = int(width)
    msg.height = int(height)
    msg.xy = xy
    msg.counts = counts
    lineart_joint_publisher.publish(msg)
    rospy.loginfo("Published LineArtJoint(polylines=%d, points=%d)", len(counts), int(len(xy) / 2))

def publish_lineart_joint_snap(width, height, polylines):
    polylines = [pl for pl in polylines if pl and len(pl) >= 2]
    polylines = maybe_scale_polylines(polylines, width, height)

    xy = []
    counts = []
    for pl in polylines:
        counts.append(len(pl))
        for x, y in pl:
            xy.append(float(x))
            xy.append(float(y))

    msg = LineArt()
    msg.width = int(width)
    msg.height = int(height)
    msg.xy = xy
    msg.counts = counts
    lineart_joint_snap_publisher.publish(msg)
    rospy.loginfo("Published LineArtJointSnap(polylines=%d, points=%d)", len(counts), int(len(xy) / 2))

def run_haiku2lineart(haiku_text: str):
    script_path = "/home/mech-user/LinImg/haiku2lineart.py"
    with tempfile.NamedTemporaryFile(prefix="lineart_", suffix=".json", delete=False) as tf:
        out_path = tf.name
    cmd = [
        "python3",
        script_path,
        "--haiku",
        haiku_text,
        "--out",
        out_path,
        "--max_points",
        "60",
    ]
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=180, env=env)
    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def run_zedge4_lineart():
    script_path = "/home/mech-user/LinImg/zedge4.py"
    with tempfile.NamedTemporaryFile(prefix="lineart_", suffix=".json", delete=False) as tf:
        out_path = tf.name
    cmd = [
        "python3",
        script_path,
        "--out",
        out_path,
        "--no-gui",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=180)
    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def lineart_to_svg(data, viewbox=None, scale=4):
    width = int(data.get("width", 0))
    height = int(data.get("height", 0))
    xy = data.get("xy", [])
    counts = data.get("counts", [])
    if width <= 0 or height <= 0 or not counts:
        return "<p>プレビュー用の線画データが空です。</p>"

    if viewbox is None:
        viewbox = f"0 0 {width} {height}"
    svg_lines = [
        f'<svg viewBox="{viewbox}" width="{width * scale}" height="{height * scale}" '
        'xmlns="http://www.w3.org/2000/svg" '
        'style="max-width:100%;height:auto;background:#fff;border:1px solid #b8c8d9;border-radius:12px;">'
    ]
    idx = 0
    for c in counts:
        pts = []
        for _ in range(int(c)):
            if idx + 1 >= len(xy):
                break
            x = float(xy[idx])
            y = float(xy[idx + 1])
            pts.append(f"{x},{y}")
            idx += 2
        if len(pts) >= 2:
            svg_lines.append(
                f'<polyline points="{" ".join(pts)}" '
                'fill="none" stroke="#1b3a57" stroke-width="2"/>'
            )
    svg_lines.append("</svg>")
    return "\n".join(svg_lines)

def lineart_to_svg_colored(data, viewbox=None, scale=4):
    width = int(data.get("width", 0))
    height = int(data.get("height", 0))
    xy = data.get("xy", [])
    counts = data.get("counts", [])
    if width <= 0 or height <= 0 or not counts:
        return "<p>プレビュー用の線画データが空です。</p>", ""

    if viewbox is None:
        viewbox = f"0 0 {width} {height}"

    palette = [
        "#1b3a57", "#2b6c9f", "#3b8c7a", "#8d6b3a", "#a23e3e",
        "#6b4aa5", "#2f6f2f", "#9c3f7c", "#3a6a8a", "#6c7a89",
        "#b05a7a", "#5b7f9c",
    ]
    legend_items = []
    svg_lines = [
        f'<svg viewBox="{viewbox}" width="{width * scale}" height="{height * scale}" '
        'xmlns="http://www.w3.org/2000/svg" '
        'style="max-width:100%;height:auto;background:#fff;border:1px solid #b8c8d9;border-radius:12px;">'
    ]
    idx = 0
    for i, c in enumerate(counts):
        color = palette[i % len(palette)]
        pts = []
        for _ in range(int(c)):
            if idx + 1 >= len(xy):
                break
            x = float(xy[idx])
            y = float(xy[idx + 1])
            pts.append(f"{x},{y}")
            idx += 2
        if len(pts) >= 2:
            svg_lines.append(
                f'<polyline points="{" ".join(pts)}" '
                f'fill="none" stroke="{color}" stroke-width="2"/>'
            )
            legend_items.append(
                f'<div style="display:flex;align-items:center;gap:6px;'
                f'font-size:12px;color:#3a4d63;">'
                f'<span style="display:inline-block;width:10px;height:10px;'
                f'border-radius:50%;background:{color};"></span>'
                f'Line {i + 1}</div>'
            )
    svg_lines.append("</svg>")
    legend = (
        '<div style="display:flex;flex-wrap:wrap;gap:10px 14px;'
        'margin:8px 0 12px;">'
        + "".join(legend_items)
        + "</div>"
    )
    return "\n".join(svg_lines), legend

def snap_lineart_polylines(polylines, grid_step=10.0, snap_threshold=5.0, seg_threshold=30.0, grid_origin=None):
    def nearest_grid_value(v, origin):
        idx = math.floor((v - origin) / grid_step)
        g0 = origin + idx * grid_step
        g1 = g0 + grid_step
        return g0 if abs(v - g0) <= abs(v - g1) else g1

    def corner_points(x, y, origin):
        ox, oy = origin
        gx0 = ox + math.floor((x - ox) / grid_step) * grid_step
        gx1 = gx0 + grid_step
        gy0 = oy + math.floor((y - oy) / grid_step) * grid_step
        gy1 = gy0 + grid_step
        return ((gx0, gy0), (gx1, gy0), (gx0, gy1), (gx1, gy1))

    def append_unique(out, last, x, y):
        if last is None or abs(x - last[0]) > 1e-6 or abs(y - last[1]) > 1e-6:
            out.append((x, y))
            return (x, y)
        return last

    snapped = []
    for pl in polylines:
        if not pl:
            continue
        xs = [p[0] for p in pl]
        ys = [p[1] for p in pl]
        x_min = min(xs)
        y_min = min(ys)
        origin = grid_origin or (x_min, y_min)

        expanded = []
        last = None
        last = append_unique(expanded, last, pl[0][0], pl[0][1])
        for i in range(len(pl) - 1):
            x0, y0 = pl[i]
            x1, y1 = pl[i + 1]
            dx = x1 - x0
            dy = y1 - y0
            dist = math.hypot(dx, dy)
            if dist >= seg_threshold:
                n = int(dist // grid_step)
                for k in range(n):
                    frac = (k + 1.0) / (n + 1.0)
                    xi = x0 + dx * frac
                    yi = y0 + dy * frac
                    gx = nearest_grid_value(xi, origin[0])
                    gy = nearest_grid_value(yi, origin[1])
                    dd = math.hypot(xi - gx, yi - gy)
                    if dd <= snap_threshold:
                        last = append_unique(expanded, last, gx, gy)
            last = append_unique(expanded, last, x1, y1)

        snapped_pts = []
        for x, y in expanded:
            corners = corner_points(x, y, origin)
            best = corners[0]
            min_d = math.hypot(x - best[0], y - best[1])
            for gx, gy in corners[1:]:
                d = math.hypot(x - gx, y - gy)
                if d < min_d:
                    min_d = d
                    best = (gx, gy)
            if min_d < snap_threshold:
                snapped_pts.append(best)
            else:
                snapped_pts.append((x, y))

        if len(snapped_pts) < 2:
            if len(pl) >= 2:
                snapped.append(pl)
            elif len(pl) == 1:
                x0, y0 = pl[0]
                snapped.append([(x0, y0), (x0 + 1.0, y0)])
        else:
            snapped.append(snapped_pts)
    return snapped

def frange(start, stop, step):
    if step <= 0:
        return []
    vals = []
    v = start
    if start <= stop:
        while v <= stop + 1e-6:
            vals.append(v)
            v += step
    else:
        while v >= stop - 1e-6:
            vals.append(v)
            v -= step
    return vals

def transform_lineart_for_robot(data):
    width = float(data.get("width", 0))
    height = float(data.get("height", 0))
    xy = data.get("xy", [])
    counts = data.get("counts", [])
    if width <= 0.0 or height <= 0.0 or not counts:
        return None

    x_min, x_max = 120.0, 190.0
    y_min, y_max = -150.0, -50.0
    xr = x_max - x_min
    yr = y_max - y_min
    bounds = lineart_bounds(xy)
    if not bounds:
        return None
    xmin = bounds["min_x"]
    xmax = bounds["max_x"]
    ymin = bounds["min_y"]
    ymax = bounds["max_y"]
    bw = max(1.0, xmax - xmin)
    bh = max(1.0, ymax - ymin)

    scale = min(xr / bw, yr / bh)
    xoff = (xr - (bw * scale)) * 0.5
    yoff = (yr - (bh * scale)) * 0.5
    scale_r = min(xr / bh, yr / bw)
    xoff_r = (xr - (bh * scale_r)) * 0.5
    yoff_r = (yr - (bw * scale_r)) * 0.5
    rotate = scale_r > scale
    if rotate:
        scale = scale_r
        xoff = xoff_r
        yoff = yoff_r

    out_xy = []
    idx = 0
    for c in counts:
        for _ in range(int(c)):
            if idx + 1 >= len(xy):
                break
            x = float(xy[idx])
            y = float(xy[idx + 1])
            idx += 2
            bw = max(1.0, xmax - xmin)
            bh = max(1.0, ymax - ymin)
            if rotate:
                xf = y - ymin
                yf = x - xmin
                yf = bw - yf
            else:
                xf = x - xmin
                yf = bh - (y - ymin)
            xr_m = x_min + xoff + (xf * scale)
            yr_m = y_min + yoff + (yf * scale)
            out_xy.append(xr_m)
            out_xy.append(yr_m)
    return {
        "width": int(xr),
        "height": int(yr),
        "xy": out_xy,
        "counts": counts,
        "viewbox": f"{x_min} {y_min} {xr} {yr}",
        "scale": scale,
        "xoff": xoff,
        "yoff": yoff,
        "rotate": rotate,
    }

def lineart_bounds(xy):
    if not xy or len(xy) < 2:
        return None
    xs = xy[0::2]
    ys = xy[1::2]
    return {
        "min_x": min(xs),
        "max_x": max(xs),
        "min_y": min(ys),
        "max_y": max(ys),
    }

def build_polylines_from_xy(xy, counts):
    polylines = []
    idx = 0
    for c in counts:
        pts = []
        for _ in range(int(c)):
            if idx + 1 >= len(xy):
                break
            x = float(xy[idx])
            y = float(xy[idx + 1])
            pts.append((x, y))
            idx += 2
        if len(pts) >= 2:
            polylines.append(pts)
    return polylines

@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_TEMPLATE

@app.get("/haiku", response_class=HTMLResponse)
def haiku_index():
    return render_haiku_page()

def call_gpt(haiku_text: str):
    """
    俳句文字列を GPT に渡し、
    {"score":0-3, "comment":"...", "revised":"..."} を受け取る。
    """
    system_prompt = (
        "あなたは松尾芭蕉になりきって俳句を添削する人物です。"
        "入力された俳句に対して、0〜3点で採点し、講評と添削後の句を出してください。"
        "3点を出すのは松尾芭蕉の句のときのみとしてください。"
        "0点のときは激怒した講評にしてください。"
        "必ず JSON だけを出力し、以下の形式に厳密に従ってください："
        "{"
        "\"score\": 0〜3の整数,"
        "\"comment\": \"講評の日本語テキスト\","
        "\"revised\": \"添削後の俳句\""
        "}"
    )
    user_content = f"俳句: {haiku_text}"

    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    content = response.choices[0].message.content
    data = json.loads(content)  # {"score":..., "comment":..., "revised":...}

    score = int(data["score"])
    comment = str(data["comment"])
    revised = str(data["revised"])

    if score < 0 or score > 3:
        rospy.logwarn("GPTから想定外のscoreが返ってきました: %s", score)
        score = 0

    return score, comment, revised

@app.post("/haiku", response_class=HTMLResponse)
@app.post("/evaluate", response_class=HTMLResponse)
def evaluate(haiku: str = Form(...)):
    haiku = haiku.strip()
    if not haiku:
        return render_haiku_page(error="俳句を入力してください。")

    try:
        score, comment, revised = call_gpt(haiku)
    except Exception as e:
        return render_haiku_page(haiku=haiku, error=f"GPT呼び出しでエラー: {e}")

    # ROS1 に publish（EusLisp 側がこれを subscribe）
    msg = HaikuResult()
    msg.original = haiku
    msg.score = score
    msg.comment = comment
    msg.revised = revised
    publisher.publish(msg)
    rospy.loginfo("Published HaikuResult(score=%d)", score)

    # Web画面に結果表示
    return render_haiku_page(haiku=haiku, score=score, comment=comment, revised=revised)

@app.get("/haiku_lineart", response_class=HTMLResponse)
def haiku_lineart_index():
    form_block = """
    <form method="post" action="/haiku_lineart">
      <p>俳句を入力してください：</p>
      <p><textarea name="haiku" placeholder="古池や 蛙飛びこむ 水の音"></textarea></p>
      <p><button type="submit">線画生成して送信</button></p>
    </form>
    """
    return render_lineart_page(title="俳句から線画", form_block=form_block)

@app.post("/haiku_lineart", response_class=HTMLResponse)
def haiku_lineart(haiku: str = Form(...)):
    haiku = haiku.strip()
    if not haiku:
        return render_lineart_page(
            title="俳句から線画",
            form_block="",
            error="俳句を入力してください。",
        )
    try:
        data = run_haiku2lineart(haiku)
        width = int(data.get("width", 0))
        height = int(data.get("height", 0))
        polylines = data.get("polylines", [])
        reason = str(data.get("reason", "")).strip()
        polylines = maybe_scale_polylines(polylines, width, height)
        lineart_width = 70
        lineart_height = 100
        polylines = scale_polylines_to_size(
            polylines,
            width,
            height,
            lineart_width,
            lineart_height,
        )
        preview_xy = []
        preview_counts = []
        for pl in polylines:
            preview_counts.append(len(pl))
            for x, y in pl:
                preview_xy.append(float(x))
                preview_xy.append(float(y))
        preview_data = {
            "width": lineart_width,
            "height": lineart_height,
            "xy": preview_xy,
            "counts": preview_counts,
        }
        svg = lineart_to_svg(preview_data)
        colored_svg = ""
        legend = ""
        colored_svg, legend = lineart_to_svg_colored(preview_data)
        robot_data = transform_lineart_for_robot(preview_data)
        if robot_data:
            robot_svg = lineart_to_svg(
                robot_data,
                viewbox=robot_data["viewbox"],
                scale=4,
            )
            bounds = lineart_bounds(robot_data["xy"])
            if bounds:
                bounds_text = (
                    f"<p>bounds: "
                    f"left={bounds['min_x']:.2f} "
                    f"right={bounds['max_x']:.2f} "
                    f"top={bounds['max_y']:.2f} "
                    f"bottom={bounds['min_y']:.2f}</p>"
                )
            else:
                bounds_text = "<p>bounds: N/A</p>"
            robot_meta = (
                f"<p>scale={robot_data['scale']:.4f} "
                f"xoff={robot_data['xoff']:.2f} "
                f"yoff={robot_data['yoff']:.2f} "
                f"rotate={'yes' if robot_data.get('rotate') else 'no'}</p>"
            )
        else:
            robot_svg = "<p>ロボット用プレビューを生成できません。</p>"
            robot_meta = ""
            bounds_text = ""
        publish_lineart(lineart_width, lineart_height, polylines)
        total_points = sum(len(pl) for pl in polylines)
        reason_block = f"<p><b>描画説明（芭蕉風）</b><br>{simple_escape(reason) if reason else '（なし）'}</p>"
        snap_svg = ""
        if robot_data:
            snap_polylines = snap_lineart_polylines(
                build_polylines_from_xy(robot_data["xy"], robot_data["counts"]),
                10.0,
                5.0,
                30.0,
                grid_origin=(120.0, -150.0),
            )
            snap_preview = build_preview_from_polylines({
                "polylines": snap_polylines,
                "width": int(robot_data["width"]),
                "height": int(robot_data["height"]),
            })
            snap_svg = lineart_to_svg(
                snap_preview,
                viewbox=robot_data["viewbox"],
                scale=4,
            )
        result_block = (
            f"<p>線画を送信しました。polylines={len(polylines)} points={total_points}</p>"
            f"{reason_block}"
            f"<p><b>元の線画プレビュー</b></p>{svg}"
            f"<p><b>スプライン色分け</b></p>{legend}{colored_svg}"
            f"<p><b>格子点スナップ経路</b></p>{snap_svg}"
            f"<p><b>ロボット指令ベース</b></p>{robot_meta}{bounds_text}{robot_svg}"
        )
        return render_lineart_page(
            title="俳句から線画",
            form_block="",
            result=result_block,
        )
    except Exception as e:
        return render_lineart_page(
            title="俳句から線画",
            form_block="",
            error=f"線画生成でエラー: {e}",
        )

@app.get("/haiku_lineart_joint", response_class=HTMLResponse)
def haiku_lineart_joint_index():
    form_block = """
    <form method="post" action="/haiku_lineart_joint">
      <p>俳句を入力してください：</p>
      <p><textarea name="haiku" placeholder="古池や 蛙飛びこむ 水の音"></textarea></p>
      <p><button type="submit">角度指定で送信</button></p>
    </form>
    """
    return render_lineart_page(title="俳句線画（角度指定）", form_block=form_block)

@app.post("/haiku_lineart_joint", response_class=HTMLResponse)
def haiku_lineart_joint(haiku: str = Form(...)):
    haiku = haiku.strip()
    if not haiku:
        return render_lineart_page(
            title="俳句線画（角度指定）",
            form_block="",
            error="俳句を入力してください。",
        )
    try:
        data = run_haiku2lineart(haiku)
        width = int(data.get("width", 0))
        height = int(data.get("height", 0))
        polylines = data.get("polylines", [])
        reason = str(data.get("reason", "")).strip()
        polylines = maybe_scale_polylines(polylines, width, height)
        lineart_width = 70
        lineart_height = 100
        polylines = scale_polylines_to_size(
            polylines,
            width,
            height,
            lineart_width,
            lineart_height,
        )
        preview_xy = []
        preview_counts = []
        for pl in polylines:
            preview_counts.append(len(pl))
            for x, y in pl:
                preview_xy.append(float(x))
                preview_xy.append(float(y))
        preview_data = {
            "width": lineart_width,
            "height": lineart_height,
            "xy": preview_xy,
            "counts": preview_counts,
        }
        svg = lineart_to_svg(preview_data)
        colored_svg = ""
        legend = ""
        colored_svg, legend = lineart_to_svg_colored(preview_data)
        robot_data = transform_lineart_for_robot(preview_data)
        if robot_data:
            robot_svg = lineart_to_svg(
                robot_data,
                viewbox=robot_data["viewbox"],
                scale=4,
            )
            bounds = lineart_bounds(robot_data["xy"])
            if bounds:
                bounds_text = (
                    f"<p>bounds: "
                    f"left={bounds['min_x']:.2f} "
                    f"right={bounds['max_x']:.2f} "
                    f"top={bounds['max_y']:.2f} "
                    f"bottom={bounds['min_y']:.2f}</p>"
                )
            else:
                bounds_text = "<p>bounds: N/A</p>"
            robot_meta = (
                f"<p>scale={robot_data['scale']:.4f} "
                f"xoff={robot_data['xoff']:.2f} "
                f"yoff={robot_data['yoff']:.2f} "
                f"rotate={'yes' if robot_data.get('rotate') else 'no'}</p>"
            )
        else:
            robot_svg = "<p>ロボット用プレビューを生成できません。</p>"
            robot_meta = ""
            bounds_text = ""
        publish_lineart_joint(lineart_width, lineart_height, polylines)
        total_points = sum(len(pl) for pl in polylines)
        reason_block = f"<p><b>描画説明（芭蕉風）</b><br>{simple_escape(reason) if reason else '（なし）'}</p>"
        result_block = (
            f"<p>線画を送信しました。polylines={len(polylines)} points={total_points}</p>"
            f"{reason_block}"
            f"<p><b>元の線画プレビュー</b></p>{svg}"
            f"<p><b>スプライン色分け</b></p>{legend}{colored_svg}"
            f"<p><b>ロボット指令ベース</b></p>{robot_meta}{bounds_text}{robot_svg}"
        )
        return render_lineart_page(
            title="俳句線画（角度指定）",
            form_block="",
            result=result_block,
        )
    except Exception as e:
        return render_lineart_page(
            title="俳句線画（角度指定）",
            form_block="",
            error=f"線画生成でエラー: {e}",
        )

@app.get("/haiku_lineart_joint_snap", response_class=HTMLResponse)
def haiku_lineart_joint_snap_index():
    form_block = """
    <form method="post" action="/haiku_lineart_joint_snap">
      <p>俳句を入力してください：</p>
      <p><textarea name="haiku" placeholder="古池や 蛙飛びこむ 水の音"></textarea></p>
      <p><button type="submit">角度指定（スナップ）で送信</button></p>
    </form>
    """
    return render_lineart_page(title="俳句線画（角度指定・スナップ）", form_block=form_block)

@app.post("/haiku_lineart_joint_snap", response_class=HTMLResponse)
def haiku_lineart_joint_snap(haiku: str = Form(...)):
    haiku = haiku.strip()
    if not haiku:
        return render_lineart_page(
            title="俳句線画（角度指定・スナップ）",
            form_block="",
            error="俳句を入力してください。",
        )
    try:
        data = run_haiku2lineart(haiku)
        width = int(data.get("width", 0))
        height = int(data.get("height", 0))
        polylines = data.get("polylines", [])
        reason = str(data.get("reason", "")).strip()
        polylines = maybe_scale_polylines(polylines, width, height)
        lineart_width = 70
        lineart_height = 100
        polylines = scale_polylines_to_size(
            polylines,
            width,
            height,
            lineart_width,
            lineart_height,
        )
        preview_xy = []
        preview_counts = []
        for pl in polylines:
            preview_counts.append(len(pl))
            for x, y in pl:
                preview_xy.append(float(x))
                preview_xy.append(float(y))
        preview_data = {
            "width": lineart_width,
            "height": lineart_height,
            "xy": preview_xy,
            "counts": preview_counts,
        }
        svg = lineart_to_svg(preview_data)
        colored_svg, legend = lineart_to_svg_colored(preview_data)
        robot_data = transform_lineart_for_robot(preview_data)
        if robot_data:
            robot_svg = lineart_to_svg(
                robot_data,
                viewbox=robot_data["viewbox"],
                scale=4,
            )
            bounds = lineart_bounds(robot_data["xy"])
            if bounds:
                bounds_text = (
                    f"<p>bounds: "
                    f"left={bounds['min_x']:.2f} "
                    f"right={bounds['max_x']:.2f} "
                    f"top={bounds['max_y']:.2f} "
                    f"bottom={bounds['min_y']:.2f}</p>"
                )
            else:
                bounds_text = "<p>bounds: N/A</p>"
            robot_meta = (
                f"<p>scale={robot_data['scale']:.4f} "
                f"xoff={robot_data['xoff']:.2f} "
                f"yoff={robot_data['yoff']:.2f} "
                f"rotate={'yes' if robot_data.get('rotate') else 'no'}</p>"
            )
        else:
            robot_svg = "<p>ロボット用プレビューを生成できません。</p>"
            robot_meta = ""
            bounds_text = ""
        publish_lineart_joint_snap(lineart_width, lineart_height, polylines)
        total_points = sum(len(pl) for pl in polylines)
        reason_block = f"<p><b>描画説明（芭蕉風）</b><br>{simple_escape(reason) if reason else '（なし）'}</p>"
        result_block = (
            f"<p>線画を送信しました。polylines={len(polylines)} points={total_points}</p>"
            f"{reason_block}"
            f"<p><b>元の線画プレビュー</b></p>{svg}"
            f"<p><b>スプライン色分け</b></p>{legend}{colored_svg}"
            f"<p><b>ロボット指令ベース</b></p>{robot_meta}{bounds_text}{robot_svg}"
        )
        return render_lineart_page(
            title="俳句線画（角度指定・スナップ）",
            form_block="",
            result=result_block,
        )
    except Exception as e:
        return render_lineart_page(
            title="俳句線画（角度指定・スナップ）",
            form_block="",
            error=f"線画生成でエラー: {e}",
        )

@app.get("/quiz", response_class=HTMLResponse)
def quiz_index():
    q = random.choice(QUIZ_QUESTIONS)
    form_block = f"""
    <form method="post" action="/quiz">
      <p>次の俳句の〇〇〇〇〇〇〇を埋めてください：</p>
      <p><b>{q["masked"]}</b></p>
      <p><input type="text" name="answer" placeholder="7文字をひらがなで入力" style="width:100%;padding:10px;border:1px solid #b8c8d9;border-radius:8px;"></p>
      <input type="hidden" name="expected" value="{q["answer"]}">
      <input type="hidden" name="full" value="{q["full"]}">
      <p><button type="submit">解答する</button></p>
    </form>
    """
    return render_lineart_page(title="穴埋めクイズ", form_block=form_block)

@app.post("/quiz", response_class=HTMLResponse)
def quiz_answer(answer: str = Form(...), expected: str = Form(...), full: str = Form(...)):
    user = normalize_quiz_text(answer)
    exp = normalize_quiz_text(expected)
    is_ok = user == exp

    msg = HaikuResult()
    msg.original = full
    msg.score = 3 if is_ok else 0
    if is_ok:
        msg.comment = "正解。見事な解答です。"
        msg.revised = full
    else:
        msg.comment = "不正解。もう一度修行し直してこい。"
        msg.revised = full
    publisher.publish(msg)
    rospy.loginfo("Published HaikuResult(score=%d) from quiz", msg.score)

    result_block = f"""
    <div class="section">
      <p><b>問題：</b><br>{simple_escape(full)}</p>
      <p><b>あなたの解答：</b> {simple_escape(answer)}</p>
      <p><b>結果：</b> {"正解" if is_ok else "不正解"}</p>
    </div>
    """
    return render_lineart_page(title="穴埋めクイズ", form_block="", result=result_block)

@app.get("/lineart_quiz", response_class=HTMLResponse)
def lineart_quiz_index():
    choices = random.sample(QUIZ_QUESTIONS, k=3)
    correct = random.choice(choices)
    try:
        data = run_haiku2lineart(correct["full"])
        polylines = data.get("polylines", [])
        width = int(data.get("width", 0))
        height = int(data.get("height", 0))
        polylines = maybe_scale_polylines(polylines, width, height)
        preview_xy = []
        preview_counts = []
        for pl in polylines:
            preview_counts.append(len(pl))
            for x, y in pl:
                preview_xy.append(float(x))
                preview_xy.append(float(y))
        preview_data = {
            "width": width,
            "height": height,
            "xy": preview_xy,
            "counts": preview_counts,
        }
        svg = lineart_to_svg(preview_data)
    except Exception as e:
        svg = f"<p>線画生成でエラー: {simple_escape(str(e))}</p>"

    buttons = []
    for c in choices:
        buttons.append(
            f"""
            <form method="post" action="/lineart_quiz" style="margin: 6px 0;">
              <input type="hidden" name="selected" value="{c['full']}">
              <input type="hidden" name="correct" value="{correct['full']}">
              <p><button type="submit">{c['full']}</button></p>
            </form>
            """
        )
    form_block = f"""
    <p>次の線画に対応する句を3択から選んでください：</p>
    <div>{svg}</div>
    <div style="margin-top: 12px;">
      {''.join(buttons)}
    </div>
    """
    return render_lineart_page(title="名句線画クイズ", form_block=form_block)

@app.post("/lineart_quiz", response_class=HTMLResponse)
def lineart_quiz_answer(selected: str = Form(...), correct: str = Form(...)):
    is_ok = selected == correct
    msg = HaikuResult()
    msg.original = correct
    msg.score = 3 if is_ok else 0
    msg.comment = "正解。見事な解答です。" if is_ok else "不正解。もう一度修行し直してこい。"
    msg.revised = correct
    publisher.publish(msg)
    rospy.loginfo("Published HaikuResult(score=%d) from lineart_quiz", msg.score)

    result_block = f"""
    <div class="section">
      <p><b>正解の句：</b><br>{simple_escape(correct)}</p>
      <p><b>あなたの選択：</b><br>{simple_escape(selected)}</p>
      <p><b>結果：</b> {"正解" if is_ok else "不正解"}</p>
    </div>
    """
    return render_lineart_page(title="名句線画クイズ", form_block="", result=result_block)

@app.get("/music_draw", response_class=HTMLResponse)
def music_draw_index():
    form_block = """
    <form method="post" action="/music_draw">
      <p>俳句を入力すると、音列に変換して再生・描画します。</p>
      <p><textarea name="haiku" placeholder="俳句を入力してください" style="width:100%;padding:10px;border:1px solid #b8c8d9;border-radius:8px;"></textarea></p>
      <p><button type="submit">作曲して描画</button></p>
    </form>
    """
    return render_lineart_page(title="音列ドロー", form_block=form_block)

@app.post("/music_draw", response_class=HTMLResponse)
def music_draw(haiku: str = Form("")):
    haiku = haiku.strip()
    if not haiku:
        return render_lineart_page(title="音列ドロー", form_block="", error="俳句を入力してください。")
    prompt = f"次の俳句の情景を、短い旋律として表現してください。俳句: {haiku}"
    try:
        notes, bpm, reason = call_gpt_notes(prompt)
    except Exception as e:
        return render_lineart_page(title="音列ドロー", form_block="", error=f"作曲でエラー: {e}")

    freqs = notes_to_frequencies(notes)
    if not freqs:
        return render_lineart_page(title="音列ドロー", form_block="", error="音列の解析に失敗しました。")

    with tempfile.NamedTemporaryFile(prefix="notes_", suffix=".wav", delete=False) as tf:
        wav_path = tf.name
    generate_wav(freqs, bpm, wav_path)
    played = try_play_audio(wav_path)

    lineart = build_lineart_from_notes(freqs, width=100, height=100)
    if not lineart:
        return render_lineart_page(title="音列ドロー", form_block="", error="線画生成に失敗しました。")

    preview = build_preview_from_polylines(lineart)
    svg = lineart_to_svg(preview)
    robot_data = transform_lineart_for_robot(preview)
    if robot_data:
        robot_svg = lineart_to_svg(robot_data, viewbox=robot_data["viewbox"], scale=4)
    else:
        robot_svg = "<p>ロボット用プレビューを生成できません。</p>"

    polylines = lineart["polylines"]
    publish_lineart(lineart["width"], lineart["height"], polylines)
    reason_block = f"<p><b>作曲理由（芭蕉風）</b><br>{simple_escape(reason) if reason else '（なし）'}</p>"
    result_block = (
        f"<p>notes={simple_escape(','.join(notes))} bpm={bpm} played={'yes' if played else 'no'}</p>"
        f"{reason_block}"
        f"<p><b>元の線画プレビュー</b></p>{svg}"
        f"<p><b>ロボット指令ベース</b></p>{robot_svg}"
    )
    return render_lineart_page(title="音列ドロー", form_block="", result=result_block)

@app.post("/control/reset", response_class=HTMLResponse)
def control_reset():
    msg = HaikuResult()
    msg.original = "control:reset"
    msg.score = 4
    msg.comment = "reset-pose"
    msg.revised = ""
    publisher.publish(msg)
    rospy.loginfo("Published HaikuResult(score=4) for reset")
    return render_lineart_page(title="操作パネル", form_block="", result="<p>リセット姿勢を送信しました。</p>")

@app.post("/control/pose", response_class=HTMLResponse)
def control_pose():
    msg = HaikuResult()
    msg.original = "control:pose"
    msg.score = 5
    msg.comment = "angle-vector"
    msg.revised = ""
    publisher.publish(msg)
    rospy.loginfo("Published HaikuResult(score=5) for pose")
    return render_lineart_page(title="操作パネル", form_block="", result="<p>指定角度姿勢を送信しました。</p>")


@app.get("/face", response_class=HTMLResponse)
def face_lineart_index():
    form_block = """
    <form method="post" action="/face">
      <p>カメラ撮影して線画を生成します。</p>
      <p><button type="submit">撮影して送信</button></p>
    </form>
    """
    return render_lineart_page(title="顔認識から線画", form_block=form_block)

@app.post("/face", response_class=HTMLResponse)
def face_lineart():
    try:
        data = run_zedge4_lineart()
        width = int(data.get("width", 0))
        height = int(data.get("height", 0))
        xy = data.get("xy", [])
        counts = data.get("counts", [])
        if not counts:
            return render_lineart_page(
                title="顔認識から線画",
                form_block="",
                error="線画データが空でした。",
            )
        svg = lineart_to_svg(data)
        colored_svg, legend = lineart_to_svg_colored(data)
        robot_data = transform_lineart_for_robot(data)
        if robot_data:
            robot_svg = lineart_to_svg(
                robot_data,
                viewbox=robot_data["viewbox"],
                scale=4,
            )
            bounds = lineart_bounds(robot_data["xy"])
            if bounds:
                bounds_text = (
                    f"<p>bounds: "
                    f"left={bounds['min_x']:.2f} "
                    f"right={bounds['max_x']:.2f} "
                    f"top={bounds['max_y']:.2f} "
                    f"bottom={bounds['min_y']:.2f}</p>"
                )
            else:
                bounds_text = "<p>bounds: N/A</p>"
            robot_meta = (
                f"<p>scale={robot_data['scale']:.4f} "
                f"xoff={robot_data['xoff']:.2f} "
                f"yoff={robot_data['yoff']:.2f} "
                f"rotate={'yes' if robot_data.get('rotate') else 'no'}</p>"
            )
        else:
            robot_svg = "<p>ロボット用プレビューを生成できません。</p>"
            robot_meta = ""
            bounds_text = ""
        polylines = []
        idx = 0
        for c in counts:
            c = int(c)
            pts = []
            for _ in range(c):
                if idx + 1 >= len(xy):
                    break
                x = float(xy[idx])
                y = float(xy[idx + 1])
                pts.append((x, y))
                idx += 2
            if len(pts) >= 2:
                polylines.append(pts)
        publish_lineart(width, height, polylines)
        total_points = sum(len(pl) for pl in polylines)
        snap_svg = ""
        if robot_data:
            snap_polylines = snap_lineart_polylines(
                build_polylines_from_xy(robot_data["xy"], robot_data["counts"]),
                10.0,
                5.0,
                30.0,
                grid_origin=(120.0, -150.0),
            )
            snap_preview = build_preview_from_polylines({
                "polylines": snap_polylines,
                "width": int(robot_data["width"]),
                "height": int(robot_data["height"]),
            })
            snap_svg = lineart_to_svg(
                snap_preview,
                viewbox=robot_data["viewbox"],
                scale=4,
            )
        result_block = (
            f"<p>線画を送信しました。polylines={len(polylines)} points={total_points}</p>"
            f"<p><b>元の線画プレビュー</b></p>{svg}"
            f"<p><b>スプライン色分け</b></p>{legend}{colored_svg}"
            f"<p><b>格子点スナップ経路</b></p>{snap_svg}"
            f"<p><b>ロボット指令ベース</b></p>{robot_meta}{bounds_text}{robot_svg}"
        )
        return render_lineart_page(
            title="顔認識から線画",
            form_block="",
            result=result_block,
        )
    except Exception as e:
        return render_lineart_page(
            title="顔認識から線画",
            form_block="",
            error=f"線画生成でエラー: {e}",
        )

@app.get("/face_joint", response_class=HTMLResponse)
def face_lineart_joint_index():
    form_block = """
    <form method="post" action="/face_joint">
      <p>カメラ撮影して線画を生成します（角度指定）。</p>
      <p><button type="submit">撮影して送信</button></p>
    </form>
    """
    return render_lineart_page(title="顔線画（角度指定）", form_block=form_block)

@app.post("/face_joint", response_class=HTMLResponse)
def face_lineart_joint():
    try:
        data = run_zedge4_lineart()
        width = int(data.get("width", 0))
        height = int(data.get("height", 0))
        xy = data.get("xy", [])
        counts = data.get("counts", [])
        if not counts:
            return render_lineart_page(
                title="顔線画（角度指定）",
                form_block="",
                error="線画データが空でした。",
            )
        svg = lineart_to_svg(data)
        colored_svg, legend = lineart_to_svg_colored(data)
        robot_data = transform_lineart_for_robot(data)
        if robot_data:
            robot_svg = lineart_to_svg(
                robot_data,
                viewbox=robot_data["viewbox"],
                scale=4,
            )
            bounds = lineart_bounds(robot_data["xy"])
            if bounds:
                bounds_text = (
                    f"<p>bounds: "
                    f"left={bounds['min_x']:.2f} "
                    f"right={bounds['max_x']:.2f} "
                    f"top={bounds['max_y']:.2f} "
                    f"bottom={bounds['min_y']:.2f}</p>"
                )
            else:
                bounds_text = "<p>bounds: N/A</p>"
            robot_meta = (
                f"<p>scale={robot_data['scale']:.4f} "
                f"xoff={robot_data['xoff']:.2f} "
                f"yoff={robot_data['yoff']:.2f} "
                f"rotate={'yes' if robot_data.get('rotate') else 'no'}</p>"
            )
        else:
            robot_svg = "<p>ロボット用プレビューを生成できません。</p>"
            robot_meta = ""
            bounds_text = ""
        polylines = []
        idx = 0
        for c in counts:
            c = int(c)
            pts = []
            for _ in range(c):
                if idx + 1 >= len(xy):
                    break
                x = float(xy[idx])
                y = float(xy[idx + 1])
                pts.append((x, y))
                idx += 2
            if len(pts) >= 2:
                polylines.append(pts)
        publish_lineart_joint(width, height, polylines)
        total_points = sum(len(pl) for pl in polylines)
        result_block = (
            f"<p>線画を送信しました。polylines={len(polylines)} points={total_points}</p>"
            f"<p><b>元の線画プレビュー</b></p>{svg}"
            f"<p><b>スプライン色分け</b></p>{legend}{colored_svg}"
            f"<p><b>ロボット指令ベース</b></p>{robot_meta}{bounds_text}{robot_svg}"
        )
        return render_lineart_page(
            title="顔線画（角度指定）",
            form_block="",
            result=result_block,
        )
    except Exception as e:
        return render_lineart_page(
            title="顔線画（角度指定）",
            form_block="",
            error=f"線画生成でエラー: {e}",
        )

@app.get("/face_joint_snap", response_class=HTMLResponse)
def face_lineart_joint_snap_index():
    form_block = """
    <form method="post" action="/face_joint_snap">
      <p>カメラ撮影して線画を生成します（角度指定・スナップ）。</p>
      <p><button type="submit">撮影して送信</button></p>
    </form>
    """
    return render_lineart_page(title="顔線画（角度指定・スナップ）", form_block=form_block)

@app.post("/face_joint_snap", response_class=HTMLResponse)
def face_lineart_joint_snap():
    try:
        data = run_zedge4_lineart()
        width = int(data.get("width", 0))
        height = int(data.get("height", 0))
        xy = data.get("xy", [])
        counts = data.get("counts", [])
        if not counts:
            return render_lineart_page(
                title="顔線画（角度指定・スナップ）",
                form_block="",
                error="線画データが空でした。",
            )
        svg = lineart_to_svg(data)
        colored_svg, legend = lineart_to_svg_colored(data)
        robot_data = transform_lineart_for_robot(data)
        if robot_data:
            robot_svg = lineart_to_svg(
                robot_data,
                viewbox=robot_data["viewbox"],
                scale=4,
            )
            bounds = lineart_bounds(robot_data["xy"])
            if bounds:
                bounds_text = (
                    f"<p>bounds: "
                    f"left={bounds['min_x']:.2f} "
                    f"right={bounds['max_x']:.2f} "
                    f"top={bounds['max_y']:.2f} "
                    f"bottom={bounds['min_y']:.2f}</p>"
                )
            else:
                bounds_text = "<p>bounds: N/A</p>"
            robot_meta = (
                f"<p>scale={robot_data['scale']:.4f} "
                f"xoff={robot_data['xoff']:.2f} "
                f"yoff={robot_data['yoff']:.2f} "
                f"rotate={'yes' if robot_data.get('rotate') else 'no'}</p>"
            )
        else:
            robot_svg = "<p>ロボット用プレビューを生成できません。</p>"
            robot_meta = ""
            bounds_text = ""
        polylines = []
        idx = 0
        for c in counts:
            c = int(c)
            pts = []
            for _ in range(c):
                if idx + 1 >= len(xy):
                    break
                x = float(xy[idx])
                y = float(xy[idx + 1])
                pts.append((x, y))
                idx += 2
            if len(pts) >= 2:
                polylines.append(pts)
        snap_svg = ""
        if robot_data:
            snap_polylines = snap_lineart_polylines(
                build_polylines_from_xy(robot_data["xy"], robot_data["counts"]),
                10.0,
                5.0,
                30.0,
                grid_origin=(120.0, -150.0),
            )
            snap_preview = build_preview_from_polylines({
                "polylines": snap_polylines,
                "width": int(robot_data["width"]),
                "height": int(robot_data["height"]),
            })
            snap_svg = lineart_to_svg(
                snap_preview,
                viewbox=robot_data["viewbox"],
                scale=4,
            )
        publish_lineart_joint_snap(width, height, polylines)
        total_points = sum(len(pl) for pl in polylines)
        result_block = (
            f"<p>線画を送信しました。polylines={len(polylines)} points={total_points}</p>"
            f"<p><b>元の線画プレビュー</b></p>{svg}"
            f"<p><b>スプライン色分け</b></p>{legend}{colored_svg}"
            f"<p><b>格子点スナップ経路</b></p>{snap_svg}"
            f"<p><b>ロボット指令ベース</b></p>{robot_meta}{bounds_text}{robot_svg}"
        )
        return render_lineart_page(
            title="顔線画（角度指定・スナップ）",
            form_block="",
            result=result_block,
        )
    except Exception as e:
        return render_lineart_page(
            title="顔線画（角度指定・スナップ）",
            form_block="",
            error=f"線画生成でエラー: {e}",
        )


if __name__ == "__main__":
    # uvicorn で FastAPI を起動（reload=False にして二重起動を防ぐ）
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
