#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
from typing import List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI

Point = Tuple[float, float]
Polyline = List[Point]

SYSTEM_STYLE = """あなたはプロッタ/ロボットが描ける線画を設計するエンジニアです。
出力は必ず指定JSONスキーマに従ってください。余計な文章は禁止。
加えて、reason に松尾芭蕉になりきって描いた内容の説明を入れてください。"""

JSON_SCHEMA = {
    "name": "haiku_polylines",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "width": {"type": "integer", "minimum": 200, "maximum": 2000},
            "height": {"type": "integer", "minimum": 200, "maximum": 2000},
            "polylines": {
                "type": "array",
                "maxItems": 50,
                "items": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 50,
                    "items": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                },
            },
            "reason": {"type": "string"},
        },
        "required": ["width", "height", "polylines", "reason"],
    },
}

def log(msg: str, debug: bool = True) -> None:
    if debug:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

def build_prompt(haiku: str, max_points_total: int) -> str:
    return f"""次の俳句をモチーフに、白紙に黒ペンで描く「線画」を設計してください。

【俳句】
{haiku}

【出力の意味】
- polylines は折れ線の集合です。各 polyline は点列 [[x,y],[x,y],...] で、点同士を直線で結びます。
- 座標は正規化(0.0〜1.0)。(0,0)が左上、(1,1)が右下。
- ただし「全ポリライン合計の点数」は最大 {max_points_total} 点以内にしてください（厳守）。
- 合計点数は最大 {max_points_total} 点まで使い切るつもりで、できるだけ {max_points_total} に近づけてください（重要）。
- 線が交差しすぎたり、密になりすぎないように。シンプルに。

【絵のスタイル制約】
- 輪郭中心の線画。陰影・塗り・網点・テクスチャなし。
- 俳句の情景が伝わるように、主題を中央寄りに配置。
- 用紙サイズ width/height は 800x800 を基本にしてOK。
"""

def enforce_max_points(polylines: List[Polyline], max_total: int) -> List[Polyline]:
    total = sum(len(pl) for pl in polylines)
    if total <= max_total:
        return polylines

    flat: List[Point] = []
    offsets: List[int] = []
    acc = 0
    for pl in polylines:
        offsets.append(acc)
        acc += len(pl)
        flat.extend(pl)

    keep_idx = set(np.linspace(0, len(flat) - 1, max_total).round().astype(int).tolist())

    new_polylines: List[Polyline] = []
    for i, pl in enumerate(polylines):
        base = offsets[i]
        new_pl = [p for j, p in enumerate(pl) if (base + j) in keep_idx]
        if len(new_pl) >= 2:
            new_polylines.append(new_pl)

    while sum(len(pl) for pl in new_polylines) > max_total and new_polylines:
        idx = int(np.argmax([len(pl) for pl in new_polylines]))
        if len(new_polylines[idx]) > 2:
            new_polylines[idx].pop()
        else:
            new_polylines.pop(idx)

    return new_polylines


def print_point_stats(polylines: List[Polyline]) -> None:
    counts = [len(pl) for pl in polylines]
    total = sum(counts)
    if not counts:
        print("[stats] polylines=0, total_points=0", flush=True)
        return

    print(f"[stats] polylines={len(polylines)}", flush=True)
    print(f"[stats] total_points={total}", flush=True)
    print(f"[stats] points_per_polyline(min/median/max)={min(counts)}/{int(np.median(counts))}/{max(counts)}", flush=True)

    # 上位いくつかのポリライン点数も表示
    top = sorted(counts, reverse=True)[:10]
    print(f"[stats] top10 polyline point counts={top}", flush=True)


def plot_polylines(polylines: List[Polyline], width: int, height: int) -> None:
    plt.figure()
    for pl in polylines:
        xs = [p[0] * width for p in pl]
        ys = [p[1] * height for p in pl]
        plt.plot(xs, ys)

    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(0, width)
    plt.ylim(height, 0)
    plt.title("Line art (polylines)")
    plt.show()

def safe_preview(obj: Any, limit: int = 1200) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        s = str(obj)
    if len(s) > limit:
        return s[:limit] + "\n...(truncated)..."
    return s

def main():
    ap = argparse.ArgumentParser(description="Haiku -> polylines -> display (with debug logs)")
    ap.add_argument("--haiku", type=str, default=None, help="俳句（省略時は標準入力1行）")
    ap.add_argument("--out", type=str, default="polylines.json", help="出力JSON")
    ap.add_argument("--model", type=str, default="gpt-5.2", help="利用モデル名")
    ap.add_argument("--max_points", type=int, default=50, help="合計点数上限")
    ap.add_argument("--debug", action="store_true", help="デバッグログを詳細に出す")
    ap.add_argument("--timeout", type=float, default=120.0, help="APIタイムアウト秒（SDK側）")
    args = ap.parse_args()

    debug = args.debug

    haiku = args.haiku or input("俳句を1行で入力してください: ").strip()
    if not haiku:
        raise ValueError("俳句が空です")

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("環境変数 OPENAI_API_KEY が見つかりません。.bashrc などに設定してください。")

    log(f"Model={args.model}, max_points={args.max_points}, timeout={args.timeout}s", debug)
    log(f"Haiku: {haiku}", debug)

    # timeoutはOpenAIクライアント生成時に指定（SDKのHTTPタイムアウト）
    client = OpenAI(timeout=args.timeout)

    prompt = build_prompt(haiku, args.max_points)
    if debug:
        log("Prompt preview:", debug)
        log(prompt[:600] + ("\n...(truncated)..." if len(prompt) > 600 else ""), debug)

    log("Calling OpenAI Responses API...", debug)
    t0 = time.time()

    try:
        resp = client.responses.create(
            model=args.model,
            input=[
                {"role": "system", "content": SYSTEM_STYLE},
                {"role": "user", "content": prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": JSON_SCHEMA["name"],
                    "schema": JSON_SCHEMA["schema"],
                }
            },
        )
    except Exception as e:
        log(f"API call failed: {repr(e)}", True)
        raise

    dt = time.time() - t0
    log(f"API returned in {dt:.2f}s", debug)

    # ここから先で “空/変な返り” を可視化
    if debug:
        # できる範囲でメタ情報も出す
        rid = getattr(resp, "id", None)
        log(f"response.id={rid}", debug)
        # 全体ダンプは大きいのでプレビューだけ
        try:
            dump = resp.model_dump()
            log("Response dump preview:", debug)
            log(safe_preview(dump, limit=1600), debug)
        except Exception as e:
            log(f"Could not model_dump(): {repr(e)}", debug)

    # output_text が空のときがあるので対策：rawから拾う
    raw_text = ""
    try:
        raw_text = resp.output_text or ""
    except Exception:
        raw_text = ""

    if debug:
        log(f"resp.output_text length={len(raw_text)}", debug)

    if not raw_text.strip():
        # 最後の保険：resp.output から探す（SDK差異対策）
        if debug:
            log("output_text is empty. Trying to extract from resp.output...", debug)
        try:
            # resp.output は通常 list。中に content/text が入る
            out = getattr(resp, "output", None)
            if out:
                # 可能な限り文字列を探索
                candidates = []
                for item in out:
                    content = getattr(item, "content", None)
                    if not content:
                        continue
                    for c in content:
                        # c.text or c.get("text")
                        txt = getattr(c, "text", None)
                        if isinstance(txt, str) and txt.strip():
                            candidates.append(txt)
                if candidates:
                    raw_text = candidates[0]
                    log(f"Extracted text candidate length={len(raw_text)}", debug)
        except Exception as e:
            log(f"Failed to extract from resp.output: {repr(e)}", debug)

    if not raw_text.strip():
        raise RuntimeError("モデル出力が空でした（output_textも抽出候補も空）。debugログの Response dump を確認してください。")

    # JSONパース
    log("Parsing JSON...", debug)
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        log("JSON parse failed. Raw text preview:", True)
        print(raw_text[:2000], flush=True)
        raise RuntimeError(f"JSON decode error: {e}") from e

    # 構造確認
    log("Validating keys...", debug)
    for k in ("width", "height", "polylines", "reason"):
        if k not in data:
            raise RuntimeError(f"Missing key in output JSON: {k}")

    width = int(data["width"])
    height = int(data["height"])

    polylines: List[Polyline] = []
    for pl in data["polylines"]:
        poly = [(float(x), float(y)) for x, y in pl]
        if len(poly) >= 2:
            polylines.append(poly)

    polylines = enforce_max_points(polylines, args.max_points)

    total_pts = sum(len(pl) for pl in polylines)
    log(f"polylines={len(polylines)}, total_points={total_pts}", debug)

    reason = str(data.get("reason", "")).strip()
    out_obj = {"width": width, "height": height, "polylines": polylines, "reason": reason}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    log(f"Saved JSON to: {args.out}", debug)
    log("Displaying plot window...", debug)
    print_point_stats(polylines)
    plot_polylines(polylines, width, height)

if __name__ == "__main__":
    main()
