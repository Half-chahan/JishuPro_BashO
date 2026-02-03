#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Face line drawing + Glasses (final: no tuner)

仕様（完成版）:
- 画面表示は「点＋線」のみ（1ウィンドウ）
- 各折れ線（ポリライン）はそれぞれ違う色で描画
- 点は、どの折れ線色とも異なる “固定色” で描画
- パラメータはコードに直書き（添付画像の値を反映）

描画仕様（前回までの仕様を維持）:
- メガネ判定(glasses=ON)のとき: 目(leye/reye)は描画しない、メガネを描画する
- メガネ判定(glasses=OFF)のとき: メガネは描画しない、目は必ず描画（予算削減でも消えにくい）

依存:
pip install mediapipe scikit-image opencv-python
"""

import argparse
import json
import cv2
import numpy as np
import math
from skimage.morphology import thin
import mediapipe as mp


# =========================
# Parameters (from screenshot)
# =========================
EPSILON_PX = 4.0
MAX_LEN = 220.0
BG_STROKES = 0
TARGET_PTS = 60

GL_CANNY_LO = 25
GL_CANNY_HI = 90

NUM_RAYS = 18
R_MAX_FACTOR = 3.00          # r_max% = 300%
START_OFFSET = 6.0
STEP = 1.0

CAP_MODE = 0                  # 0=NONE(r_max), 1=t_box*factor, 2=t_box+margin
CAP_FACTOR = 2.20             # cap_factor% = 220%
CAP_MARGIN_PX = 120.0
CAP_MIN_PX = 120.0

BRIDGE_PTS = 3
GL_HIT_RATE_TH = 0.20         # gl_hit_rate% = 20%


# =========================
# Capture
# =========================
def capture_frame_with_preview(cam_index: int = 0):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError("カメラが開けませんでした。cam_index を確認してください。")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[Capture] プレビュー：Spaceで撮影 / qまたはESCで終了")
    frame = None
    while True:
        ret, f = cap.read()
        if not ret or f is None:
            continue
        cv2.imshow("preview (SPACE to capture)", f)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            cap.release()
            cv2.destroyWindow("preview (SPACE to capture)")
            raise RuntimeError("ユーザーが終了しました。")
        if k == ord(" "):
            frame = f.copy()
            break

    cap.release()
    cv2.destroyWindow("preview (SPACE to capture)")
    return frame


# =========================
# Geometry
# =========================
def poly_length(p: np.ndarray) -> float:
    if len(p) < 2:
        return 0.0
    d = np.diff(p.astype(np.float32), axis=0)
    return float(np.sum(np.linalg.norm(d, axis=1)))

def approx_rdp_open(pts: np.ndarray, epsilon_px: float) -> np.ndarray:
    if len(pts) < 3 or epsilon_px <= 0:
        return pts.astype(np.int32)
    cnt = pts.astype(np.int32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(cnt, epsilon=float(epsilon_px), closed=False)
    return approx.reshape(-1, 2).astype(np.int32)

def split_long_segments(pts: np.ndarray, max_segment_length: float) -> np.ndarray:
    if max_segment_length <= 0 or len(pts) < 2:
        return pts.astype(np.int32)
    refined = [pts[0].astype(np.int32)]
    for i in range(1, len(pts)):
        a = refined[-1].astype(np.float32)
        b = pts[i].astype(np.float32)
        dist = float(np.linalg.norm(b - a))
        if dist <= max_segment_length:
            refined.append(pts[i].astype(np.int32))
        else:
            n = int(math.ceil(dist / max_segment_length))
            for k in range(1, n):
                t = k / n
                p = (1 - t) * a + t * b
                refined.append(p.astype(np.int32))
            refined.append(pts[i].astype(np.int32))
    return np.array(refined, dtype=np.int32)

def resample_polyline_by_arclen(pts: np.ndarray, n: int, closed: bool = False) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 2 or n <= 2:
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.int32)
        if len(pts) == 1:
            return np.array([pts[0], pts[0]], dtype=np.int32)
        return np.array([pts[0], pts[-1]], dtype=np.int32)

    if closed:
        pts2 = np.vstack([pts, pts[0]])
    else:
        pts2 = pts

    seg = np.diff(pts2, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    total = float(np.sum(seglen))
    if total < 1e-6:
        out = np.repeat(pts2[:1], n, axis=0)
        return out.astype(np.int32)

    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    targets = np.linspace(0.0, total, n, endpoint=not closed)

    out = []
    j = 0
    for t in targets:
        while j + 1 < len(cum) and cum[j + 1] < t:
            j += 1
        if j + 1 >= len(cum):
            out.append(pts2[-1])
            continue
        t0, t1 = cum[j], cum[j + 1]
        if t1 - t0 < 1e-9:
            out.append(pts2[j])
        else:
            r = (t - t0) / (t1 - t0)
            p = (1 - r) * pts2[j] + r * pts2[j + 1]
            out.append(p)
    return np.array(out, dtype=np.int32)


# =========================
# FaceMesh connections -> ordered path
# =========================
def _ordered_path_from_connections(connections):
    nbr = {}
    nodes = set()
    for a, b in connections:
        nodes.add(a); nodes.add(b)
        nbr.setdefault(a, set()).add(b)
        nbr.setdefault(b, set()).add(a)
    if not nodes:
        return []

    deg = {n: len(nbr.get(n, [])) for n in nodes}
    ends = [n for n, d in deg.items() if d == 1]
    start = min(ends) if ends else min(nodes)

    path = [start]
    prev = None
    cur = start
    for _ in range(len(nodes) + 50):
        candidates = list(nbr.get(cur, []))
        if not candidates:
            break
        if prev is None:
            nxt = min(candidates)
        else:
            c2 = [c for c in candidates if c != prev]
            nxt = min(c2) if c2 else candidates[0]

        if nxt == start and prev is not None:
            path.append(start)
            break
        if nxt in path and ends:
            break

        path.append(nxt)
        prev, cur = cur, nxt

        if len(ends) >= 2 and cur in ends and cur != start:
            break

    return path

def extract_facemesh_feature_polylines_and_landmarks(frame_bgr: np.ndarray):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    mp_fm = mp.solutions.face_mesh
    with mp_fm.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as fm:
        res = fm.process(rgb)

    if not res.multi_face_landmarks:
        return None, None

    lm = res.multi_face_landmarks[0].landmark
    lm_pts = np.array([(int(p.x * w), int(p.y * h)) for p in lm], dtype=np.int32)

    def pt(i: int):
        return (int(lm[i].x * w), int(lm[i].y * h))

    conns = {
        "oval":  mp.solutions.face_mesh.FACEMESH_FACE_OVAL,
        "lips":  mp.solutions.face_mesh.FACEMESH_LIPS,
        "leye":  mp.solutions.face_mesh.FACEMESH_LEFT_EYE,
        "reye":  mp.solutions.face_mesh.FACEMESH_RIGHT_EYE,
        "lbrow": mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW,
        "rbrow": mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW,
        "nose":  mp.solutions.face_mesh.FACEMESH_NOSE,
    }

    features = {}
    for name, cc in conns.items():
        path_idx = _ordered_path_from_connections(cc)
        if len(path_idx) < 2:
            continue
        pts = np.array([pt(i) for i in path_idx], dtype=np.int32)
        closed = bool(len(pts) >= 3 and (pts[0] == pts[-1]).all())
        if closed:
            pts = pts[:-1]
        features[name] = (pts, closed)

    return features, lm_pts


# =========================
# Background coarse (optional)
# =========================
def detect_edges_background_coarse(gray: np.ndarray):
    h, w = gray.shape[:2]
    scale = 0.20
    sw, sh = max(64, int(w * scale)), max(64, int(h * scale))
    small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (13, 13), 0)
    edges_small = cv2.Canny(small, 170, 255)
    edges = cv2.resize(edges_small, (w, h), interpolation=cv2.INTER_NEAREST)
    return edges

def skeletonize_edges(edges: np.ndarray):
    return (thin(edges > 0).astype(np.uint8)) * 255

def find_raw_polylines_from_skeleton(skel: np.ndarray):
    contours_info = cv2.findContours(skel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    MIN_POINTS = 120
    raw = [c.reshape(-1, 2) for c in contours if len(c) >= MIN_POINTS]
    raw.sort(key=poly_length, reverse=True)
    return raw


# =========================
# Glasses raw extraction
# =========================
def extract_glasses_outline(gray: np.ndarray, face_oval_pts: np.ndarray,
                           canny_lo: int, canny_hi: int):
    if face_oval_pts is None or len(face_oval_pts) < 10:
        return None, None, None

    H, W = gray.shape[:2]
    x, y, w, h = cv2.boundingRect(face_oval_pts.astype(np.int32))

    y0 = y + int(h * 0.18)
    y1 = y + int(h * 0.70)
    x0 = x + int(w * 0.05)
    x1 = x + int(w * 0.95)
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(W, x1); y1 = min(H, y1)
    if x1 - x0 < 80 or y1 - y0 < 60:
        return None, None, None

    roi = gray[y0:y1, x0:x1]
    roi = cv2.bilateralFilter(roi, d=7, sigmaColor=70, sigmaSpace=70)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    roi = clahe.apply(roi)
    blur = cv2.GaussianBlur(roi, (0, 0), 1.2)
    roi = cv2.addWeighted(roi, 1.9, blur, -0.9, 0)

    edges = cv2.Canny(roi, int(canny_lo), int(canny_hi))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=1)

    contours_info = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    if not contours:
        return None, (x0, y0, x1, y1), edges

    best = None
    best_score = -1.0
    for c in contours:
        if len(c) < 80:
            continue
        pts = c.reshape(-1, 2)
        rx, ry, rw, rh = cv2.boundingRect(pts.astype(np.int32))
        area = float(rw * rh)
        aspect = rw / max(1.0, rh)
        if area < (roi.shape[0] * roi.shape[1]) * 0.02:
            continue
        score = area * (0.7 + 0.3 * min(2.0, aspect))
        if score > best_score:
            best_score = score
            best = pts

    if best is None:
        return None, (x0, y0, x1, y1), edges

    best2 = best.astype(np.int32)
    best2[:, 0] += x0
    best2[:, 1] += y0
    return best2, (x0, y0, x1, y1), edges


# =========================
# Eye / boxes
# =========================
def eye_centers_and_dist(lm_pts: np.ndarray):
    L_OUT, L_IN = 33, 133
    R_OUT, R_IN = 263, 362
    left_c = ((lm_pts[L_OUT].astype(np.float32) + lm_pts[L_IN].astype(np.float32)) * 0.5)
    right_c = ((lm_pts[R_OUT].astype(np.float32) + lm_pts[R_IN].astype(np.float32)) * 0.5)
    eye_dist = float(np.linalg.norm(right_c - left_c))
    mid_x = float((left_c[0] + right_c[0]) * 0.5)
    return left_c, right_c, eye_dist, mid_x

def eye_box_from_feature(eye_pts: np.ndarray, expand: float = 1.35):
    x, y, w, h = cv2.boundingRect(eye_pts.astype(np.int32))
    cx = x + w * 0.5
    cy = y + h * 0.5
    w2 = w * expand
    h2 = h * expand
    x0 = int(round(cx - w2 * 0.5))
    y0 = int(round(cy - h2 * 0.5))
    x1 = int(round(cx + w2 * 0.5))
    y1 = int(round(cy + h2 * 0.5))
    return (x0, y0, x1, y1), np.array([cx, cy], dtype=np.float32)

def clamp_box(box, W, H):
    x0, y0, x1, y1 = box
    x0 = max(0, min(W - 1, x0))
    x1 = max(0, min(W - 1, x1))
    y0 = max(0, min(H - 1, y0))
    y1 = max(0, min(H - 1, y1))
    if x1 < x0: x0, x1 = x1, x0
    if y1 < y0: y0, y1 = y1, y0
    return (x0, y0, x1, y1)


# =========================
# Raycast on raw mask
# =========================
def build_raw_mask(shape_hw, raw_poly: np.ndarray, thickness: int = 2):
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if raw_poly is None or len(raw_poly) < 2:
        return mask
    cv2.polylines(mask, [raw_poly.reshape(-1, 1, 2).astype(np.int32)], True, 255, thickness)
    return mask

def raycast_first_hit(mask: np.ndarray,
                      origin: np.ndarray,
                      direction: np.ndarray,
                      r_max: float,
                      step: float = 1.0):
    h, w = mask.shape[:2]
    ox, oy = float(origin[0]), float(origin[1])
    dx, dy = float(direction[0]), float(direction[1])

    r = 0.0
    while r <= r_max:
        x = int(round(ox + dx * r))
        y = int(round(oy + dy * r))
        if x < 0 or y < 0 or x >= w or y >= h:
            break
        if mask[y, x] > 0:
            return np.array([x, y], dtype=np.int32)
        r += float(step)
    return None

def ray_rect_boundary_start(center_xy: np.ndarray, direction_xy: np.ndarray, box, eps: float = 1e-6):
    cx, cy = float(center_xy[0]), float(center_xy[1])
    dx, dy = float(direction_xy[0]), float(direction_xy[1])
    x0, y0, x1, y1 = box

    candidates = []

    if abs(dx) > eps:
        t = (x0 - cx) / dx
        if t > 0:
            y = cy + t * dy
            if y0 - 1e-3 <= y <= y1 + 1e-3:
                candidates.append((t, x0, y))
        t = (x1 - cx) / dx
        if t > 0:
            y = cy + t * dy
            if y0 - 1e-3 <= y <= y1 + 1e-3:
                candidates.append((t, x1, y))

    if abs(dy) > eps:
        t = (y0 - cy) / dy
        if t > 0:
            x = cx + t * dx
            if x0 - 1e-3 <= x <= x1 + 1e-3:
                candidates.append((t, x, y0))
        t = (y1 - cy) / dy
        if t > 0:
            x = cx + t * dx
            if x0 - 1e-3 <= x <= x1 + 1e-3:
                candidates.append((t, x, y1))

    if not candidates:
        return None
    tmin, bx, by = min(candidates, key=lambda z: z[0])
    return np.array([bx, by], dtype=np.float32), float(tmin)

def lens_points_from_360_rays(mask: np.ndarray,
                              eye_box,
                              num_rays: int,
                              r_max: float,
                              step: float,
                              start_offset: float,
                              cap_factor: float,
                              cap_mode: int,
                              cap_margin_px: float,
                              cap_min_px: float):
    x0, y0, x1, y1 = eye_box
    center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float32)

    num_rays = int(max(3, num_rays))
    cap_factor = float(max(0.1, cap_factor))
    cap_mode = int(cap_mode)
    cap_margin_px = float(max(0.0, cap_margin_px))
    cap_min_px = float(max(0.0, cap_min_px))

    angs = [2.0 * math.pi * i / num_rays for i in range(num_rays)]

    hits = []
    hit_ok = 0

    for a in angs:
        d = np.array([math.cos(a), math.sin(a)], dtype=np.float32)

        bpt = ray_rect_boundary_start(center, d, eye_box)
        boundary_pt = None
        t_box = None
        if bpt is not None:
            boundary_pt, t_box = bpt

        origin = center.copy()

        if cap_mode == 0:
            r_cap = float(r_max)
        else:
            if t_box is None:
                r_cap = float(r_max)
            else:
                if cap_mode == 1:
                    r_cap = float(t_box) * cap_factor + float(start_offset)
                else:
                    r_cap = float(t_box) + cap_margin_px
                r_cap = max(r_cap, cap_min_px)
                r_cap = min(float(r_max), float(r_cap))

        hit = raycast_first_hit(mask, origin=origin, direction=d, r_max=r_cap, step=step)
        if hit is not None:
            hit_ok += 1
            hits.append(hit)
        else:
            if boundary_pt is not None:
                fallback = (boundary_pt + d * float(start_offset)).astype(np.int32)
            else:
                fallback = (origin + d * float(r_cap)).astype(np.int32)
            hits.append(fallback)

    if len(hits) < 3:
        return None, hit_ok, num_rays

    hits = np.array(hits, dtype=np.int32)

    # 角度順
    hcx, hcy = center[0], center[1]
    hang = np.arctan2(hits[:, 1].astype(np.float32) - hcy, hits[:, 0].astype(np.float32) - hcx)
    order = np.argsort(hang)
    hits = hits[order]

    # dedup
    dedup = [hits[0]]
    for p in hits[1:]:
        if np.linalg.norm(p.astype(np.float32) - dedup[-1].astype(np.float32)) >= 3.0:
            dedup.append(p)
    if len(dedup) < 3:
        return None, hit_ok, num_rays

    return np.array(dedup, dtype=np.int32), hit_ok, num_rays

def mirror_poly_x(poly: np.ndarray, mid_x: float):
    if poly is None or len(poly) == 0:
        return None
    out = poly.copy().astype(np.int32)
    out[:, 0] = (2.0 * float(mid_x) - out[:, 0].astype(np.float32)).round().astype(np.int32)
    return out

def close_loop(poly: np.ndarray):
    if poly is None or len(poly) < 3:
        return poly
    if (poly[0] == poly[-1]).all():
        return poly
    return np.vstack([poly, poly[0].reshape(1, 2)]).astype(np.int32)

def closest_points_between_polys(A: np.ndarray, B: np.ndarray):
    if A is None or B is None or len(A) == 0 or len(B) == 0:
        return None, None

    Af = A.astype(np.float32)
    Bf = B.astype(np.float32)

    best_d2 = 1e18
    best_i = 0
    best_j = 0
    for i in range(len(Af)):
        diff = Bf - Af[i]
        d2 = np.sum(diff * diff, axis=1)
        j = int(np.argmin(d2))
        if float(d2[j]) < best_d2:
            best_d2 = float(d2[j])
            best_i = i
            best_j = j

    return A[best_i].astype(np.int32), B[best_j].astype(np.int32)

def build_glasses_from_360_rays(frame_shape, lm_pts, features,
                                glasses_raw: np.ndarray):
    H, W = frame_shape[:2]
    _, _, eye_dist, mid_x = eye_centers_and_dist(lm_pts)

    leye_pts = features["leye"][0] if "leye" in features else None
    reye_pts = features["reye"][0] if "reye" in features else None
    if leye_pts is None or reye_pts is None:
        return None, {"hit_rate_L": 0.0, "hit_rate_R": 0.0}

    lbox, _ = eye_box_from_feature(leye_pts, expand=1.35)
    rbox, _ = eye_box_from_feature(reye_pts, expand=1.35)
    lbox = clamp_box(lbox, W, H)
    rbox = clamp_box(rbox, W, H)

    mask = build_raw_mask((H, W), glasses_raw, thickness=2)
    r_max = max(60.0, float(eye_dist) * float(R_MAX_FACTOR))

    lens_L, okL, nL = lens_points_from_360_rays(
        mask, lbox,
        num_rays=NUM_RAYS,
        r_max=r_max,
        step=STEP,
        start_offset=START_OFFSET,
        cap_factor=CAP_FACTOR,
        cap_mode=CAP_MODE,
        cap_margin_px=CAP_MARGIN_PX,
        cap_min_px=CAP_MIN_PX
    )
    lens_R, okR, nR = lens_points_from_360_rays(
        mask, rbox,
        num_rays=NUM_RAYS,
        r_max=r_max,
        step=STEP,
        start_offset=START_OFFSET,
        cap_factor=CAP_FACTOR,
        cap_mode=CAP_MODE,
        cap_margin_px=CAP_MARGIN_PX,
        cap_min_px=CAP_MIN_PX
    )

    hit_rate_L = float(okL) / max(1, int(nL))
    hit_rate_R = float(okR) / max(1, int(nR))

    if lens_L is None and lens_R is None:
        return None, {"hit_rate_L": hit_rate_L, "hit_rate_R": hit_rate_R}

    if lens_L is None and lens_R is not None:
        lens_L = mirror_poly_x(lens_R, mid_x=mid_x)
    if lens_R is None and lens_L is not None:
        lens_R = mirror_poly_x(lens_L, mid_x=mid_x)

    lens_Lc = close_loop(lens_L) if lens_L is not None else None
    lens_Rc = close_loop(lens_R) if lens_R is not None else None

    bridge = None
    pL, pR = closest_points_between_polys(lens_L, lens_R)
    if pL is not None and pR is not None:
        bridge = []
        for i in range(int(max(2, BRIDGE_PTS))):
            t = i / (max(2, BRIDGE_PTS) - 1)
            p = (1 - t) * pL.astype(np.float32) + t * pR.astype(np.float32)
            bridge.append(p.astype(np.int32))
        bridge = np.array(bridge, dtype=np.int32)

    parts = []
    if lens_Lc is not None and len(lens_Lc) >= 4:
        parts.append(lens_Lc.astype(np.int32))
    if bridge is not None and len(bridge) >= 2:
        parts.append(bridge.astype(np.int32))
    if lens_Rc is not None and len(lens_Rc) >= 4:
        parts.append(lens_Rc.astype(np.int32))

    parts = [split_long_segments(p, max_segment_length=MAX_LEN) for p in parts]

    return parts, {"hit_rate_L": hit_rate_L, "hit_rate_R": hit_rate_R}


# =========================
# Allocation
# =========================
def allocate_points(target_total: int, has_glasses: bool):
    alloc = {
        "oval": 16,
        "lips": 10,
        "nose": 8,
        "lbrow": 5,
        "rbrow": 5,
        "leye": 4,
        "reye": 4,
        "glasses": 0,
        "bg": 0
    }
    if has_glasses:
        alloc["glasses"] = 26
        alloc["leye"] = 3
        alloc["reye"] = 3

    order = ["bg", "nose", "lbrow", "rbrow", "leye", "reye", "lips", "glasses", "oval"]
    def total(a): return sum(max(0, v) for v in a.values())

    while total(alloc) > target_total:
        reduced = False
        for k in order:
            if alloc.get(k, 0) > 2 and total(alloc) > target_total:
                alloc[k] -= 1
                reduced = True
        if not reduced:
            break

    alloc["oval"] = max(10, alloc["oval"])
    alloc["lips"] = max(6, alloc["lips"])
    alloc["nose"] = max(6, alloc["nose"])
    if has_glasses:
        alloc["glasses"] = max(18, alloc["glasses"])
    return alloc


# =========================
# Rendering (different color per polyline + fixed point color)
# =========================
POLYLINE_COLORS = [
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (180, 80, 255),  # Pink-ish
    (255, 180, 80),  # Orange-ish
    (80, 180, 255),  # Light blue-ish
    (80, 255, 180),  # Mint-ish
]
POINT_COLOR = (0, 0, 0)  # Black (must be different from any polyline color above)

def render_points_and_lines(shape_hw, polylines, show_points=True):
    h, w = shape_hw
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    # lines (each polyline different color)
    for i, pts in enumerate(polylines):
        if pts is None or len(pts) < 2:
            continue
        col = POLYLINE_COLORS[i % len(POLYLINE_COLORS)]
        pts2 = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts2], isClosed=False, color=col, thickness=2)

    # points (fixed color)
    if show_points:
        for pts in polylines:
            if pts is None or len(pts) < 1:
                continue
            for (x, y) in pts:
                cv2.circle(canvas, (int(x), int(y)), 3, POINT_COLOR, -1)

    return canvas


# =========================
# Output
# =========================
def polylines_to_lineart(polylines, width: int, height: int):
    xy = []
    counts = []
    for pl in polylines:
        if pl is None or len(pl) < 2:
            continue
        counts.append(int(len(pl)))
        for x, y in pl:
            xy.append(float(x))
            xy.append(float(y))
    return {
        "width": int(width),
        "height": int(height),
        "xy": xy,
        "counts": counts,
    }


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser(description="Face line drawing to lineart JSON")
    ap.add_argument("--out", type=str, default=None, help="lineart JSON output path")
    ap.add_argument("--no-gui", action="store_true", help="disable GUI display")
    args = ap.parse_args()

    frame = capture_frame_with_preview(0)
    cv2.imwrite("captured_frame.jpg", frame)
    print("[Debug] saved captured_frame.jpg")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # FaceMesh one-shot
    features, lm_pts = extract_facemesh_feature_polylines_and_landmarks(frame)
    if features is None:
        print("[FaceMesh] 顔が取れませんでした（FaceMesh）")
        return

    oval_pts = features["oval"][0] if "oval" in features else None

    # background one-shot（機能は残す / ただし BG_STROKES=0 で出ない）
    edges_bg = detect_edges_background_coarse(gray)
    skel_bg = skeletonize_edges(edges_bg)
    raw_bg = find_raw_polylines_from_skeleton(skel_bg)

    # Glasses raw
    glasses_raw, _, _ = (None, None, None)
    if oval_pts is not None:
        glasses_raw, _, _ = extract_glasses_outline(gray, oval_pts, GL_CANNY_LO, GL_CANNY_HI)
    has_raw = glasses_raw is not None and len(glasses_raw) >= 80

    # Glasses final
    glasses_parts = None
    hitL = hitR = 0.0
    if has_raw:
        glasses_parts, dbg = build_glasses_from_360_rays(frame.shape, lm_pts, features, glasses_raw)
        hitL = float(dbg.get("hit_rate_L", 0.0))
        hitR = float(dbg.get("hit_rate_R", 0.0))

    has_glasses = bool(
        has_raw
        and glasses_parts is not None
        and len(glasses_parts) >= 1
        and (hitL >= GL_HIT_RATE_TH)
        and (hitR >= GL_HIT_RATE_TH)
    )

    alloc = allocate_points(TARGET_PTS, has_glasses=has_glasses)

    # polylines with lock flag (budget clamp)
    items = []  # {"pts":..., "lock":bool, "name":str}

    def add_item(pts, name, lock):
        if pts is None or len(pts) < 2:
            return
        items.append({"pts": pts.astype(np.int32), "lock": bool(lock), "name": str(name)})

    def add_feature(name: str, closed_default: bool, lock: bool):
        if name not in features:
            return
        pts, closed = features[name]
        closed = bool(closed) if name in ("oval", "lips", "leye", "reye") else closed_default
        n = int(alloc.get(name, 0))
        if n <= 0:
            return

        if name == "nose":
            local_eps = max(0.0, EPSILON_PX * 0.20)
            n = max(n, 7)
        else:
            local_eps = max(0.0, EPSILON_PX * 0.70)

        q = resample_polyline_by_arclen(pts, n=n, closed=closed)
        q = split_long_segments(q, max_segment_length=MAX_LEN)
        q = approx_rdp_open(q, epsilon_px=local_eps)

        if name == "nose" and len(q) < 2:
            q = split_long_segments(
                resample_polyline_by_arclen(pts, n=max(7, n), closed=False),
                max_segment_length=MAX_LEN
            )

        if len(q) >= 2:
            add_item(q, name=name, lock=lock)

    # face
    add_feature("oval", True,  lock=True)
    add_feature("lips", True,  lock=True)
    add_feature("nose", False, lock=True)
    add_feature("lbrow", False, lock=False)
    add_feature("rbrow", False, lock=False)

    # eyes vs glasses (spec)
    if not has_glasses:
        add_feature("leye", True, lock=True)
        add_feature("reye", True, lock=True)

    if has_glasses and glasses_parts is not None:
        for part in glasses_parts:
            if part is None or len(part) < 2:
                continue
            g = part.copy().astype(np.int32)
            g = approx_rdp_open(g, epsilon_px=max(0.0, EPSILON_PX * 0.18))
            g = split_long_segments(g, max_segment_length=MAX_LEN)
            if len(g) >= 2:
                add_item(g, name="glasses", lock=True)

    # background (kept; normally disabled by BG_STROKES=0)
    if BG_STROKES > 0:
        for p in raw_bg[:min(BG_STROKES, len(raw_bg))]:
            p1 = approx_rdp_open(p, epsilon_px=max(7.0, EPSILON_PX * 3.8 + 4.0))
            p1 = split_long_segments(p1, max_segment_length=MAX_LEN * 1.8)
            if len(p1) >= 2:
                add_item(p1, name="bg", lock=False)

    # -------------------------
    # Budget clamp (lock-aware)
    # -------------------------
    def total_points(itms):
        return int(sum(len(d["pts"]) for d in itms if d["pts"] is not None))

    items = [d for d in items if d["pts"] is not None and len(d["pts"]) >= 2]
    items.sort(key=lambda d: poly_length(d["pts"]), reverse=True)

    while len(items) > 1 and total_points(items) > TARGET_PTS:
        unlock_idxs = [i for i, d in enumerate(items) if not d["lock"]]
        if unlock_idxs:
            i_min = min(unlock_idxs, key=lambda i: poly_length(items[i]["pts"]))
            items.pop(i_min)
        else:
            items.sort(key=lambda d: poly_length(d["pts"]))
            items.pop(0)

    def reduce_one_point(d):
        p = d["pts"]
        if len(p) <= 2:
            return False
        d["pts"] = resample_polyline_by_arclen(p, n=max(2, len(p) - 1), closed=False).astype(np.int32)
        return True

    guard = 0
    while total_points(items) > TARGET_PTS and guard < 2000:
        guard += 1
        unlock = [d for d in items if not d["lock"] and len(d["pts"]) > 2]
        if unlock:
            unlock.sort(key=lambda d: len(d["pts"]), reverse=True)
            if not reduce_one_point(unlock[0]):
                break
        else:
            lock = [d for d in items if d["lock"] and len(d["pts"]) > 2]
            if not lock:
                break
            lock.sort(key=lambda d: len(d["pts"]), reverse=True)
            if not reduce_one_point(lock[0]):
                break

    polylines = [d["pts"] for d in items]

    if args.out:
        h, w = frame.shape[:2]
        out_obj = polylines_to_lineart(polylines, width=w, height=h)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        print(f"[Output] saved lineart JSON: {args.out}")

    if args.no_gui:
        return

    # -------------------------
    # Display (single window)
    # -------------------------
    canvas = render_points_and_lines(frame.shape[:2], polylines, show_points=True)

    title = f"points+lines  glasses={'ON' if has_glasses else 'OFF'}  (ESC/q to exit)"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1200, 900)
    cv2.imshow(title, canvas)

    while True:
        k = cv2.waitKey(30) & 0xFF
        if k in (27, ord("q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
