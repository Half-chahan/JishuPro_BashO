#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Face line drawing + Glasses (360deg rays -> first hit on raw)

更新点（今回）:
- 鼻が出ないことがある → Nose の簡略化を弱め、最低点数を増やし、RDP epsilon を控えめに
- メガネ:
  - 左右それぞれ「閉じた曲線」でレンズを生成（360°レイのヒット点列を閉ループ化）
  - 左右レンズ間で最短距離となる2点を探索し、その2点をブリッジの端点にする

表示色:
- 緑: 顔輪郭
- 青: glasses_raw（Canny抽出輪郭）
- シアン: 目ボックス
- 黄: レイ（走査方向の可視化。ヒット無しでも表示）
- 赤: ヒット点 + 最終メガネ線

依存:
pip install mediapipe scikit-image opencv-python
"""

import cv2
import numpy as np
import math
from skimage.morphology import thin
import mediapipe as mp


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
    x0,y0,x1,y1 = box
    x0 = max(0, min(W-1, x0))
    x1 = max(0, min(W-1, x1))
    y0 = max(0, min(H-1, y0))
    y1 = max(0, min(H-1, y1))
    if x1 < x0: x0, x1 = x1, x0
    if y1 < y0: y0, y1 = y1, y0
    return (x0,y0,x1,y1)


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


# =========================
# 360deg rays helpers
# =========================
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


# =========================
# 360deg rays from CENTER with cutoff control
# =========================
def lens_points_from_360_rays(mask: np.ndarray,
                              eye_box,
                              num_rays: int,
                              r_max: float,
                              step: float,
                              start_offset: float,
                              cap_factor: float):
    """
    - 起点: ボックス中心
    - 打ち切り: r_cap = min(r_max, t_box*cap_factor + start_offset)
    - ヒット無し: ボックス境界(+offset)を代替点として採用
    """
    x0, y0, x1, y1 = eye_box
    center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float32)

    num_rays = int(max(3, num_rays))
    cap_factor = float(max(0.1, cap_factor))
    angs = [2.0 * math.pi * i / num_rays for i in range(num_rays)]

    hits = []
    rays = []

    for a in angs:
        d = np.array([math.cos(a), math.sin(a)], dtype=np.float32)

        bpt = ray_rect_boundary_start(center, d, eye_box)
        boundary_pt = None
        t_box = None
        if bpt is not None:
            boundary_pt, t_box = bpt

        origin = center.copy()

        if t_box is not None:
            r_cap = min(float(r_max), float(t_box) * cap_factor + float(start_offset))
        else:
            r_cap = float(r_max)

        hit = raycast_first_hit(mask, origin=origin, direction=d, r_max=r_cap, step=step)

        if hit is not None:
            hits.append(hit)
            rays.append((origin.astype(np.int32), hit.astype(np.int32), True))
        else:
            if boundary_pt is not None:
                fallback = (boundary_pt + d * float(start_offset)).astype(np.int32)
            else:
                fallback = (origin + d * float(r_cap)).astype(np.int32)
            hits.append(fallback)
            rays.append((origin.astype(np.int32), fallback.astype(np.int32), False))

    if len(hits) < 3:
        return None, rays, angs

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
        return None, rays, angs

    return np.array(dedup, dtype=np.int32), rays, angs


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
    """
    A, B: (N,2), (M,2)
    戻り: (pA, pB) 最短距離の点ペア
    """
    if A is None or B is None or len(A) == 0 or len(B) == 0:
        return None, None

    Af = A.astype(np.float32)
    Bf = B.astype(np.float32)

    # ブルートフォース（点数が少ない前提）
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
                                glasses_raw: np.ndarray,
                                num_rays: int,
                                r_max_factor: float,
                                step: float,
                                start_offset: float,
                                bridge_pts: int,
                                max_len: float,
                                cap_factor: float):
    H, W = frame_shape[:2]
    _, _, eye_dist, mid_x = eye_centers_and_dist(lm_pts)

    leye_pts = features["leye"][0] if "leye" in features else None
    reye_pts = features["reye"][0] if "reye" in features else None
    if leye_pts is None or reye_pts is None:
        return None, {}

    lbox, _ = eye_box_from_feature(leye_pts, expand=1.35)
    rbox, _ = eye_box_from_feature(reye_pts, expand=1.35)
    lbox = clamp_box(lbox, W, H)
    rbox = clamp_box(rbox, W, H)

    mask = build_raw_mask((H, W), glasses_raw, thickness=2)
    r_max = max(40.0, float(eye_dist) * float(r_max_factor))

    lens_L, rays_L, _ = lens_points_from_360_rays(
        mask, lbox, num_rays=num_rays, r_max=r_max, step=step,
        start_offset=start_offset, cap_factor=cap_factor
    )
    lens_R, rays_R, _ = lens_points_from_360_rays(
        mask, rbox, num_rays=num_rays, r_max=r_max, step=step,
        start_offset=start_offset, cap_factor=cap_factor
    )

    if lens_L is None and lens_R is None:
        return None, {"eye_boxes": (lbox, rbox), "mask": mask, "rays_L": rays_L, "rays_R": rays_R, "r_max": r_max}

    # 片方しか取れないときは mirror
    if lens_L is None and lens_R is not None:
        lens_L = mirror_poly_x(lens_R, mid_x=mid_x)
    if lens_R is None and lens_L is not None:
        lens_R = mirror_poly_x(lens_L, mid_x=mid_x)

    # レンズは閉ループにする
    lens_Lc = close_loop(lens_L) if lens_L is not None else None
    lens_Rc = close_loop(lens_R) if lens_R is not None else None

    # ブリッジ：左右レンズの最短距離の2点
    bridge = None
    pL, pR = closest_points_between_polys(lens_L, lens_R)
    if pL is not None and pR is not None:
        bridge_pts = int(max(2, bridge_pts))
        bridge = []
        for i in range(bridge_pts):
            t = i / (bridge_pts - 1)
            p = (1 - t) * pL.astype(np.float32) + t * pR.astype(np.float32)
            bridge.append(p.astype(np.int32))
        bridge = np.array(bridge, dtype=np.int32)

    # 出力は「別ポリラインのリスト」にする（閉曲線同士を勝手に連結しない）
    parts = []
    if lens_Lc is not None and len(lens_Lc) >= 4:
        parts.append(lens_Lc.astype(np.int32))
    if bridge is not None and len(bridge) >= 2:
        parts.append(bridge.astype(np.int32))
    if lens_Rc is not None and len(lens_Rc) >= 4:
        parts.append(lens_Rc.astype(np.int32))

    # それぞれの中で長いセグメントを分割（見た目用）
    parts = [split_long_segments(p, max_segment_length=max_len) for p in parts]

    dbg = {
        "eye_boxes": (lbox, rbox),
        "lens_L": lens_Lc,
        "lens_R": lens_Rc,
        "bridge": bridge,
        "mask": mask,
        "rays_L": rays_L,
        "rays_R": rays_R,
        "r_max": r_max,
        "mid_x": mid_x
    }
    return parts, dbg


# =========================
# Points allocation
# =========================
def allocate_points(target_total: int, has_glasses: bool):
    alloc = {
        "oval": 16,
        "lips": 10,
        "nose": 8,   # ← 鼻が消えやすいので増やす
        "lbrow": 5,
        "rbrow": 5,
        "leye": 4,
        "reye": 4,
        "glasses": 0,
        "bg": 0
    }
    if has_glasses:
        alloc["glasses"] = 26   # ← 左右レンズ(閉)＋ブリッジで増やす
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
# Drawing / debug
# =========================
def draw_polylines(canvas: np.ndarray, polylines, offset_y=0, show_points=False, color=(0,0,0)):
    for pts in polylines:
        if pts is None or len(pts) < 2:
            continue
        pts2 = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        pts2[:, 0, 1] += offset_y
        cv2.polylines(canvas, [pts2], isClosed=False, color=color, thickness=1)
        if show_points:
            for (x, y) in pts:
                cv2.circle(canvas, (int(x), int(y + offset_y)), 3, (0, 0, 255), -1)

def make_lines_only_canvas(shape, polylines):
    h, w = shape[:2]
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    draw_polylines(canvas, polylines, offset_y=0, show_points=False, color=(0,0,0))
    return canvas

def make_tuner_canvas(shape, polylines, header_h=120, show_points=True, header_lines=None):
    h, w = shape[:2]
    canvas = np.ones((h + header_h, w, 3), dtype=np.uint8) * 255
    cv2.rectangle(canvas, (5, 5), (w - 5, header_h - 5), (0, 0, 0), 2)
    if header_lines:
        y = 40
        for s, col in header_lines:
            cv2.putText(canvas, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.78, col, 2, cv2.LINE_AA)
            y += 35
    draw_polylines(canvas, polylines, offset_y=header_h, show_points=show_points, color=(0,0,0))
    return canvas

def draw_eye_boxes(img, eye_boxes):
    if eye_boxes is None:
        return
    lbox, rbox = eye_boxes
    for (x0,y0,x1,y1) in [lbox, rbox]:
        cv2.rectangle(img, (x0,y0), (x1,y1), (255, 200, 0), 2)

def draw_rays(img, rays, color=(0, 255, 255)):
    if rays is None:
        return
    for (s, e, ok) in rays:
        s = tuple(map(int, s))
        e = tuple(map(int, e))
        cv2.line(img, s, e, color, 2)
        cv2.circle(img, s, 3, color, -1)
        if ok:
            cv2.circle(img, e, 6, (0, 0, 255), -1)  # hit

def make_debug_overlay_lines_only(shape, oval_pts, glasses_raw, glasses_parts, eye_boxes=None, rays_L=None, rays_R=None):
    h, w = shape[:2]
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    draw_eye_boxes(canvas, eye_boxes)
    draw_rays(canvas, rays_L)
    draw_rays(canvas, rays_R)

    if oval_pts is not None:
        cv2.polylines(canvas, [oval_pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
    if glasses_raw is not None:
        cv2.polylines(canvas, [glasses_raw.reshape(-1, 1, 2)], True, (255, 0, 0), 2)

    if glasses_parts is not None:
        for p in glasses_parts:
            if p is None or len(p) < 2:
                continue
            cv2.polylines(canvas, [p.reshape(-1, 1, 2)], False, (0, 0, 255), 2)
    return canvas


# =========================
# Main
# =========================
def main():
    frame = capture_frame_with_preview(0)
    cv2.imwrite("captured_frame.jpg", frame)
    print("[Debug] saved captured_frame.jpg")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    win = "tuner"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1200, 900)

    cv2.namedWindow("lines_only", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("lines_only", 1200, 900)

    cv2.namedWindow("debug_original", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("debug_original", 1200, 900)

    cv2.namedWindow("debug_overlay_lines_only", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("debug_overlay_lines_only", 1200, 900)

    cv2.namedWindow("debug_edges_roi", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("debug_edges_roi", 900, 600)

    # sliders
    cv2.createTrackbar("epsilon_px", win, 4, 30, lambda x: None)
    cv2.createTrackbar("max_len", win, 220, 800, lambda x: None)
    cv2.createTrackbar("bg_strokes", win, 0, 40, lambda x: None)
    cv2.createTrackbar("target_pts", win, 60, 220, lambda x: None)
    cv2.createTrackbar("show_pts", win, 1, 1, lambda x: None)

    # raw glasses
    cv2.createTrackbar("gl_canny_lo", win, 25, 80, lambda x: None)
    cv2.createTrackbar("gl_canny_hi", win, 90, 200, lambda x: None)

    # 360 rays params
    cv2.createTrackbar("num_rays", win, 18, 80, lambda x: None)
    cv2.createTrackbar("r_max%", win, 220, 420, lambda x: None)
    cv2.createTrackbar("start_offset", win, 6, 40, lambda x: None)
    cv2.createTrackbar("step", win, 1, 4, lambda x: None)

    # cutoff control
    cv2.createTrackbar("cap_factor%", win, 160, 600, lambda x: None)   # 1.6x default, up to 6.0x

    cv2.createTrackbar("bridge_pts", win, 3, 10, lambda x: None)

    HEADER_H = 120

    # FaceMesh one-shot
    features, lm_pts = extract_facemesh_feature_polylines_and_landmarks(frame)
    if features is None:
        print("[FaceMesh] 顔が取れませんでした（FaceMesh）")
        return

    oval_pts = features["oval"][0] if "oval" in features else None

    # background one-shot
    edges_bg = detect_edges_background_coarse(gray)
    skel_bg = skeletonize_edges(edges_bg)
    raw_bg = find_raw_polylines_from_skeleton(skel_bg)

    while True:
        eps = float(cv2.getTrackbarPos("epsilon_px", win))
        max_len = float(max(10, cv2.getTrackbarPos("max_len", win)))
        bg_max_strokes = int(max(0, cv2.getTrackbarPos("bg_strokes", win)))
        target_pts = int(cv2.getTrackbarPos("target_pts", win))
        show_pts = (cv2.getTrackbarPos("show_pts", win) == 1)

        gl_lo = int(cv2.getTrackbarPos("gl_canny_lo", win))
        gl_hi = int(max(gl_lo + 1, cv2.getTrackbarPos("gl_canny_hi", win)))

        num_rays = int(max(5, cv2.getTrackbarPos("num_rays", win)))
        r_max_factor = float(cv2.getTrackbarPos("r_max%", win)) / 100.0
        start_offset = float(cv2.getTrackbarPos("start_offset", win))
        step = float(max(1, cv2.getTrackbarPos("step", win)))
        cap_factor = float(max(10, cv2.getTrackbarPos("cap_factor%", win))) / 100.0
        bridge_pts = int(max(2, cv2.getTrackbarPos("bridge_pts", win)))

        # glasses raw
        glasses_raw, roi_rect, edges_roi = (None, None, None)
        if oval_pts is not None:
            glasses_raw, roi_rect, edges_roi = extract_glasses_outline(gray, oval_pts, gl_lo, gl_hi)
        has_raw = glasses_raw is not None and len(glasses_raw) >= 80

        # glasses final (360 rays)
        glasses_parts = None
        glasses_dbg = {}
        eye_boxes = None
        rays_L = rays_R = None

        if has_raw:
            glasses_parts, glasses_dbg = build_glasses_from_360_rays(
                frame_shape=frame.shape,
                lm_pts=lm_pts,
                features=features,
                glasses_raw=glasses_raw,
                num_rays=num_rays,
                r_max_factor=r_max_factor,
                step=step,
                start_offset=start_offset,
                bridge_pts=bridge_pts,
                max_len=max_len,
                cap_factor=cap_factor
            )
            eye_boxes = glasses_dbg.get("eye_boxes", None)
            rays_L = glasses_dbg.get("rays_L", None)
            rays_R = glasses_dbg.get("rays_R", None)

        has_glasses = glasses_parts is not None and len(glasses_parts) >= 1

        # allocate
        alloc = allocate_points(target_pts, has_glasses=has_glasses)

        polylines: list[np.ndarray] = []

        def add_feature(name: str, closed_default: bool):
            if name not in features:
                return
            pts, closed = features[name]
            closed = bool(closed) if name in ("oval", "lips", "leye", "reye") else closed_default
            n = int(alloc.get(name, 0))
            if n <= 0:
                return

            # 鼻は簡略化しすぎると消えやすいので特別扱い（epsilonを弱める）
            if name == "nose":
                local_eps = max(0.0, eps * 0.20)  # ← 鼻だけRDP弱め
                n = max(n, 7)
            else:
                local_eps = max(0.0, eps * 0.70)

            q = resample_polyline_by_arclen(pts, n=n, closed=closed)
            q = split_long_segments(q, max_segment_length=max_len)
            q = approx_rdp_open(q, epsilon_px=local_eps)

            # 鼻が短すぎるときは、RDPなしの点列に戻す（“消える”対策）
            if name == "nose" and len(q) < 2:
                q = split_long_segments(resample_polyline_by_arclen(pts, n=max(7, n), closed=False),
                                        max_segment_length=max_len)

            if len(q) >= 2:
                polylines.append(q.astype(np.int32))

        add_feature("oval", True)
        add_feature("lips", True)
        add_feature("nose", False)
        add_feature("lbrow", False)
        add_feature("rbrow", False)
        add_feature("leye", True)
        add_feature("reye", True)

        if has_glasses:
            # それぞれ独立ポリラインとして追加（閉ループも保持）
            for part in glasses_parts:
                if part is None or len(part) < 2:
                    continue
                g = part.copy().astype(np.int32)
                # メガネはRDP控えめ（形を保つ）
                g = approx_rdp_open(g, epsilon_px=max(0.0, eps * 0.18))
                g = split_long_segments(g, max_segment_length=max_len)
                if len(g) >= 2:
                    polylines.append(g.astype(np.int32))

        # background (optional)
        if bg_max_strokes > 0:
            for p in raw_bg[:min(bg_max_strokes, len(raw_bg))]:
                p1 = approx_rdp_open(p, epsilon_px=max(7.0, eps * 3.8 + 4.0))
                p1 = split_long_segments(p1, max_segment_length=max_len * 1.8)
                if len(p1) >= 2:
                    polylines.append(p1.astype(np.int32))

        # point budget clamp
        def total_points(pls): return sum(len(p) for p in pls if p is not None)
        polylines = [p for p in polylines if p is not None and len(p) >= 2]

        polylines.sort(key=poly_length, reverse=True)
        while len(polylines) > 1 and total_points(polylines) > target_pts:
            polylines.pop(-1)

        while total_points(polylines) > target_pts and len(polylines) > 0:
            polylines.sort(key=lambda p: len(p), reverse=True)
            p = polylines[0]
            if len(p) <= 2:
                break
            polylines[0] = resample_polyline_by_arclen(p, n=max(2, len(p) - 1), closed=False)

        # canvases
        actual = total_points(polylines)
        gline = (
            f"glasses={'ON' if has_glasses else 'OFF'} raw={has_raw} "
            f"num_rays={num_rays} r_max%={int(r_max_factor*100)} "
            f"start_offset={start_offset:.0f} step={step:.0f} cap_factor={cap_factor:.2f}"
        )

        tuner = make_tuner_canvas(
            frame.shape,
            polylines,
            header_h=HEADER_H,
            show_points=show_pts,
            header_lines=[
                (f"target_pts={target_pts} actual_pts={actual}", (0, 0, 0)),
                (f"eps={int(eps)} max_len={int(max_len)} bg={bg_max_strokes} gl_canny={gl_lo}/{gl_hi} bridge={bridge_pts}", (0, 0, 255)),
                (gline, (0, 120, 0)),
            ],
        )
        lines_only = make_lines_only_canvas(frame.shape, polylines)

        cv2.imshow(win, tuner)
        cv2.imshow("lines_only", lines_only)

        # debug_original
        vis = frame.copy()

        if eye_boxes is not None:
            draw_eye_boxes(vis, eye_boxes)

        if oval_pts is not None:
            cv2.polylines(vis, [oval_pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2)

        if has_raw:
            cv2.polylines(vis, [glasses_raw.reshape(-1, 1, 2)], True, (255, 0, 0), 2)

        draw_rays(vis, rays_L)
        draw_rays(vis, rays_R)

        if has_glasses and glasses_parts is not None:
            for p in glasses_parts:
                if p is None or len(p) < 2:
                    continue
                cv2.polylines(vis, [p.reshape(-1, 1, 2)], False, (0, 0, 255), 2)

        if roi_rect is not None:
            x0, y0, x1, y1 = roi_rect
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 200, 0), 2)

        cv2.imshow("debug_original", vis)

        overlay = make_debug_overlay_lines_only(
            frame.shape, oval_pts,
            glasses_raw if has_raw else None,
            glasses_parts if has_glasses else None,
            eye_boxes=eye_boxes,
            rays_L=rays_L,
            rays_R=rays_R
        )
        cv2.imshow("debug_overlay_lines_only", overlay)

        if edges_roi is not None:
            e3 = cv2.cvtColor(edges_roi, cv2.COLOR_GRAY2BGR)
            cv2.imshow("debug_edges_roi", e3)
        else:
            cv2.imshow("debug_edges_roi", np.zeros((240, 320, 3), dtype=np.uint8))

        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
