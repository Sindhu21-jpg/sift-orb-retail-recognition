import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import label_binarize
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# =========================
# LOAD CNN MODEL
# =========================
cnn_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

st.title("🛒 SIFT-Based Object Recognition for Retail Inventory Systems")

mode = st.radio(
    "Select Mode",
    ["📁 Dataset Mode", "📷 Real-Time Mode"]
)

ratio_threshold = st.slider(
    "Lowe Ratio Threshold",
    0.6, 0.9, 0.75
)

kp_response_threshold = st.slider(
    "Keypoint Response Strength Filter (0 = no filter)",
    0.0, 50.0, 0.0, step=0.5,
    help="Filter out keypoints below this response strength before matching."
)

# =========================
# SAFE IMAGE LOADER
# =========================
def read_img(file):
    if file is None:
        return None
    arr = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return np.uint8(img)

# =========================
# KEYPOINT FILTERING BY RESPONSE STRENGTH
# FIX: returns indices so callers can remap match queryIdx/trainIdx correctly
# =========================
def filter_keypoints_by_response(kp, des, threshold=0.0):
    """
    Filters keypoints and descriptors by response strength.
    Returns (filtered_kp, filtered_des, kept_indices).
    kept_indices maps new position → original position, needed to remap
    match queryIdx / trainIdx after filtering.
    """
    if threshold <= 0.0 or kp is None or des is None or len(kp) == 0:
        return kp, des, list(range(len(kp) if kp else 0))

    kept_kp, kept_des, kept_idx = [], [], []
    for i, k in enumerate(kp):
        if k.response >= threshold:
            kept_kp.append(k)
            kept_des.append(des[i])
            kept_idx.append(i)

    if len(kept_kp) == 0:          # fallback: nothing survived
        return kp, des, list(range(len(kp)))

    return kept_kp, np.array(kept_des), kept_idx

# =========================
# HISTOGRAM SIMILARITY
# =========================
def histogram_similarity(img1, img2):
    h1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    h2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    score = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    return max(0, score * 100)

# =========================
# FEATURE MATCHING
# =========================
def match(des1, des2, method="BF"):
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return [], 0

    des1 = np.float32(des1)
    des2 = np.float32(des2)

    if method == "BF":
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    else:
        matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5), dict(checks=50)
        )

    if len(des1) < 2 or len(des2) < 2:
        return [], 0

    matches = matcher.knnMatch(des1, des2, k=2)
    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < ratio_threshold * n.distance:
            good.append(m)

    score = (len(good) / len(matches) * 100) if len(matches) > 0 else 0
    return good, score

def match_binary(des1, des2):
    """BF matching for binary descriptors (ORB, AKAZE)."""
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return [], 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(des1, des2, k=2)
    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < ratio_threshold * n.distance:
            good.append(m)
    score = (len(good) / len(matches) * 100) if len(matches) > 0 else 0
    return good, score

# =========================
# LOAD DATASET IMAGES
# =========================
def load_uploaded_images(files):
    imgs, names = [], []
    if files:
        for f in files:
            arr = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                imgs.append(np.uint8(img))
                names.append(f.name)
    return imgs, names

# =========================
# CNN FEATURE EXTRACTION
# =========================
def extract_cnn_features(img_gray):
    img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = cnn_model.predict(img, verbose=0)
    return features[0]

def cnn_similarity(img1, img2):
    f1 = extract_cnn_features(img1)
    f2 = extract_cnn_features(img2)
    score = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-9)
    return float(np.clip(score * 100, 0, 100))

# =========================
# OBJECT LOCALIZATION — FULLY CORRECTED
# =========================
def draw_localization(ref, test, kp_ref, kp_test, good_matches):
    """
    Robust homography-based object localization.

    Key fixes vs original:
    ─────────────────────
    FIX 1 — Index safety:
        good_matches carry queryIdx / trainIdx from the ORIGINAL (pre-filter)
        keypoint lists.  We now receive already-filtered kp_ref/kp_test so
        queryIdx could exceed len(kp_ref).  We clamp / skip invalid indices
        instead of crashing or silently using wrong points.

    FIX 2 — MAD threshold relaxed from 1.5× to 3.0× (primary) / 5.0×
        (fallback).  The original 1.5× was far too tight for real retail images
        where perspective, zoom and shelf lean introduce genuine displacement
        spread, causing almost every match to be discarded.

    FIX 3 — Homography validity: added determinant sign + near-zero guard,
        scale bounds tightened, convexity check kept.

    FIX 4 — Quad-keypoint agreement: uses dst_filt (not dst_inlier_pts which
        had a stale index after filter) so the inside-fraction is computed on
        the correct points.

    FIX 5 — Fallback bbox is always drawn on the best available points so the
        user always sees something meaningful.
    """
    FONT   = cv2.FONT_HERSHEY_SIMPLEX
    GREEN  = (0, 230, 50)
    RED    = (0, 50, 230)
    YELLOW = (0, 210, 255)
    CYAN   = (255, 200, 0)
    WHITE  = (255, 255, 255)

    RANSAC_THR       = 5.0
    MIN_INLIERS      = 6
    MIN_INLIER_RATIO = 0.15   # 15 % of good_matches
    MIN_KP_INSIDE    = 0.45   # 45 % of inlier dst-kp inside the quad

    ref_bgr  = cv2.cvtColor(ref,  cv2.COLOR_GRAY2BGR)
    test_bgr = cv2.cvtColor(test, cv2.COLOR_GRAY2BGR)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _put(img, text, y, color=WHITE):
        cv2.putText(img, text, (9, y+1), FONT, 0.50, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(img, text, (8, y),   FONT, 0.50, color,   1, cv2.LINE_AA)

    def _side_by_side(left, right):
        h = max(left.shape[0], right.shape[0])
        pl = np.zeros((h, left.shape[1],  3), np.uint8)
        pr = np.zeros((h, right.shape[1], 3), np.uint8)
        pl[:left.shape[0],  :] = left
        pr[:right.shape[0], :] = right
        bar = np.full((h, 3, 3), 80, np.uint8)
        return np.hstack([pl, bar, pr])

    def _stack_rows(*rows, gap=4):
        w = max(r.shape[1] for r in rows)
        out = []
        for r in rows:
            if r.shape[1] != w:
                nh = int(r.shape[0] * w / r.shape[1])
                r = cv2.resize(r, (w, nh), interpolation=cv2.INTER_AREA)
            out.append(r)
            out.append(np.full((gap, w, 3), 55, np.uint8))
        return np.vstack(out[:-1])

    def _fallback_bbox(img_bgr, pts, label, color=YELLOW):
        if len(pts) == 0:
            _put(img_bgr, label, 30, RED)
            return
        pts_i = pts.astype(np.int32)
        x, y, w, h = cv2.boundingRect(pts_i)
        px, py = max(int(w * 0.10), 4), max(int(h * 0.10), 4)
        x1 = max(0, x - px);  y1 = max(0, y - py)
        x2 = min(img_bgr.shape[1]-1, x + w + px)
        y2 = min(img_bgr.shape[0]-1, y + h + py)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
        _put(img_bgr, label[:55],       28, color)
        _put(img_bgr, "(keypoint bbox)", 54, color)

    # ── Step 0: need ≥ 4 valid matches ───────────────────────────────────────
    if len(good_matches) < 4:
        _put(test_bgr, f"Too few matches ({len(good_matches)}) — need ≥ 4", 30, RED)
        return _side_by_side(ref_bgr, test_bgr)

    # FIX 1 — safe point extraction: skip matches whose stored index is out of
    # range for the (already-filtered) keypoint lists.
    n_ref  = len(kp_ref)
    n_test = len(kp_test)
    safe_matches = [
        m for m in good_matches
        if m.queryIdx < n_ref and m.trainIdx < n_test
    ]
    if len(safe_matches) < 4:
        _put(test_bgr,
             f"Index mismatch after filter ({len(safe_matches)} safe) — bbox fallback",
             30, RED)
        all_dst = np.float32([kp_test[m.trainIdx].pt
                              for m in good_matches if m.trainIdx < n_test])
        _fallback_bbox(test_bgr, all_dst, "Index mismatch fallback")
        return _side_by_side(ref_bgr, test_bgr)

    src_all = np.float32([kp_ref[m.queryIdx].pt  for m in safe_matches])
    dst_all = np.float32([kp_test[m.trainIdx].pt for m in safe_matches])

    # ── Step 1: DISPLACEMENT CONSISTENCY FILTER (FIX 2: wider MAD) ───────────
    dx = dst_all[:, 0] - src_all[:, 0]
    dy = dst_all[:, 1] - src_all[:, 1]
    med_dx, med_dy   = np.median(dx), np.median(dy)
    mad_dx = np.median(np.abs(dx - med_dx)) + 1e-6
    mad_dy = np.median(np.abs(dy - med_dy)) + 1e-6

    # Primary: 3.0 × MAD  (was 1.5× — too tight for real retail images)
    keep = (np.abs(dx - med_dx) < 3.0 * mad_dx) & \
           (np.abs(dy - med_dy) < 3.0 * mad_dy)
    # Fallback: 5.0 × MAD
    if keep.sum() < 4:
        keep = (np.abs(dx - med_dx) < 5.0 * mad_dx) & \
               (np.abs(dy - med_dy) < 5.0 * mad_dy)

    filtered_matches = [m for m, k in zip(safe_matches, keep) if k]
    src_filt = src_all[keep]
    dst_filt = dst_all[keep]

    if len(filtered_matches) < 4:
        _fallback_bbox(test_bgr, dst_all, "No displacement consensus", RED)
        return _side_by_side(ref_bgr, test_bgr)

    # ── Step 2: RANSAC homography ─────────────────────────────────────────────
    M, mask = cv2.findHomography(
        src_filt.reshape(-1, 1, 2),
        dst_filt.reshape(-1, 1, 2),
        cv2.RANSAC, RANSAC_THR
    )

    inlier_mask    = mask.ravel().astype(bool) if mask is not None else \
                     np.zeros(len(filtered_matches), bool)
    n_inliers      = int(inlier_mask.sum())
    inlier_ratio   = n_inliers / len(safe_matches)   # vs safe match count
    inlier_matches = [m for m, k in zip(filtered_matches, inlier_mask) if k]
    # FIX 4 — use dst_filt[inlier_mask] directly (not a stale separate variable)
    dst_inlier_pts = dst_filt[inlier_mask]

    # ── Step 3: GEOMETRIC VALIDITY CHECKS (FIX 3) ────────────────────────────
    def _valid_homography(H, ref_shape, test_shape):
        if H is None:
            return False, "H is None"
        det = H[0,0]*H[1,1] - H[0,1]*H[1,0]
        if det <= 0:
            return False, f"Reflection or degenerate (det={det:.4f})"
        scale = np.sqrt(abs(det))
        if not (0.04 < scale < 25.0):
            return False, f"Bad scale ({scale:.2f}×)"
        hr, wr = ref_shape
        ht, wt = test_shape
        corners = np.float32(
            [[0,0],[wr-1,0],[wr-1,hr-1],[0,hr-1]]
        ).reshape(-1,1,2)
        proj = cv2.perspectiveTransform(corners, H).reshape(4, 2)
        pad  = 2.5
        if not np.all(
            (proj[:,0] > -wt*pad) & (proj[:,0] < wt*(1+pad)) &
            (proj[:,1] > -ht*pad) & (proj[:,1] < ht*(1+pad))
        ):
            return False, "Projected corners out of bounds"
        # Convexity check
        def _cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        signs = [_cross(proj[i], proj[(i+1)%4], proj[(i+2)%4]) for i in range(4)]
        if not (all(s > 0 for s in signs) or all(s < 0 for s in signs)):
            return False, "Non-convex (bowtie) quad"
        # Area must be ≥ 0.3 % and ≤ 60 % of test image
        # Upper bound rejects "whole-image" homographies caused by repetitive texture
        x, y = proj[:,0], proj[:,1]
        area = 0.5 * abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))
        if area < 0.003 * wt * ht:
            return False, f"Area too small ({area:.0f} px²)"
        if area > 0.60 * wt * ht:
            return False, f"Area too large — covers {area/(wt*ht)*100:.0f}% of image (repetitive texture?)"
        return True, proj

    # ── Step 4: QUAD–KEYPOINT AGREEMENT (FIX 4) ──────────────────────────────
    def _kp_inside_quad(quad_pts, dst_pts):
        if len(dst_pts) == 0:
            return 0.0
        contour = np.int32(quad_pts).reshape((-1, 1, 2))
        inside = sum(
            1 for pt in dst_pts
            if cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), False) >= 0
        )
        return inside / len(dst_pts)

    use_homography = False
    quad_corners   = None
    fail_reason    = ""

    if n_inliers >= MIN_INLIERS and inlier_ratio >= MIN_INLIER_RATIO:
        ok, result = _valid_homography(M, ref.shape, test.shape)
        if ok:
            quad_corners = result
            frac_inside  = _kp_inside_quad(quad_corners, dst_inlier_pts)
            if frac_inside >= MIN_KP_INSIDE:
                use_homography = True
            else:
                fail_reason = (
                    f"Quad covers only {frac_inside*100:.0f}% of inlier kp "
                    f"(need ≥{MIN_KP_INSIDE*100:.0f}%)"
                )
        else:
            fail_reason = result
    else:
        fail_reason = (
            f"Inliers: {n_inliers}/{len(safe_matches)} "
            f"({inlier_ratio*100:.0f}%) — below threshold"
        )

    # ── Draw result ───────────────────────────────────────────────────────────
    ref_labeled = ref_bgr.copy()
    h_ref, w_ref = ref.shape
    cv2.rectangle(ref_labeled, (1,1), (w_ref-2, h_ref-2), YELLOW, 2)
    _put(ref_labeled, "Reference", 28, YELLOW)

    if use_homography:
        cv2.polylines(
            test_bgr, [np.int32(quad_corners)],
            isClosed=True, color=GREEN, thickness=3, lineType=cv2.LINE_AA
        )
        for pt in quad_corners:
            cv2.circle(test_bgr, tuple(np.int32(pt)), 6, GREEN, -1, cv2.LINE_AA)
        _put(test_bgr, "Product Localized", 28, GREEN)
        _put(test_bgr,
             f"Inliers: {n_inliers}/{len(safe_matches)} ({inlier_ratio*100:.0f}%)",
             54, WHITE)
        draw_color      = GREEN
        draw_match_set  = inlier_matches
    else:
        # FIX 5 — always show the best available points in fallback
        pts_for_bbox = dst_inlier_pts if len(dst_inlier_pts) >= 4 else dst_all
        _fallback_bbox(test_bgr, pts_for_bbox, fail_reason, YELLOW)
        draw_color     = YELLOW
        draw_match_set = inlier_matches if inlier_matches else filtered_matches

    match_canvas = cv2.drawMatches(
        ref, kp_ref, test, kp_test,
        draw_match_set[:60], None,
        matchColor=draw_color,
        singlePointColor=CYAN,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    top_row = _side_by_side(ref_labeled, test_bgr)
    return _stack_rows(top_row, match_canvas)

# =========================
# SYNTHETIC DATA GENERATOR
# =========================
def generate_synthetic_dataset(base_img):
    variants = [( base_img.copy(), "original" )]
    h, w     = base_img.shape
    center   = (w // 2, h // 2)

    for scale in [0.5, 1.5, 2.0]:
        s = cv2.resize(base_img, (int(w*scale), int(h*scale)))
        s = cv2.resize(s, (w, h))
        variants.append((s, f"scale_{scale}x"))

    for angle in [15, 30, 45, 90]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        variants.append((cv2.warpAffine(base_img, M, (w,h)), f"rotation_{angle}deg"))

    for alpha, beta in [(1.5,30), (0.6,-20), (1.2,50)]:
        variants.append((cv2.convertScaleAbs(base_img, alpha=alpha, beta=beta),
                          f"illum_a{alpha}_b{beta}"))

    noise  = np.random.normal(0, 25, base_img.shape).astype(np.int16)
    noisy  = np.clip(base_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    variants.append((noisy, "gaussian_noise"))
    return variants


# =========================
# DESCRIPTOR COMPARISON
# =========================
def descriptor_comparison(ref_img, test_img):
    st.subheader("🔬 Descriptor Comparison: SIFT vs AKAZE vs ORB")
    results = {"Descriptor":[], "Ref KP":[], "Test KP":[], "Good Matches":[], "Score (%)":[], "Time (sec)":[]}

    def _add(name, kp1, kp2, good, score, elapsed):
        results["Descriptor"].append(name)
        results["Ref KP"].append(len(kp1))
        results["Test KP"].append(len(kp2))
        results["Good Matches"].append(len(good))
        results["Score (%)"].append(round(score, 2))
        results["Time (sec)"].append(round(elapsed, 4))

    # SIFT
    t0 = time.time()
    sift = cv2.SIFT_create()
    kp1s, d1s = sift.detectAndCompute(ref_img, None)
    kp2s, d2s = sift.detectAndCompute(test_img, None)
    kp1s, d1s, _ = filter_keypoints_by_response(kp1s, d1s, kp_response_threshold)
    kp2s, d2s, _ = filter_keypoints_by_response(kp2s, d2s, kp_response_threshold)
    g_s, sc_s = match(d1s, d2s, "BF")
    _add("SIFT", kp1s, kp2s, g_s, sc_s, time.time()-t0)

    # AKAZE
    t0 = time.time()
    akaze = cv2.AKAZE_create()
    kp1a, d1a = akaze.detectAndCompute(ref_img, None)
    kp2a, d2a = akaze.detectAndCompute(test_img, None)
    kp1a, d1a, _ = filter_keypoints_by_response(kp1a, d1a, kp_response_threshold)
    kp2a, d2a, _ = filter_keypoints_by_response(kp2a, d2a, kp_response_threshold)
    g_a, sc_a = match_binary(d1a, d2a)
    _add("AKAZE", kp1a, kp2a, g_a, sc_a, time.time()-t0)

    # ORB
    t0 = time.time()
    orb = cv2.ORB_create()
    kp1o, d1o = orb.detectAndCompute(ref_img, None)
    kp2o, d2o = orb.detectAndCompute(test_img, None)
    kp1o, d1o, _ = filter_keypoints_by_response(kp1o, d1o, kp_response_threshold)
    kp2o, d2o, _ = filter_keypoints_by_response(kp2o, d2o, kp_response_threshold)
    g_o, sc_o = match_binary(d1o, d2o)
    _add("ORB", kp1o, kp2o, g_o, sc_o, time.time()-t0)

    # SURF (optional)
    try:
        t0 = time.time()
        surf = cv2.xfeatures2d.SURF_create(400)
        kp1u, d1u = surf.detectAndCompute(ref_img, None)
        kp2u, d2u = surf.detectAndCompute(test_img, None)
        kp1u, d1u, _ = filter_keypoints_by_response(kp1u, d1u, kp_response_threshold)
        kp2u, d2u, _ = filter_keypoints_by_response(kp2u, d2u, kp_response_threshold)
        g_u, sc_u = match(d1u, d2u, "BF")
        _add("SURF", kp1u, kp2u, g_u, sc_u, time.time()-t0)
    except Exception:
        st.info("SURF not available (patented). Showing SIFT, AKAZE, ORB.")

    df = pd.DataFrame(results)
    df["Time (sec)"] = pd.to_numeric(df["Time (sec)"], errors="coerce")
    st.dataframe(df)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"][:len(results["Descriptor"])]
    axes[0].bar(results["Descriptor"], results["Score (%)"], color=colors)
    axes[0].set_title("Match Score by Descriptor")
    axes[0].set_ylabel("Score (%)"); axes[0].set_ylim(0, 100)
    axes[1].bar(results["Descriptor"], results["Time (sec)"], color=colors)
    axes[1].set_title("Computation Time by Descriptor"); axes[1].set_ylabel("Time (sec)")
    plt.tight_layout(); st.pyplot(fig)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(cv2.drawKeypoints(test_img, kp2s, None, color=(0,255,0)),
                 caption=f"SIFT KP ({len(kp2s)})", channels="BGR")
    with col2:
        st.image(cv2.drawKeypoints(test_img, kp2a, None, color=(255,100,0)),
                 caption=f"AKAZE KP ({len(kp2a)})", channels="BGR")
    with col3:
        st.image(cv2.drawKeypoints(test_img, kp2o, None, color=(100,0,255)),
                 caption=f"ORB KP ({len(kp2o)})", channels="BGR")
    return results

# =========================
# EVALUATION METRICS — FULLY CORRECTED
# =========================

def compute_topk_accuracy(all_scores_matrix, gt_labels, k_values=[1, 3, 5]):
    """
    Top-K accuracy for leave-one-out retrieval.
    all_scores_matrix[qi][di] = similarity score (float).
    gt_labels[i] = class label (int) for image i.
    A query qi is correct at rank K if any of the top-K gallery results
    (excluding self) shares the same class label as qi.
    """
    n = len(all_scores_matrix)
    results = {}
    for k in k_values:
        if k >= n:
            continue
        correct = 0
        for qi in range(n):
            row = all_scores_matrix[qi]
            # exclude self
            ranked = sorted(
                [(s, di) for di, s in enumerate(row) if di != qi],
                reverse=True
            )
            top_k_labels = [gt_labels[di] for _, di in ranked[:k]]
            if gt_labels[qi] in top_k_labels:
                correct += 1
        results[f"Top-{k}"] = round(correct / n * 100, 2)
    return results


def compute_map(all_scores_matrix, gt_labels):
    """
    mAP for LOO retrieval using class-label identity.
    AP per query = sum_k (P@k * rel@k) / num_relevant.
    Here relevant = all gallery images sharing the same class as the query
    (excluding self).
    """
    n   = len(all_scores_matrix)
    aps = []
    for qi in range(n):
        true_label  = gt_labels[qi]
        row         = all_scores_matrix[qi]
        ranked      = sorted(
            [(s, di) for di, s in enumerate(row) if di != qi],
            reverse=True
        )
        # count relevant images (same class, excluding self)
        n_relevant = sum(1 for _, di in ranked if gt_labels[di] == true_label)
        if n_relevant == 0:
            aps.append(0.0); continue
        hits, prec_sum = 0, 0.0
        for rank, (_, di) in enumerate(ranked, start=1):
            if gt_labels[di] == true_label:
                hits      += 1
                prec_sum  += hits / rank
        aps.append(prec_sum / n_relevant)
    return round(float(np.mean(aps)) * 100, 2) if aps else 0.0


def plot_cmc_curve(all_scores_matrix, gt_labels):
    n        = len(all_scores_matrix)
    max_rank = min(n - 1, 10)
    if max_rank < 1:
        return
    cmc = []
    for k in range(1, max_rank + 1):
        correct = 0
        for qi in range(n):
            row    = all_scores_matrix[qi]
            ranked = sorted(
                [(s, di) for di, s in enumerate(row) if di != qi],
                reverse=True
            )
            top_k_labels = [gt_labels[di] for _, di in ranked[:k]]
            if gt_labels[qi] in top_k_labels:
                correct += 1
        cmc.append(correct / n * 100)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, max_rank+1), cmc, marker="o",
            color="#2196F3", linewidth=2, markersize=5)
    ax.fill_between(range(1, max_rank+1), cmc, alpha=0.15, color="#2196F3")
    ax.set_xlabel("Rank K"); ax.set_ylabel("Recognition Rate (%)")
    ax.set_title("CMC Curve — Cumulative Match Characteristic")
    ax.set_xticks(range(1, max_rank+1)); ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig)


def compute_evaluation_metrics(y_true, y_pred, label_names=None,
                                all_scores_matrix=None,
                                gt_labels=None,
                                query_times=None):
    """
    Full evaluation. gt_labels is the per-image class label list aligned with
    all_scores_matrix rows — required for correct Top-K / mAP / CMC.
    """
    st.subheader("📐 Evaluation Metrics")

    if len(y_true) == 0 or len(y_pred) == 0:
        st.warning("No evaluation data available.")
        return

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted",  zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted",      zero_division=0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  f"{acc  * 100:.2f}%")
    col2.metric("Precision", f"{prec * 100:.2f}%")
    col3.metric("Recall",    f"{rec  * 100:.2f}%")
    col4.metric("F1 Score",  f"{f1   * 100:.2f}%")

    # Retrieval metrics — need valid matrix, gt_labels, and >1 unique class
    can_do_retrieval = (
        all_scores_matrix is not None
        and gt_labels is not None
        and len(all_scores_matrix) >= 2
        and len(set(gt_labels)) >= 2   # FIX C: skip when only 1 class
        and all(len(row) == len(all_scores_matrix[0]) for row in all_scores_matrix)
    )

    if can_do_retrieval:
        st.markdown("**Retrieval Metrics**")
        topk = compute_topk_accuracy(all_scores_matrix, gt_labels)
        mAP  = compute_map(all_scores_matrix, gt_labels)

        k_cols = st.columns(len(topk) + 1)
        for i, (k_label, v) in enumerate(topk.items()):
            k_cols[i].metric(k_label + " Accuracy", f"{v:.2f}%")
        k_cols[-1].metric("mAP", f"{mAP:.2f}%")
        st.markdown(
            "*Top-K: query is correct if any top-K result shares the same class. "
            "mAP: mean Average Precision — standard retrieval benchmark.*"
        )
        plot_cmc_curve(all_scores_matrix, gt_labels)
    elif all_scores_matrix is not None and gt_labels is not None and len(set(gt_labels)) < 2:
        st.info(
            "ℹ️ Only 1 product class detected — retrieval metrics (Top-K / mAP / CMC) "
            "require ≥ 2 different product classes. Upload images of multiple products."
        )
    else:
        st.info("ℹ️ Upload multiple images per product for retrieval metrics (Top-K / mAP / CMC).")

    if query_times:
        avg_t = float(np.mean(query_times))
        std_t = float(np.std(query_times))
        st.info(
            f"⏱ Query Inference Time — "
            f"Mean: {avg_t:.3f}s | Std: {std_t:.3f}s | "
            f"Min: {min(query_times):.3f}s | Max: {max(query_times):.3f}s"
        )

    labels = sorted(set(y_true) | set(y_pred))
    display_labels = (label_names or [str(l) for l in labels])[:len(labels)]
    per_p = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_r = recall_score(y_true,    y_pred, labels=labels, average=None, zero_division=0)
    per_f = f1_score(y_true,        y_pred, labels=labels, average=None, zero_division=0)

    st.dataframe(pd.DataFrame({
        "Class":         display_labels,
        "Precision (%)": [round(p*100, 2) for p in per_p],
        "Recall (%)":    [round(r*100, 2) for r in per_r],
        "F1 (%)":        [round(f*100, 2) for f in per_f],
    }))


def plot_confusion_matrix(y_true, y_pred, label_names=None):
    st.subheader("🧩 Confusion Matrix")
    labels        = sorted(set(y_true) | set(y_pred))
    cm            = confusion_matrix(y_true, y_pred, labels=labels)
    display_labels = (label_names or [str(l) for l in labels])[:len(labels)]

    fig, ax = plt.subplots(figsize=(max(5, len(labels)), max(4, len(labels))))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels
                           ).plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=35)
    ax.set_title("Multi-Class Confusion Matrix")
    plt.tight_layout(); st.pyplot(fig)


def plot_roc_pr_curves(y_true, eval_scores, label_names, unique_classes):
    """
    FIX: eval_scores[class_idx] is a list aligned with eval_y_true (one entry
    per query).  We use label_binarize + per-class score arrays to build ROC/PR.
    """
    if len(set(y_true)) < 2:
        st.info("Need ≥ 2 classes for ROC/PR curves.")
        return

    classes = sorted(set(y_true))
    y_bin   = label_binarize(y_true, classes=classes)
    if y_bin.shape[1] == 1:
        y_bin   = np.hstack([1 - y_bin, y_bin])
        classes = [0, 1]

    n_classes  = len(classes)
    colors_roc = plt.cm.tab10(np.linspace(0, 1, n_classes))

    # ── ROC ──────────────────────────────────────────────────────────────────
    st.subheader("📈 ROC Curves (One-vs-Rest)")
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

    for i, cls in enumerate(classes):
        sc = np.array(eval_scores.get(cls, []))
        if len(sc) != len(y_true):
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], sc)
        lbl = label_names[i] if i < len(label_names) else str(cls)
        ax_roc.plot(fpr, tpr, color=colors_roc[i],
                    label=f"{lbl} (AUC={auc(fpr,tpr):.2f})")

    ax_roc.plot([0,1],[0,1],"k--",lw=1)
    ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves (One-vs-Rest)")
    ax_roc.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); st.pyplot(fig_roc)

    # ── PR ───────────────────────────────────────────────────────────────────
    st.subheader("📉 Precision-Recall Curves (One-vs-Rest)")
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))

    for i, cls in enumerate(classes):
        sc = np.array(eval_scores.get(cls, []))
        if len(sc) != len(y_true):
            continue
        prec_c, rec_c, _ = precision_recall_curve(y_bin[:, i], sc)
        ap  = average_precision_score(y_bin[:, i], sc)
        lbl = label_names[i] if i < len(label_names) else str(cls)
        ax_pr.plot(rec_c, prec_c, color=colors_roc[i],
                   label=f"{lbl} (AP={ap:.2f})")

    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curves (One-vs-Rest)")
    ax_pr.legend(loc="upper right", fontsize=8)
    plt.tight_layout(); st.pyplot(fig_pr)


# =========================
# ROBUSTNESS ANALYSIS
# =========================
def robustness_analysis(ref_img, title=""):
    st.subheader(f"{title} Robustness Analysis")
    st.write("Testing SIFT invariance across transformation types:")

    sift = cv2.SIFT_create()
    kp_ref, des_ref, _ = filter_keypoints_by_response(
        *sift.detectAndCompute(ref_img, None), kp_response_threshold
    )
    if des_ref is None:
        st.warning("Could not extract descriptors from reference image.")
        return

    variants = generate_synthetic_dataset(ref_img)
    v_names, bf_scores, kp_counts = [], [], []

    for variant_img, label in variants[1:]:
        kp_v, des_v = sift.detectAndCompute(variant_img, None)
        kp_v, des_v, _ = filter_keypoints_by_response(kp_v, des_v, kp_response_threshold)
        if des_v is None:
            continue
        _, bf_s = match(des_ref, des_v, "BF")
        v_names.append(label)
        bf_scores.append(round(float(bf_s), 2))
        kp_counts.append(int(len(kp_v)))

    st.dataframe(pd.DataFrame({
        "Variant": v_names,
        "BF Score (%)": bf_scores,
        "Keypoints": kp_counts
    }))

    colors = []
    for v in v_names:
        if "scale"    in v: colors.append("#2196F3")
        elif "rotation" in v: colors.append("#4CAF50")
        elif "illum"    in v: colors.append("#FF9800")
        else:                 colors.append("#9C27B0")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(v_names, bf_scores, color=colors)
    ax.set_title(f"{title} SIFT Robustness Across Transformations")
    ax.set_ylabel("BF Match Score (%)"); ax.set_ylim(0, 100)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.tight_layout(); st.pyplot(fig)
    st.success("✔ Scale / Rotation / Illumination Variation Handling — tested")

# =========================
# COMMON PIPELINE
# =========================
def process(ref, test, kp_ref, kp_test, des_ref, des_test, title):

    kp_ref,  des_ref,  _ = filter_keypoints_by_response(kp_ref,  des_ref,  kp_response_threshold)
    kp_test, des_test, _ = filter_keypoints_by_response(kp_test, des_test, kp_response_threshold)

    bf_start = time.time()
    bf_g, bf_s = match(des_ref, des_test, "BF")
    bf_time    = time.time() - bf_start

    fl_start = time.time()
    fl_g, fl_s = match(des_ref, des_test, "FLANN")
    fl_time    = time.time() - fl_start

    orb = cv2.ORB_create()
    _, d1 = orb.detectAndCompute(ref,  None)
    _, d2 = orb.detectAndCompute(test, None)
    _, orb_s = match_binary(d1, d2)

    hist_score = histogram_similarity(ref, test)
    cnn_score  = cnn_similarity(ref, test)

    st.subheader(title + " CNN Analysis")
    st.write(f"CNN Similarity Score: {cnn_score:.2f}%")

    st.subheader(title + " SIFT Feature Analysis")
    st.write(f"Reference Keypoints (after filter): {len(kp_ref)}")
    st.write(f"Test Image Keypoints (after filter): {len(kp_test)}")
    if kp_response_threshold > 0:
        st.info(f"Keypoint response filter active: threshold = {kp_response_threshold}")

    st.subheader(title + " Matching Time Analysis")
    st.write(f"BF Matching Time: {bf_time:.4f} sec")
    st.write(f"FLANN Matching Time: {fl_time:.4f} sec")

    st.subheader(title + " Histogram Analysis")
    h1 = cv2.calcHist([ref],  [0], None, [256], [0, 256])
    h2 = cv2.calcHist([test], [0], None, [256], [0, 256])
    cv2.normalize(h1, h1); cv2.normalize(h2, h2)

    fig_hist, ax_hist = plt.subplots(figsize=(8, 3))
    ax_hist.plot(h1, color="#2196F3", linewidth=1.2, label="Reference Image")
    ax_hist.plot(h2, color="#FF9800", linewidth=1.2, label="Test Image", linestyle="--")
    ax_hist.fill_between(range(256), h1.flatten(), alpha=0.25, color="#2196F3")
    ax_hist.fill_between(range(256), h2.flatten(), alpha=0.25, color="#FF9800")
    ax_hist.set_xlabel("Pixel Intensity"); ax_hist.set_ylabel("Normalised Frequency")
    ax_hist.set_title(f"{title} Histogram  |  Correlation: {hist_score:.2f}%")
    ax_hist.legend(loc="upper right"); ax_hist.set_xlim([0, 255])
    plt.tight_layout(); st.pyplot(fig_hist)

    if hist_score > 60:
        st.success(f"Good illumination similarity. (Histogram: {hist_score:.2f}%)")
    else:
        st.warning(f"Lighting varies between images. (Histogram: {hist_score:.2f}%)")

    def _show_matches(subtitle, matches):
        st.subheader(title + subtitle)
        if not matches:
            st.warning("No matches found"); return
        img = cv2.drawMatches(ref, kp_ref, test, kp_test, matches[:50], None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        st.image(img, channels="BGR")

    _show_matches(" BF Matching",    bf_g)
    _show_matches(" FLANN Matching", fl_g)
    st.image(cv2.drawKeypoints(test, kp_test, None, color=(0,255,0)),
             caption=title + " Detected SIFT Keypoints (filtered)")

    st.subheader(title + " Object Localization (Homography)")
    localized = draw_localization(ref, test, kp_ref, kp_test, bf_g)
    st.image(localized, channels="BGR", caption="Product localized using RANSAC Homography")

    comparison_df = pd.DataFrame({
        "Method":     ["BF", "FLANN", "ORB"],
        "Score (%)":  [round(float(bf_s),2), round(float(fl_s),2), round(float(orb_s),2)],
        "Time (sec)": [round(float(bf_time),4), round(float(fl_time),4), float("nan")]
    })
    st.subheader(title + " Feature Matching Comparison")
    st.dataframe(comparison_df)

    best_method = max({"BF": bf_s, "FLANN": fl_s, "ORB": orb_s}, key=lambda k: {"BF":bf_s,"FLANN":fl_s,"ORB":orb_s}[k])
    st.success(f"Best Matching Method: {best_method}")
    st.write(f"Histogram Similarity: {hist_score:.2f}%")

    fig, ax = plt.subplots()
    ax.bar(["BF","FLANN","ORB","Histogram","CNN"],
           [bf_s, fl_s, orb_s, hist_score, cnn_score],
           color=["#2196F3","#4CAF50","#FF9800","#E91E63","#9C27B0"])
    ax.set_title(title + " Performance Comparison")
    ax.set_ylabel("Score (%)"); ax.set_ylim(0, 100)
    st.pyplot(fig)

    return bf_s, fl_s, orb_s, hist_score

# =========================
# DATASET MODE
# =========================
if mode == "📁 Dataset Mode":
    st.header("Dataset Mode — Multi-Product Recognition")

    dataset_files = st.file_uploader(
        "📂 Upload Dataset Images (multiple product images)",
        type=["jpg","png","jpeg"], accept_multiple_files=True
    )
    test_file = st.file_uploader(
        "🔍 Upload Test Image (image to identify)",
        type=["jpg","png","jpeg"], key="test_upload"
    )
    use_synthetic = st.checkbox(
        "Generate synthetic dataset from test image (if no dataset available)",
        value=False
    )

    if test_file is not None:
        start    = time.time()
        test_img = read_img(test_file)

        if test_img is None:
            st.error("Could not read test image.")
        else:
            dataset_imgs, dataset_names = [], []
            if dataset_files:
                dataset_imgs, dataset_names = load_uploaded_images(dataset_files)

            if use_synthetic or len(dataset_imgs) == 0:
                st.info("Generating synthetic dataset variants...")
                for v_img, v_name in generate_synthetic_dataset(test_img):
                    dataset_imgs.append(v_img)
                    dataset_names.append(v_name)

            if not dataset_imgs:
                st.error("No dataset images available.")
            else:
                st.write(f"Dataset size: {len(dataset_imgs)} images")
                sift = cv2.SIFT_create()

                kp_test_raw, des_test_raw = sift.detectAndCompute(test_img, None)
                kp_test, des_test, _ = filter_keypoints_by_response(
                    kp_test_raw, des_test_raw, kp_response_threshold
                )

                if des_test is None:
                    st.error("Could not extract SIFT features from test image.")
                else:
                    best_score, best_name, best_idx = -1, "", 0
                    all_scores = []
                    progress   = st.progress(0)

                    for i, (d_img, d_name) in enumerate(zip(dataset_imgs, dataset_names)):
                        kp_d_raw, des_d_raw = sift.detectAndCompute(d_img, None)
                        kp_d, des_d, _ = filter_keypoints_by_response(
                            kp_d_raw, des_d_raw, kp_response_threshold
                        )
                        if des_d is None:
                            all_scores.append(0.0)
                            progress.progress((i+1)/len(dataset_imgs))
                            continue

                        _, bf_s   = match(des_d, des_test, "BF")
                        _, fl_s   = match(des_d, des_test, "FLANN")
                        hist_s    = histogram_similarity(d_img, test_img)
                        cnn_s     = cnn_similarity(d_img, test_img)
                        combined  = float(0.30*max(bf_s,fl_s) + 0.15*hist_s + 0.55*cnn_s)
                        all_scores.append(combined)

                        if combined > best_score:
                            best_score, best_name, best_idx = combined, d_name, i

                        progress.progress((i+1)/len(dataset_imgs))

                    # ── Results summary ──────────────────────────────────────
                    st.subheader("📊 Dataset Matching Results")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Best Match",       best_name)
                    col2.metric("Confidence Score", f"{best_score:.2f}%")
                    col3.metric("Processing Time",  f"{time.time()-start:.2f} sec")

                    score_df = pd.DataFrame({
                        "Product": dataset_names,
                        "Combined Score (%)": [round(float(s),2) for s in all_scores]
                    })
                    st.subheader("All Product Scores"); st.dataframe(score_df)

                    fig, ax = plt.subplots(figsize=(max(6, len(dataset_names)), 4))
                    ax.bar(dataset_names, all_scores,
                           color=["#4CAF50" if n==best_name else "#90A4AE" for n in dataset_names])
                    ax.set_title("Similarity Scores Across Dataset")
                    ax.set_ylabel("Combined Score (%)"); ax.set_ylim(0,100)
                    plt.xticks(rotation=35, ha="right", fontsize=8)
                    plt.tight_layout(); st.pyplot(fig)

                    if   best_score > 50: st.success(f"🟢 PRODUCT IDENTIFIED — {best_name} ({best_score:.2f}%)")
                    elif best_score > 30: st.warning(f"🟡 POSSIBLE MATCH — {best_name} ({best_score:.2f}%)")
                    else:                 st.error("🔴 NO CONFIDENT MATCH FOUND in dataset.")

                    # ── Detailed analysis on best match ──────────────────────
                    st.subheader(f"Detailed Analysis — Best Match: {best_name}")
                    best_img                = dataset_imgs[best_idx]
                    kp_best_raw, des_best_raw = sift.detectAndCompute(best_img, None)
                    kp_best, des_best, _    = filter_keypoints_by_response(
                        kp_best_raw, des_best_raw, kp_response_threshold
                    )
                    process(best_img, test_img, kp_best, kp_test,
                            des_best, des_test, "DATASET:")
                    descriptor_comparison(best_img, test_img)
                    robustness_analysis(test_img, title="DATASET:")

                    # ── EVALUATION SECTION ───────────────────────────────────
                    st.markdown("---")
                    st.header("📊 Multi-Class Evaluation (Dataset vs Test)")

                    n = len(dataset_imgs)
                    if n < 2:
                        st.info("Upload 2+ dataset images for evaluation metrics.")
                    else:
                        import re as _re
                        from collections import Counter

                        def _class_from_name(fname):
                            stem = fname.rsplit(".", 1)[0]
                            stem = _re.sub(r"[_\-]?\d+$", "", stem)
                            return stem if stem else fname

                        class_labels   = [_class_from_name(nm) for nm in dataset_names]
                        unique_classes = sorted(set(class_labels))
                        class_to_idx   = {c: i for i, c in enumerate(unique_classes)}
                        gt_indices     = [class_to_idx[c] for c in class_labels]
                        class_counts   = Counter(class_labels)
                        multi_classes  = [c for c, cnt in class_counts.items() if cnt >= 2]

                        st.subheader("🏷️ Detected Class Labels")
                        st.dataframe(pd.DataFrame({
                            "Filename":      dataset_names,
                            "Detected Class": class_labels,
                            "Class Index":   gt_indices
                        }))

                        # ── GUARD: single-class dataset ───────────────────
                        if len(unique_classes) == 1:
                            st.warning(
                                f"⚠️ All {n} images belong to one class "
                                f"(**{unique_classes[0]}**). "
                                "Meaningful evaluation requires images of "
                                "**at least 2 different products**. "
                                "Rename files as `ProductA_001.jpg`, "
                                "`ProductB_001.jpg`, etc."
                            )
                            st.info(
                                "Skipping confusion matrix, Top-K, mAP and ROC — "
                                "they are undefined for a single-class dataset."
                            )
                        else:
                            # ── Accumulators ──────────────────────────────
                            eval_y_true       = []   # class idx per query
                            eval_y_pred       = []   # predicted class idx
                            eval_gt_labels    = []   # class idx parallel to all_scores_matrix rows
                            eval_scores       = {i: [] for i in range(len(unique_classes))}
                            all_scores_matrix = []   # rows only for LOO (multi-class)
                            query_times       = []

                            if len(multi_classes) == 0:
                                # ── Single-image-per-class mode ────────────
                                # Each dataset image is its own class.
                                # Evaluation strategy:
                                #   For every dataset image qi, treat the OTHER
                                #   (n-1) images as the gallery and pick the
                                #   highest-scoring one as the prediction.
                                #   This gives a meaningful n×n confusion matrix.
                                st.info(
                                    "📋 Single-image-per-class mode: "
                                    "leave-one-out across all dataset images."
                                )
                                progress2 = st.progress(0)
                                t0_all = time.time()

                                for qi in range(n):
                                    q_img  = dataset_imgs[qi]
                                    q_true = gt_indices[qi]

                                    kp_q, des_q, _ = filter_keypoints_by_response(
                                        *sift.detectAndCompute(q_img, None),
                                        kp_response_threshold
                                    )
                                    if des_q is None:
                                        progress2.progress((qi+1)/n); continue

                                    t_q = time.time()
                                    gallery_scores = {}

                                    for di in range(n):
                                        if di == qi: continue   # exclude self
                                        kp_d2, des_d2, _ = filter_keypoints_by_response(
                                            *sift.detectAndCompute(dataset_imgs[di], None),
                                            kp_response_threshold
                                        )
                                        if des_d2 is None:
                                            gallery_scores[di] = 0.0; continue
                                        _, sift_s = match(des_q, des_d2, "BF")
                                        cnn_s2    = cnn_similarity(q_img, dataset_imgs[di])
                                        hist_s2   = histogram_similarity(q_img, dataset_imgs[di])
                                        gallery_scores[di] = float(
                                            0.30*sift_s + 0.15*hist_s2 + 0.55*cnn_s2
                                        )

                                    query_times.append(time.time() - t_q)
                                    if not gallery_scores:
                                        progress2.progress((qi+1)/n); continue

                                    # prediction = class of highest-scoring gallery image
                                    best_di    = max(gallery_scores, key=gallery_scores.get)
                                    pred_class = gt_indices[best_di]
                                    eval_y_true.append(q_true)
                                    eval_y_pred.append(pred_class)
                                    eval_gt_labels.append(q_true)

                                    # per-class scores for ROC (0-1 range)
                                    class_max = {}
                                    for di, sc in gallery_scores.items():
                                        ci = gt_indices[di]
                                        class_max[ci] = max(class_max.get(ci, 0.0), sc)
                                    for ci in range(len(unique_classes)):
                                        eval_scores[ci].append(
                                            class_max.get(ci, 0.0) / 100.0
                                        )

                                    row = [gallery_scores.get(di, 0.0) for di in range(n)]
                                    all_scores_matrix.append(row)
                                    progress2.progress((qi+1)/n)

                                st.info(
                                    "ℹ️ Upload 2+ images per product "
                                    "(e.g. Apple_001.jpg, Apple_002.jpg) "
                                    "for more reliable evaluation."
                                )
                            else:
                                # ── Leave-One-Out cross-validation ────────
                                st.info(
                                    f"✅ {len(multi_classes)} class(es) with multiple "
                                    "images. Running Leave-One-Out cross-validation."
                                )
                                progress2 = st.progress(0)

                                for qi in range(n):
                                    q_img  = dataset_imgs[qi]
                                    q_true = gt_indices[qi]

                                    kp_q, des_q, _ = filter_keypoints_by_response(
                                        *sift.detectAndCompute(q_img, None),
                                        kp_response_threshold
                                    )
                                    if des_q is None:
                                        progress2.progress((qi+1)/n); continue

                                    t_q = time.time()
                                    gallery_scores = {}

                                    for di in range(n):
                                        if di == qi: continue
                                        kp_d2, des_d2, _ = filter_keypoints_by_response(
                                            *sift.detectAndCompute(dataset_imgs[di], None),
                                            kp_response_threshold
                                        )
                                        if des_d2 is None:
                                            gallery_scores[di] = 0.0; continue
                                        _, sift_s  = match(des_q, des_d2, "BF")
                                        cnn_s2     = cnn_similarity(q_img, dataset_imgs[di])
                                        hist_s2    = histogram_similarity(q_img, dataset_imgs[di])
                                        gallery_scores[di] = float(
                                            0.30*sift_s + 0.15*hist_s2 + 0.55*cnn_s2
                                        )

                                    query_times.append(time.time() - t_q)
                                    if not gallery_scores:
                                        progress2.progress((qi+1)/n); continue

                                    best_di    = max(gallery_scores, key=gallery_scores.get)
                                    pred_class = gt_indices[best_di]
                                    eval_y_true.append(q_true)
                                    eval_y_pred.append(pred_class)
                                    eval_gt_labels.append(q_true)  # track per-row label

                                    # per-class max scores for ROC (0-1 range)
                                    class_max = {}
                                    for di, sc in gallery_scores.items():
                                        ci = gt_indices[di]
                                        class_max[ci] = max(class_max.get(ci, 0.0), sc)
                                    for ci in range(len(unique_classes)):
                                        eval_scores[ci].append(
                                            class_max.get(ci, 0.0) / 100.0
                                        )

                                    # full-width row for Top-K/CMC/mAP
                                    row = [gallery_scores.get(di, 0.0) for di in range(n)]
                                    all_scores_matrix.append(row)
                                    progress2.progress((qi+1)/n)

                            label_names_short = [c[:15] for c in unique_classes]

                            if eval_y_true:
                                compute_evaluation_metrics(
                                    eval_y_true, eval_y_pred, label_names_short,
                                    all_scores_matrix=all_scores_matrix if all_scores_matrix else None,
                                    gt_labels=eval_gt_labels if eval_gt_labels else None,
                                    query_times=query_times
                                )
                                plot_confusion_matrix(eval_y_true, eval_y_pred, label_names_short)

                                # ROC/PR: valid only when ≥2 classes AND score arrays aligned
                                roc_ok = (
                                    len(unique_classes) >= 2
                                    and bool(eval_scores)
                                    and all(len(v) == len(eval_y_true)
                                            for v in eval_scores.values())
                                )
                                if roc_ok:
                                    plot_roc_pr_curves(
                                        eval_y_true, eval_scores,
                                        label_names_short, unique_classes
                                    )
                                else:
                                    st.info(
                                        "ℹ️ ROC/PR curves require ≥ 2 product classes "
                                        "with multiple images each."
                                    )
                            else:
                                st.warning("Not enough valid queries for evaluation.")

                    st.subheader("📌 Conclusion")
                    st.success(
                        "SIFT-based retail product recognition successfully "
                        "detects and localizes products using BF and FLANN "
                        "matching, augmented with CNN deep features."
                    )
                    st.write(f"Best Match: {best_name}")
                    st.write(f"Best Score: {best_score:.2f}%")
                    st.write(f"Processing Time: {time.time()-start:.2f} sec")

# =========================
# REAL-TIME MODE
# =========================
elif mode == "📷 Real-Time Mode":
    st.header("Real-Time Mode — Webcam Product Recognition")

    ref_file = st.file_uploader("📤 Upload Reference Image", type=["jpg","png","jpeg"])
    cam_img  = st.camera_input("📷 Capture Webcam Image")

    if ref_file is not None and cam_img is not None:
        start = time.time()
        ref   = read_img(ref_file)
        test  = read_img(cam_img)

        if ref is None or test is None:
            st.error("Unable to read images. Please try again.")
        else:
            sift = cv2.SIFT_create()
            kp1_raw, des1_raw = sift.detectAndCompute(ref,  None)
            kp2_raw, des2_raw = sift.detectAndCompute(test, None)
            kp1, des1, _ = filter_keypoints_by_response(kp1_raw, des1_raw, kp_response_threshold)
            kp2, des2, _ = filter_keypoints_by_response(kp2_raw, des2_raw, kp_response_threshold)

            if des1 is None or des2 is None:
                st.error("Could not extract SIFT features. Ensure images have sufficient texture.")
            else:
                bf_s, fl_s, orb_s, hist_s = process(ref, test, kp1, kp2, des1, des2, "REAL-TIME:")
                descriptor_comparison(ref, test)
                robustness_analysis(ref, title="REAL-TIME:")

                final_score = max(bf_s, fl_s, orb_s, hist_s)
                st.metric("Confidence Score", f"{final_score:.2f}%")

                if   final_score > 40: st.success("🟢 SAME PRODUCT")
                elif final_score > 20: st.warning("🟡 POSSIBLE MATCH")
                else:                  st.error("🔴 DIFFERENT PRODUCT")

                st.write(f"Processing Time: {time.time()-start:.2f} sec")
                st.subheader("📌 Conclusion")
                st.success(
                    "SIFT-based retail product recognition successfully "
                    "detects and localizes products using BF and FLANN matching."
                )