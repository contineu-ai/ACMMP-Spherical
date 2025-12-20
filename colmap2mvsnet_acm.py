#!/usr/bin/env python
"""
COLMAP → ACMMP/MVSNet converter optimized for SPHERICAL cameras
HIGHLY OPTIMIZED VERSION - with symlink/hardlink support
"""

from __future__ import print_function
import os, shutil, struct, argparse, collections, multiprocessing as mp
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2
from tqdm import tqdm
from scipy.spatial import cKDTree
from collections import defaultdict

# ───────────────────────────────────────────────────────────────────────────────
# 1.  Named-tuple data structures
# ───────────────────────────────────────────────────────────────────────────────
CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImg = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

class Image(BaseImg):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

CAMERA_MODELS = {
    CameraModel(0, "SIMPLE_PINHOLE", 3), CameraModel(1, "PINHOLE", 4),
    CameraModel(2, "SIMPLE_RADIAL", 4), CameraModel(3, "RADIAL", 5),
    CameraModel(4, "OPENCV", 8), CameraModel(5, "OPENCV_FISHEYE", 8),
    CameraModel(6, "FULL_OPENCV", 12), CameraModel(7, "FOV", 5),
    CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4), CameraModel(9, "RADIAL_FISHEYE", 5),
    CameraModel(10, "THIN_PRISM_FISHEYE", 12),
    CameraModel(model_id=11, model_name="SPHERE", num_params=3),
    CameraModel(model_id=12, model_name="SPHERICAL", num_params=3),
}
CAMERA_MODEL_IDS = {cm.model_id: cm for cm in CAMERA_MODELS}

# ───────────────────────────────────────────────────────────────────────────────
# 2.  Low-level binary helpers
# ───────────────────────────────────────────────────────────────────────────────
def _read(fid, nbytes, fmt, endian="<"):
    return struct.unpack(endian + fmt, fid.read(nbytes))

# ───────────────────────────────────────────────────────────────────────────────
# 3.  COLMAP text / binary readers
# ───────────────────────────────────────────────────────────────────────────────
def read_cameras_text(path):
    cams = {}
    with open(path) as f:
        for ln in f:
            if ln.lstrip().startswith("#") or not ln.strip(): continue
            s = ln.split(); cid = int(s[0]); model = s[1]; w, h = map(int, s[2:4])
            params = np.fromiter(map(float, s[4:]), float)
            cams[cid] = Camera(cid, model, w, h, params)
    return cams

def read_cameras_binary(path):
    cams = {}
    with open(path, "rb") as f:
        n = _read(f, 8, "Q")[0]
        for _ in range(n):
            cid, mid, w, h = _read(f, 24, "iiQQ")
            cm = CAMERA_MODEL_IDS[mid]
            params = np.array(_read(f, 8 * cm.num_params, "d" * cm.num_params))
            cams[cid] = Camera(cid, cm.model_name, w, h, params)
    return cams

def read_images_text(path):
    imgs = {}
    with open(path) as f:
        while True:
            ln = f.readline()
            if not ln: break
            if ln.lstrip().startswith("#") or not ln.strip(): continue
            s = ln.split()
            iid = int(s[0]); qvec = np.fromiter(map(float, s[1:5]), float)
            tvec = np.fromiter(map(float, s[5:8]), float); cid = int(s[8]); name = s[9]
            track = f.readline().split()
            xys = np.column_stack([list(map(float, track[0::3])), list(map(float, track[1::3]))])
            pids = np.array(list(map(int, track[2::3])), dtype=int)
            imgs[iid] = Image(iid, qvec, tvec, cid, name, xys, pids)
    return imgs

def read_images_binary(path):
    imgs = {}
    with open(path, "rb") as f:
        n = _read(f, 8, "Q")[0]
        for _ in range(n):
            iid, *vals = _read(f, 64, "idddddddi")
            qvec = np.array(vals[0:4]); tvec = np.array(vals[4:7]); cid = vals[7]
            name = b""
            while True:
                c = _read(f, 1, "c")[0]
                if c == b"\x00": break
                name += c
            name = name.decode()
            npts = _read(f, 8, "Q")[0]
            data = _read(f, 24 * npts, "ddq" * npts)
            xys = np.column_stack([data[0::3], data[1::3]])
            pids = np.array(data[2::3], dtype=int)
            imgs[iid] = Image(iid, qvec, tvec, cid, name, xys, pids)
    return imgs

def read_points3D_text(path):
    pts = {}
    with open(path) as f:
        for ln in f:
            if ln.lstrip().startswith("#") or not ln.strip(): continue
            s = ln.split(); pid = int(s[0])
            xyz = np.fromiter(map(float, s[1:4]), float)
            rgb = np.fromiter(map(int, s[4:7]), int); err = float(s[7])
            img_ids = np.array(list(map(int, s[8::2])), dtype=int)
            idxs = np.array(list(map(int, s[9::2])), dtype=int)
            pts[pid] = Point3D(pid, xyz, rgb, err, img_ids, idxs)
    return pts

def read_points3D_binary(path):
    pts = {}
    with open(path, "rb") as f:
        n = _read(f, 8, "Q")[0]
        for _ in range(n):
            pid, x, y, z, r, g, b, err = _read(f, 43, "QdddBBBd")
            length = _read(f, 8, "Q")[0]
            track = _read(f, 8 * length, "ii" * length)
            img_ids = np.array(track[0::2], dtype=int)
            idxs = np.array(track[1::2], dtype=int)
            pts[pid] = Point3D(pid, np.array([x, y, z]), np.array([r, g, b]), err, img_ids, idxs)
    return pts

def read_model(sparse_dir, ext):
    if ext == ".txt":
        cams = read_cameras_text(os.path.join(sparse_dir, "cameras" + ext))
        imgs = read_images_text(os.path.join(sparse_dir, "images" + ext))
        pts = read_points3D_text(os.path.join(sparse_dir, "points3D" + ext))
    else:
        cams = read_cameras_binary(os.path.join(sparse_dir, "cameras" + ext))
        imgs = read_images_binary(os.path.join(sparse_dir, "images" + ext))
        pts = read_points3D_binary(os.path.join(sparse_dir, "points3D" + ext))
    return cams, imgs, pts

# ───────────────────────────────────────────────────────────────────────────────
# 4.  Math helpers
# ───────────────────────────────────────────────────────────────────────────────
def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])

# ───────────────────────────────────────────────────────────────────────────────
# 5.  Depth-range computation
# ───────────────────────────────────────────────────────────────────────────────
def compute_depth_ranges(images, points3d, extrinsic, max_d, interval_scale, cams):
    depth_ranges = {}
    skipped_images = []
    
    for i, img in images.items():
        zs = []
        for pid in img.point3D_ids:
            if pid < 0 or pid not in points3d: continue
            X = np.append(points3d[pid].xyz, 1.0)
            X_cam = extrinsic[i] @ X
            d = np.linalg.norm(X_cam[:3])
            if d <= 0: continue
            zs.append(d)
        
        if len(zs) < 10:
            skipped_images.append((i, len(zs)))
            continue
            
        zs_sorted = sorted(zs)
        dmin = zs_sorted[int(len(zs_sorted) * 0.2)] * 0.75
        dmax = zs_sorted[int(len(zs_sorted) * 0.8)] * 1.25
        
        if max_d == 0:
            log_range = np.log(dmax / dmin)
            if log_range < 1.0: depth_num = 64
            elif log_range < 2.0: depth_num = 128
            elif log_range < 3.0: depth_num = 192
            else: depth_num = 256
            depth_num = min(256, max(32, depth_num))
        else:
            depth_num = max_d
            
        dint = (dmax - dmin) / (depth_num - 1) / interval_scale
        depth_ranges[i] = (dmin, dint, depth_num, dmax)
    
    if skipped_images:
        print(f"[WARN] Skipped {len(skipped_images)} images with insufficient points")
    
    return depth_ranges

# ───────────────────────────────────────────────────────────────────────────────
# 6.  Pair scoring (optimized)
# ───────────────────────────────────────────────────────────────────────────────
def calc_baseline_to_depth_ratio_vectorized(ci, cj, shared_xyz):
    baseline = np.linalg.norm(ci - cj)
    if len(shared_xyz) == 0: return 0.0
    depths = np.linalg.norm(shared_xyz - ci, axis=1)
    median_depth = np.median(depths)
    if median_depth < 1e-6: return 0.0
    ratio = baseline / median_depth
    if 0.05 < ratio < 0.2: return 1.0
    elif 0.03 < ratio < 0.3: return 0.7
    elif 0.01 < ratio < 0.5: return 0.4
    else: return 0.1

def calc_score_enhanced_fast(pair, images, points3d_xyz, theta0, cam_centers, min_shared=5):
    i, j = pair
    shared_pids = set(images[i].point3D_ids) & set(images[j].point3D_ids)
    shared_pids = [pid for pid in shared_pids if pid != -1 and pid in points3d_xyz]
    
    if len(shared_pids) < min_shared: return i, j, 0.0
    
    ci, cj = cam_centers[i], cam_centers[j]
    baseline = np.linalg.norm(ci - cj)
    if baseline < 0.01: return i, j, 0.0
    
    shared_xyz = np.array([points3d_xyz[pid] for pid in shared_pids])
    base_score = float(len(shared_pids))
    btd_score = calc_baseline_to_depth_ratio_vectorized(ci, cj, shared_xyz)
    
    if btd_score < 0.1: return i, j, base_score * btd_score
    
    vi, vj = shared_xyz - ci, shared_xyz - cj
    norms_i, norms_j = np.linalg.norm(vi, axis=1), np.linalg.norm(vj, axis=1)
    valid_mask = (norms_i > 1e-6) & (norms_j > 1e-6)
    if not np.any(valid_mask): return i, j, 0.0
    
    dots = np.sum(vi[valid_mask] * vj[valid_mask], axis=1)
    cos_angles = np.clip(dots / (norms_i[valid_mask] * norms_j[valid_mask]), -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_angles))
    
    angle_75 = np.percentile(angles, 75)
    if angle_75 < theta0: angle_score = 0.1
    elif angle_75 < 5.0: angle_score = 0.5
    else: angle_score = 1.0
    
    return i, j, base_score * btd_score * angle_score

# ───────────────────────────────────────────────────────────────────────────────
# 7.  Neighbor selection
# ───────────────────────────────────────────────────────────────────────────────
def select_diverse_multiscale_neighbors_fast(ref_idx, candidates, cam_centers, top_k=20, diversity_threshold=0.3):
    if not candidates: return []
    
    ci = cam_centers[ref_idx]
    candidate_indices = np.array([idx for idx, _ in candidates])
    candidate_scores = np.array([score for _, score in candidates])
    candidate_positions = np.array([cam_centers[idx] for idx, _ in candidates])
    baselines = np.linalg.norm(candidate_positions - ci, axis=1)
    
    sort_idx = np.argsort(-candidate_scores)
    candidate_indices = candidate_indices[sort_idx]
    candidate_scores = candidate_scores[sort_idx]
    baselines = baselines[sort_idx]
    candidate_positions = candidate_positions[sort_idx]
    
    if len(candidates) <= top_k:
        return [(int(idx), float(score)) for idx, score in zip(candidate_indices, candidate_scores)]
    
    min_baseline, max_baseline = np.min(baselines), np.max(baselines)
    if max_baseline / (min_baseline + 1e-6) > 10:
        log_min, log_max = np.log(min_baseline + 1e-6), np.log(max_baseline + 1e-6)
        target_baselines = np.exp(np.linspace(log_min, log_max, top_k))
    else:
        target_baselines = np.linspace(min_baseline, max_baseline, top_k)
    
    selected, selected_positions = [], []
    used_mask = np.zeros(len(candidates), dtype=bool)
    
    for target_baseline in target_baselines:
        if len(selected) >= top_k: break
        best_idx, best_metric = -1, -1
        
        for local_idx in range(len(candidates)):
            if used_mask[local_idx]: continue
            if len(selected) >= 5:
                pos = candidate_positions[local_idx]
                if selected_positions:
                    dists = np.linalg.norm(np.array(selected_positions) - pos, axis=1)
                    if np.min(dists) < diversity_threshold * baselines[local_idx]: continue
            
            baseline_diff = abs(baselines[local_idx] - target_baseline) / (target_baseline + 1e-6)
            combined_metric = 0.7 * candidate_scores[local_idx] - 0.3 * baseline_diff
            if combined_metric > best_metric:
                best_metric, best_idx = combined_metric, local_idx
        
        if best_idx >= 0:
            selected.append((int(candidate_indices[best_idx]), float(candidate_scores[best_idx])))
            selected_positions.append(candidate_positions[best_idx])
            used_mask[best_idx] = True
    
    if len(selected) < top_k:
        for local_idx in range(len(candidates)):
            if used_mask[local_idx]: continue
            selected.append((int(candidate_indices[local_idx]), float(candidate_scores[local_idx])))
            if len(selected) >= top_k: break
    
    return selected

# ───────────────────────────────────────────────────────────────────────────────
# 8.  OPTIMIZED IMAGE COPYING/LINKING
# ───────────────────────────────────────────────────────────────────────────────
COPY_BUFFER_SIZE = 16 * 1024 * 1024  # 16MB buffer

def fast_copy_file(src, dst):
    """Ultra-fast file copy using large buffer and low-level I/O"""
    try:
        if hasattr(os, 'sendfile'):
            with open(src, 'rb') as fsrc:
                with open(dst, 'wb') as fdst:
                    fsrc.seek(0, 2)
                    size = fsrc.tell()
                    fsrc.seek(0)
                    fdst_fileno = fdst.fileno()
                    fsrc_fileno = fsrc.fileno()
                    offset = 0
                    while offset < size:
                        sent = os.sendfile(fdst_fileno, fsrc_fileno, offset, min(size - offset, COPY_BUFFER_SIZE))
                        if sent == 0: break
                        offset += sent
            return True
    except (OSError, AttributeError):
        pass
    
    try:
        with open(src, 'rb') as fsrc:
            with open(dst, 'wb') as fdst:
                while True:
                    buf = fsrc.read(COPY_BUFFER_SIZE)
                    if not buf: break
                    fdst.write(buf)
        return True
    except Exception:
        return False

def create_link(src, dst, link_type='symlink', relative=False):
    """
    Create a link (symlink, hardlink, or reflink) from dst -> src
    
    Args:
        src: Source file path
        dst: Destination link path  
        link_type: 'symlink', 'hardlink', or 'reflink'
        relative: If True, create relative symlink (only for symlink type)
    
    Returns:
        (success: bool, error: str or None)
    """
    try:
        # Remove existing file/link if present
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        
        if link_type == 'symlink':
            if relative:
                # Calculate relative path from dst directory to src
                dst_dir = os.path.dirname(os.path.abspath(dst))
                src_abs = os.path.abspath(src)
                rel_path = os.path.relpath(src_abs, dst_dir)
                os.symlink(rel_path, dst)
            else:
                # Absolute symlink
                os.symlink(os.path.abspath(src), dst)
            return True, None
            
        elif link_type == 'hardlink':
            os.link(src, dst)
            return True, None
            
        elif link_type == 'reflink':
            # Try copy-on-write reflink (Linux with supported filesystem)
            try:
                import subprocess
                result = subprocess.run(
                    ['cp', '--reflink=always', src, dst],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    return True, None
                else:
                    return False, f"reflink failed: {result.stderr}"
            except Exception as e:
                return False, f"reflink not supported: {e}"
        else:
            return False, f"Unknown link type: {link_type}"
            
    except OSError as e:
        return False, str(e)

def process_single_image(task):
    """
    Worker function for single image processing.
    Handles: copy, convert, symlink, hardlink, reflink
    Note: File existence already verified during task building.
    """
    idx, src, dst, needs_convert, mode, relative_links = task
    
    try:
        if needs_convert:
            # Must convert format - can't link different format
            img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
            if img is None:
                return idx, False, f"Failed to read: {src}", 'error'
            cv2.imwrite(dst, img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            return idx, True, None, 'converted'
        
        elif mode == 'copy':
            fast_copy_file(src, dst)
            return idx, True, None, 'copied'
        
        elif mode in ('symlink', 'hardlink', 'reflink'):
            success, error = create_link(src, dst, mode, relative=relative_links)
            if success:
                return idx, True, None, f'{mode}ed'
            else:
                # Fallback to copy on link failure
                fast_copy_file(src, dst)
                return idx, True, f"Link failed ({error}), copied instead", 'copied'
        
        else:
            return idx, False, f"Unknown mode: {mode}", 'error'
            
    except Exception as e:
        return idx, False, str(e), 'error'

def process_images_optimized(imgs, depth_ranges, imgs_dir, out_img, 
                             mode='copy', relative_links=False, max_workers=None,
                             skip_missing=False):
    """
    Process images with configurable mode:
    - 'copy': Fast file copy (default)
    - 'symlink': Create symbolic links (fastest, but links break if source moves)
    - 'hardlink': Create hard links (fast, same filesystem only)
    - 'reflink': Copy-on-write reflink (fast, requires btrfs/xfs/APFS)
    
    Non-PNG images are always converted (can't link different format).
    """
    if max_workers is None:
        max_workers = min(64, mp.cpu_count() * 4)
    
    # ===== FAST VALIDATION: Check images directory exists =====
    if not os.path.isdir(imgs_dir):
        print(f"[ERROR] Images directory does not exist: {imgs_dir}")
        if skip_missing:
            print("[WARN] --skip_missing set, continuing without images")
            return [], []
        raise FileNotFoundError(f"Images directory not found: {imgs_dir}")
    
    # ===== FAST SCAN: Get all existing files in one syscall =====
    print("[INFO] Scanning source images directory...")
    existing_files = set()
    try:
        # os.scandir is MUCH faster than individual os.path.exists() calls
        with os.scandir(imgs_dir) as entries:
            for entry in entries:
                if entry.is_file():
                    existing_files.add(entry.name)
    except OSError as e:
        print(f"[ERROR] Cannot read images directory: {e}")
        if skip_missing:
            return [], []
        raise
    
    print(f"[INFO] Found {len(existing_files)} files in source directory")
    
    if not existing_files:
        print("[WARN] Images directory is empty!")
        if skip_missing:
            return [], []
        raise FileNotFoundError(f"No images found in: {imgs_dir}")
    
    # ===== BUILD TASK LIST: Only for files that actually exist =====
    tasks = []
    missing = []
    for i in sorted(depth_ranges.keys()):
        img = imgs[i]
        if img.name not in existing_files:
            missing.append((i, img.name))
            continue
        src = os.path.join(imgs_dir, img.name)
        dst = os.path.join(out_img, f"{i:08d}.png")
        needs_convert = not src.lower().endswith('.png')
        tasks.append((i, src, dst, needs_convert, mode, relative_links))
    
    if missing:
        print(f"[WARN] {len(missing)} images not found in source directory:")
        for idx, name in missing[:5]:
            print(f"       Image {idx}: {name}")
        if len(missing) > 5:
            print(f"       ... and {len(missing) - 5} more")
    
    if not tasks:
        print("[WARN] No valid images to process (all missing or filtered)")
        return [], [(i, name) for i, name in missing]
    
    # Statistics
    n_convert = sum(1 for t in tasks if t[3])
    n_linkable = len(tasks) - n_convert
    
    mode_desc = {
        'copy': 'copy',
        'symlink': 'symlink',
        'hardlink': 'hardlink', 
        'reflink': 'reflink (CoW)'
    }.get(mode, mode)
    
    print(f"[INFO] Processing {len(tasks)} images:")
    print(f"       {n_linkable} PNG files → {mode_desc}")
    print(f"       {n_convert} non-PNG files → convert to PNG")
    
    # Process with thread pool
    results = {'copied': 0, 'converted': 0, 'symlinked': 0, 
               'hardlinked': 0, 'reflinked': 0, 'failed': 0}
    successes = []
    failures = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_image, task): task[0] for task in tasks}
        
        with tqdm(total=len(tasks), desc="Processing images", unit="img") as pbar:
            for future in as_completed(futures):
                idx, success, error, action = future.result()
                if success:
                    successes.append(idx)
                    if action == 'copied': results['copied'] += 1
                    elif action == 'converted': results['converted'] += 1
                    elif action == 'symlinked': results['symlinked'] += 1
                    elif action == 'hardlinked': results['hardlinked'] += 1
                    elif action == 'reflinked': results['reflinked'] += 1
                else:
                    failures.append((idx, error))
                    results['failed'] += 1
                pbar.update(1)
    
    # Print summary
    print(f"[INFO] Results: ", end="")
    parts = []
    if results['symlinked']: parts.append(f"{results['symlinked']} symlinked")
    if results['hardlinked']: parts.append(f"{results['hardlinked']} hardlinked")
    if results['reflinked']: parts.append(f"{results['reflinked']} reflinked")
    if results['copied']: parts.append(f"{results['copied']} copied")
    if results['converted']: parts.append(f"{results['converted']} converted")
    if results['failed']: parts.append(f"{results['failed']} failed")
    print(", ".join(parts) if parts else "none")
    
    return successes, failures

# ───────────────────────────────────────────────────────────────────────────────
# 9.  Mask file processing (for MVS masking support)
# ───────────────────────────────────────────────────────────────────────────────
def process_masks_optimized(imgs, depth_ranges, masks_dir, out_mask,
                            mode='copy', relative_links=False, max_workers=None,
                            skip_missing=True):
    """
    Process mask files with configurable mode (same as image processing).
    Masks are expected to be pre-generated in masks_dir with same filenames as images.
    Output: masks/00000001.png, masks/00000002.png, etc.
    
    If masks_dir doesn't exist or is empty, silently returns (masks are optional).
    """
    if max_workers is None:
        max_workers = min(64, mp.cpu_count() * 4)
    
    # Check if masks directory exists
    if not os.path.isdir(masks_dir):
        print(f"[INFO] No masks directory found at {masks_dir} - skipping mask processing")
        return [], []
    
    # Scan existing mask files
    print("[INFO] Scanning masks directory...")
    existing_files = set()
    try:
        with os.scandir(masks_dir) as entries:
            for entry in entries:
                if entry.is_file():
                    existing_files.add(entry.name)
    except OSError as e:
        print(f"[WARN] Cannot read masks directory: {e}")
        return [], []
    
    if not existing_files:
        print("[INFO] Masks directory is empty - skipping mask processing")
        return [], []
    
    print(f"[INFO] Found {len(existing_files)} mask files")
    
    # Build task list - match masks to images by filename
    tasks = []
    missing = []
    for i in sorted(depth_ranges.keys()):
        img = imgs[i]
        # Try exact filename match first, then try .png extension
        mask_name = img.name
        if mask_name not in existing_files:
            # Try with .png extension if original wasn't png
            base = os.path.splitext(mask_name)[0]
            mask_name = base + '.png'
        if mask_name not in existing_files:
            missing.append((i, img.name))
            continue
        src = os.path.join(masks_dir, mask_name)
        dst = os.path.join(out_mask, f"{i:08d}.png")
        needs_convert = not src.lower().endswith('.png')
        tasks.append((i, src, dst, needs_convert, mode, relative_links))
    
    if not tasks:
        print(f"[INFO] No matching masks found for {len(missing)} images")
        return [], missing
    
    print(f"[INFO] Processing {len(tasks)} masks ({len(missing)} without masks)")
    
    # Use same processing logic as images
    results = {'copied': 0, 'converted': 0, 'symlinked': 0,
               'hardlinked': 0, 'reflinked': 0, 'failed': 0}
    successes = []
    failures = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_image, task): task[0] for task in tasks}
        
        with tqdm(total=len(tasks), desc="Processing masks", unit="mask") as pbar:
            for future in as_completed(futures):
                idx, success, error, action = future.result()
                if success:
                    successes.append(idx)
                    if action in results: results[action] += 1
                else:
                    failures.append((idx, error))
                    results['failed'] += 1
                pbar.update(1)
    
    print(f"[INFO] Masks: {len(successes)} processed, {len(failures)} failed")
    return successes, failures

# ───────────────────────────────────────────────────────────────────────────────
# 10. Camera file writing
# ───────────────────────────────────────────────────────────────────────────────
def write_camera_file(item, cam_dir, use_sphere_tag=True):
    """
    Write single camera file.
    
    Args:
        use_sphere_tag: If True (default), write 'SPHERE\\nf cx cy' format for spherical cameras.
                       If False, write standard 3x3 K matrix (PINHOLE format).
    
    ACMMP depth line formats:
        SPHERE:  depth_min depth_interval n_depth_planes depth_max
        PINHOLE: depth_min depth_max <dummy> <dummy>
    """
    i, data = item
    cam_path = os.path.join(cam_dir, f"{i:08d}_cam.txt")
    
    lines = ["extrinsic"]
    for r in range(4):
        lines.append(" ".join(map(str, data['extrinsic'][r])))
    lines.append("")
    lines.append("intrinsic")
    
    d0, dint, Nd, dmax = data['depth']
    
    if use_sphere_tag and data.get('is_sphere', False):
        # SPHERE format for modified ACMMP with spherical camera support
        lines.append("SPHERE")
        lines.append(f"{data['focal']} {data['cx']} {data['cy']}")
        lines.append("")
        # SPHERE depth: depth_min depth_interval n_depth_planes depth_max
        lines.append(f"{d0} {dint} {Nd} {dmax}")
    else:
        # Standard 3x3 K matrix format (PINHOLE)
        f = data['focal']
        cx = data['cx']
        cy = data['cy']
        lines.append(f"{f} 0 {cx}")
        lines.append(f"0 {f} {cy}")
        lines.append("0 0 1")
        lines.append("")
        # PINHOLE depth: depth_min depth_max <dummy> <dummy>
        # ACMMP reads: depth_min, depth_max, dummy1, dummy2
        lines.append(f"{d0} {dmax} {dint} {Nd}")
    
    with open(cam_path, "w") as f:
        f.write("\n".join(lines))

def write_cameras_parallel(cam_data, cam_dir, use_sphere_tag=False, max_workers=None):
    if max_workers is None:
        max_workers = min(32, mp.cpu_count() * 2)
    
    items = list(cam_data.items())
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(
            executor.map(lambda item: write_camera_file(item, cam_dir, use_sphere_tag), items),
            total=len(items),
            desc="Writing cameras"
        ))

# ───────────────────────────────────────────────────────────────────────────────
# 10.  Neighbor selection worker
# ───────────────────────────────────────────────────────────────────────────────
def _select_for_image_worker(img_idx, depth_ranges, img_candidates, cam_centers, top_k, diversity_threshold):
    if img_idx not in depth_ranges: return img_idx, []
    candidates = img_candidates.get(img_idx, [])
    selected = select_diverse_multiscale_neighbors_fast(img_idx, candidates, cam_centers, top_k=top_k, diversity_threshold=diversity_threshold)
    return img_idx, selected

# ───────────────────────────────────────────────────────────────────────────────
# 11.  Main processing routine
# ───────────────────────────────────────────────────────────────────────────────
def process_scene(args):
    dense = args.dense_folder
    sparse = os.path.join(dense, "sparse")
    imgs_dir = os.path.join(dense, "images")
    masks_dir = args.masks_folder if args.masks_folder else os.path.join(dense, "masks")
    out_img = os.path.join(args.save_folder, "images")
    out_mask = os.path.join(args.save_folder, "masks")
    cam_dir = os.path.join(args.save_folder, "cams")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(cam_dir, exist_ok=True)
    # Create masks output dir only if source masks exist
    if os.path.isdir(masks_dir):
        os.makedirs(out_mask, exist_ok=True)

    # Load model
    print("[INFO] Loading COLMAP model...")
    cams, imgs_raw, pts = read_model(sparse, args.model_ext)
    
    for cid, cam in cams.items():
        if cam.model == "SPHERE":
            cams[cid] = cam._replace(model="SPHERICAL")
    
    imgs = {i: imgs_raw[k] for i, k in enumerate(sorted(imgs_raw))}
    N = len(imgs)
    print(f"[INFO] Loaded {N} images, {len(pts)} 3D points")

    # Build intrinsics
    Kdict = {}
    for cid, cam in cams.items():
        if cam.model != "SPHERICAL":
            raise ValueError(f"Expected SPHERICAL camera, got {cam.model}")
        _, cx, cy = cam.params[:3]
        estimated_f = cam.width / (2 * np.pi)
        K = np.array([[estimated_f, 0, cx], [0, estimated_f, cy], [0, 0, 1]])
        Kdict[cid] = K
    print(f"[INFO] Estimated focal length: {estimated_f:.2f} pixels")

    # Extrinsics + camera centers
    print("[INFO] Computing camera extrinsics...")
    extr, cam_centers = {}, {}
    for i, img in imgs.items():
        E = np.eye(4)
        E[:3, :3] = qvec2rotmat(img.qvec)
        E[:3, 3] = img.tvec
        extr[i] = E
        cam_centers[i] = -(E[:3, :3].T @ E[:3, 3])

    # Points3D lookup
    print("[INFO] Building point cloud lookup...")
    points3d_xyz = {pid: pt.xyz for pid, pt in pts.items()}

    # Depth ranges
    print("[INFO] Computing depth ranges...")
    depth_ranges = compute_depth_ranges(imgs, pts, extr, args.max_d, args.interval_scale, cams)
    print(f"[INFO] Valid depth ranges for {len(depth_ranges)}/{N} images")
    
    if depth_ranges:
        sample_idx = next(iter(depth_ranges.keys()))
        dmin, dint, dnum, dmax = depth_ranges[sample_idx]
        print(f"[INFO] Sample depth range (image {sample_idx}): min={dmin:.2f}, max={dmax:.2f}, planes={dnum}")

    # Spatial filtering
    print("[INFO] Building spatial index...")
    valid_keys = sorted(depth_ranges.keys())
    centers = np.array([cam_centers[i] for i in valid_keys])
    tree = cKDTree(centers)
    
    k_factor = 5 if N < 100 else (3 if N < 1000 else 2)
    k_search = min(args.top_k * k_factor, len(valid_keys))
    
    print(f"[INFO] Querying {k_search} nearest neighbors...")
    _, nnidx = tree.query(centers, k=k_search)
    
    candidate_pairs = set()
    for src_idx, neighs in enumerate(nnidx):
        src = valid_keys[src_idx]
        for n in neighs:
            if n == src_idx: continue
            dst = valid_keys[n]
            a, b = min(src, dst), max(src, dst)
            candidate_pairs.add((a, b))
    
    print(f"[INFO] {len(candidate_pairs)} candidate pairs (from {N*(N-1)//2} possible)")

    # Pre-filter by shared points
    print("[INFO] Pre-filtering pairs...")
    filtered_pairs = []
    min_shared = max(5, args.min_shared // 2)
    
    for pair in tqdm(candidate_pairs, desc="Pre-filtering"):
        i, j = pair
        shared = len(set(imgs[i].point3D_ids) & set(imgs[j].point3D_ids))
        if shared >= min_shared:
            filtered_pairs.append(pair)
    
    print(f"[INFO] {len(filtered_pairs)} pairs after pre-filter")

    # Score pairs
    print("[INFO] Scoring pairs...")
    func = partial(calc_score_enhanced_fast, images=imgs, points3d_xyz=points3d_xyz,
                   theta0=args.theta0, cam_centers=cam_centers, min_shared=args.min_shared)
    
    score_dict = {}
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.imap_unordered(func, filtered_pairs, chunksize=args.chunksize)
        for i, j, s in tqdm(results, total=len(filtered_pairs), desc="Scoring"):
            if s > 0:
                score_dict[(i, j)] = s
                score_dict[(j, i)] = s
    
    print(f"[INFO] {len(score_dict) // 2} scored pairs")

    # Select neighbors
    print("[INFO] Selecting neighbors...")
    view_sel = [[] for _ in range(N)]
    img_candidates = defaultdict(list)
    for (i, j), score in score_dict.items():
        if i in depth_ranges and j in depth_ranges:
            img_candidates[i].append((j, score))
    
    if N > 500:
        worker_func = partial(_select_for_image_worker, depth_ranges=depth_ranges,
                             img_candidates=img_candidates, cam_centers=cam_centers,
                             top_k=args.top_k, diversity_threshold=args.diversity_threshold)
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(worker_func, range(N), chunksize=10)
            for img_idx, selected in results:
                view_sel[img_idx] = selected
    else:
        for i in tqdm(range(N), desc="Neighbor selection"):
            if i not in depth_ranges: continue
            view_sel[i] = select_diverse_multiscale_neighbors_fast(
                i, img_candidates[i], cam_centers, top_k=args.top_k,
                diversity_threshold=args.diversity_threshold)
    
    avg_neighbors = np.mean([len(v) for v in view_sel if v])
    print(f"[INFO] Average neighbors: {avg_neighbors:.1f}")

    # Write camera files
    print("[INFO] Writing camera files...")
    use_sphere_tag = getattr(args, 'sphere_tag', True) and not getattr(args, 'no_sphere_tag', False)
    print(f"[INFO] Camera format: {'SPHERE' if use_sphere_tag else 'PINHOLE (3x3 K matrix)'}")
    cam_data = {}
    for i in range(N):
        if i not in depth_ranges: continue
        img = imgs[i]
        cam = cams[img.camera_id]
        _, cx, cy = cam.params[:3]
        estimated_f = cam.width / (2 * np.pi)
        cam_data[i] = {
            'extrinsic': extr[i], 'is_sphere': True,
            'focal': estimated_f, 'cx': cx, 'cy': cy,
            'K_matrix': Kdict[cam.id],
            'depth': depth_ranges[i],
            'width': cam.width, 'height': cam.height
        }
    
    write_cameras_parallel(cam_data, cam_dir, use_sphere_tag=use_sphere_tag)

    # Write pair.txt
    print("[INFO] Writing pair.txt...")
    with open(os.path.join(args.save_folder, "pair.txt"), "w") as f:
        valid_images = sorted(depth_ranges.keys())
        f.write(f"{len(valid_images)}\n")
        valid_img_to_seq = {img_id: seq_idx for seq_idx, img_id in enumerate(valid_images)}
        
        for seq_idx, i in enumerate(valid_images):
            neighbors = [(j, int(score_val)) for j, score_val in view_sel[i]
                        if j in valid_images and score_val > 0]
            f.write(f"{seq_idx}\n{len(neighbors)} ")
            for j, s in neighbors:
                f.write(f"{valid_img_to_seq[j]} {s} ")
            f.write("\n")

    # Process images with selected mode
    print(f"[INFO] Processing images (mode: {args.image_mode})...")
    successes, failures = process_images_optimized(
        imgs, depth_ranges, imgs_dir, out_img,
        mode=args.image_mode,
        relative_links=args.relative_links,
        max_workers=args.copy_workers,
        skip_missing=args.skip_images
    )
    
    if failures:
        print(f"\n[WARN] {len(failures)} images failed:")
        for idx, error in failures[:5]:
            print(f"  Image {idx}: {error}")
        if len(failures) > 5:
            print(f"  ... and {len(failures) - 5} more")
    
    # Process masks if available
    mask_successes, mask_failures = process_masks_optimized(
        imgs, depth_ranges, masks_dir, out_mask,
        mode=args.image_mode,
        relative_links=args.relative_links,
        max_workers=args.copy_workers,
        skip_missing=True
    )
    
    print(f"\n{'='*70}")
    print(f"[SUCCESS] Processing complete! Output: {args.save_folder}")
    print(f"{'='*70}")
    print(f"  Total images: {N}")
    print(f"  Valid images: {len(depth_ranges)}")
    print(f"  Processed: {len(successes)}, Failed: {len(failures)}")
    if mask_successes:
        print(f"  Masks: {len(mask_successes)} processed")
    print(f"  Avg neighbors: {avg_neighbors:.1f}")
    print(f"{'='*70}")

# ───────────────────────────────────────────────────────────────────────────────
# 13. CLI
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="COLMAP SPHERICAL → ACMMP/MVSNet (OPTIMIZED)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Image modes:
  copy      Fast buffered copy (default, works everywhere)
  symlink   Symbolic links (fastest, breaks if source moves)
  hardlink  Hard links (fast, same filesystem only)
  reflink   Copy-on-write (fast, requires btrfs/xfs/APFS)

Examples:
  # Standard copy
  python %(prog)s --dense_folder /data/dense --save_folder out
  
  # Fast symlinks (relative paths)
  python %(prog)s --dense_folder /data/dense --save_folder out --link --relative
  
  # Hard links (same filesystem)
  python %(prog)s --dense_folder /data/dense --save_folder out --image_mode hardlink
        """
    )
    
    ap.add_argument("--dense_folder", required=True)
    ap.add_argument("--save_folder", required=True)
    ap.add_argument("--model_ext", default=".txt", choices=[".txt", ".bin"])
    ap.add_argument("--max_d", type=int, default=0)
    ap.add_argument("--interval_scale", type=float, default=1.0)
    ap.add_argument("--theta0", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--min_shared", type=int, default=10)
    ap.add_argument("--chunksize", type=int, default=512)
    ap.add_argument("--diversity_threshold", type=float, default=0.3)
    ap.add_argument("--sphere_tag", action="store_true", default=True,
                    help="Write 'SPHERE' tag in cam files (default: True for spherical cameras)")
    ap.add_argument("--no_sphere_tag", action="store_true",
                    help="Write standard 3x3 K matrix instead of SPHERE tag")
    
    # Image processing options
    img_group = ap.add_argument_group('Image processing')
    img_group.add_argument("--image_mode", default="copy", 
                          choices=["copy", "symlink", "hardlink", "reflink"],
                          help="How to handle images (default: copy)")
    img_group.add_argument("--link", "-l", action="store_true",
                          help="Shortcut for --image_mode symlink")
    img_group.add_argument("--relative", "--relative_links", "-r", 
                          dest="relative_links", action="store_true",
                          help="Use relative paths for symlinks")
    img_group.add_argument("--copy_workers", type=int, default=None,
                          help="Threads for image processing (default: auto)")
    img_group.add_argument("--skip_images", "--no-images", action="store_true",
                          help="Continue even if images folder is missing/empty")
    img_group.add_argument("--masks_folder", default=None,
                          help="Path to pre-generated masks (default: dense_folder/masks)")
    
    
    args = ap.parse_args()
    
    # Handle --link shortcut
    if args.link:
        args.image_mode = "symlink"
    
    # Handle sphere tag
    use_sphere_tag = args.sphere_tag and not args.no_sphere_tag
    
    os.makedirs(args.save_folder, exist_ok=True)
    
    import time
    start = time.time()
    process_scene(args)
    print(f"\n[TIMING] Total: {time.time() - start:.1f}s")