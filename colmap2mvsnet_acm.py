#!/usr/bin/env python
"""
COLMAP → ACMMP/MVSNet converter optimized for SPHERICAL cameras with:
Example
-------
python3 colmap2mvsnet_spherical_enhanced.py \
    --dense_folder  /path/to/dense            \
    --save_folder   out                       \
    --model_ext     .bin                      \
    --top_k         20                        \
    --min_shared    15                        \
    --chunksize     64
"""

from __future__ import print_function
import os, shutil, struct, argparse, collections, multiprocessing as mp
from functools import partial
from itertools import combinations
import numpy as np
import cv2
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix, csr_matrix
from collections import defaultdict

# ───────────────────────────────────────────────────────────────────────────────
# 1.  Named-tuple data structures
# ───────────────────────────────────────────────────────────────────────────────
CameraModel = collections.namedtuple("CameraModel", ["model_id",
                                                     "model_name",
                                                     "num_params"])
Camera   = collections.namedtuple("Camera",
            ["id", "model", "width", "height", "params"])
BaseImg  = collections.namedtuple("Image",
            ["id", "qvec", "tvec", "camera_id", "name",
             "xys", "point3D_ids"])
Point3D  = collections.namedtuple("Point3D",
            ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

class Image(BaseImg):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

# Supported camera models (same IDs as COLMAP)
CAMERA_MODELS = {
    CameraModel(0,  "SIMPLE_PINHOLE",        3),
    CameraModel(1,  "PINHOLE",               4),
    CameraModel(2,  "SIMPLE_RADIAL",         4),
    CameraModel(3,  "RADIAL",                5),
    CameraModel(4,  "OPENCV",                8),
    CameraModel(5,  "OPENCV_FISHEYE",        8),
    CameraModel(6,  "FULL_OPENCV",          12),
    CameraModel(7,  "FOV",                   5),
    CameraModel(8,  "SIMPLE_RADIAL_FISHEYE", 4),
    CameraModel(9,  "RADIAL_FISHEYE",        5),
    CameraModel(10, "THIN_PRISM_FISHEYE",   12),
    CameraModel(model_id=11, model_name="SPHERE",num_params=3),
    CameraModel(model_id=12, model_name="SPHERICAL",num_params=3),
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
    cams={}
    with open(path) as f:
        for ln in f:
            if ln.lstrip().startswith("#") or not ln.strip(): continue
            s=ln.split(); cid=int(s[0]); model=s[1]; w,h=map(int,s[2:4])
            params=np.fromiter(map(float,s[4:]),float)
            cams[cid]=Camera(cid,model,w,h,params)
    return cams

def read_cameras_binary(path):
    cams={}
    with open(path,"rb") as f:
        n=_read(f,8,"Q")[0]
        for _ in range(n):
            cid,mid,w,h=_read(f,24,"iiQQ")
            cm=CAMERA_MODEL_IDS[mid]
            params=np.array(_read(f,8*cm.num_params,"d"*cm.num_params))
            cams[cid]=Camera(cid,cm.model_name,w,h,params)
    return cams

def read_images_text(path):
    imgs={}
    with open(path) as f:
        while True:
            ln=f.readline();
            if not ln: break
            if ln.lstrip().startswith("#") or not ln.strip(): continue
            s=ln.split()
            iid=int(s[0]); qvec=np.fromiter(map(float,s[1:5]),float)
            tvec=np.fromiter(map(float,s[5:8]),float); cid=int(s[8]); name=s[9]
            track=f.readline().split()
            xys=np.column_stack([list(map(float,track[0::3])),
                                 list(map(float,track[1::3]))])
            pids=np.array(list(map(int,track[2::3])),dtype=int)
            imgs[iid]=Image(iid,qvec,tvec,cid,name,xys,pids)
    return imgs

def read_images_binary(path):
    imgs={}
    with open(path,"rb") as f:
        n=_read(f,8,"Q")[0]
        for _ in range(n):
            iid,*vals=_read(f,64,"idddddddi")
            qvec=np.array(vals[0:4]); tvec=np.array(vals[4:7]); cid=vals[7]
            name=b""
            while True:
                c=_read(f,1,"c")[0]
                if c==b"\x00": break
                name+=c
            name=name.decode()
            npts=_read(f,8,"Q")[0]
            data=_read(f,24*npts,"ddq"*npts)
            xys=np.column_stack([data[0::3],data[1::3]])
            pids=np.array(data[2::3],dtype=int)
            imgs[iid]=Image(iid,qvec,tvec,cid,name,xys,pids)
    return imgs

def read_points3D_text(path):
    pts={}
    with open(path) as f:
        for ln in f:
            if ln.lstrip().startswith("#") or not ln.strip(): continue
            s=ln.split(); pid=int(s[0])
            xyz=np.fromiter(map(float,s[1:4]),float)
            rgb=np.fromiter(map(int,s[4:7]),int); err=float(s[7])
            img_ids=np.array(list(map(int,s[8::2])),dtype=int)
            idxs   =np.array(list(map(int,s[9::2])),dtype=int)
            pts[pid]=Point3D(pid,xyz,rgb,err,img_ids,idxs)
    return pts

def read_points3D_binary(path):
    pts={}
    with open(path,"rb") as f:
        n=_read(f,8,"Q")[0]
        for _ in range(n):
            pid,x,y,z,r,g,b,err=_read(f,43,"QdddBBBd")
            length=_read(f,8,"Q")[0]
            track=_read(f,8*length,"ii"*length)
            img_ids=np.array(track[0::2],dtype=int)
            idxs   =np.array(track[1::2],dtype=int)
            pts[pid]=Point3D(pid,np.array([x,y,z]),np.array([r,g,b]),err,img_ids,idxs)
    return pts

def read_model(sparse_dir, ext):
    if ext==".txt":
        cams=read_cameras_text(os.path.join(sparse_dir,"cameras"+ext))
        imgs=read_images_text(os.path.join(sparse_dir,"images"+ext))
        pts =read_points3D_text(os.path.join(sparse_dir,"points3D"+ext))
    else:
        cams=read_cameras_binary(os.path.join(sparse_dir,"cameras"+ext))
        imgs=read_images_binary(os.path.join(sparse_dir,"images"+ext))
        pts =read_points3D_binary(os.path.join(sparse_dir,"points3D"+ext))
    return cams,imgs,pts

# ───────────────────────────────────────────────────────────────────────────────
# 4.  Math helpers
# ───────────────────────────────────────────────────────────────────────────────
def qvec2rotmat(q):
    w,x,y,z=q
    return np.array([
        [1-2*y*y-2*z*z,  2*x*y-2*w*z,  2*x*z+2*w*y],
        [2*x*y+2*w*z,    1-2*x*x-2*z*z,2*y*z-2*w*x],
        [2*x*z-2*w*y,    2*y*z+2*w*x,  1-2*x*x-2*y*y],
    ])

# ───────────────────────────────────────────────────────────────────────────────
# 5.  Depth-range computation (spherical-optimized)
# ───────────────────────────────────────────────────────────────────────────────
def compute_depth_ranges(images, points3d, extrinsic, max_d, interval_scale, cams):
    """Compute depth ranges using RADIAL depth for spherical cameras"""
    depth_ranges = {}
    skipped_images = []
    
    for i, img in images.items():
        zs = []
        
        for pid in img.point3D_ids:
            if pid < 0 or pid not in points3d:
                continue
            X = np.append(points3d[pid].xyz, 1.0)
            X_cam = extrinsic[i] @ X
            
            # Spherical cameras: use radial depth
            d = np.linalg.norm(X_cam[:3])
            
            if d <= 0:
                continue
            zs.append(d)
        
        # Require minimum points for reliable depth estimation
        if len(zs) < 10:
            skipped_images.append((i, len(zs)))
            continue
            
        zs_sorted = sorted(zs)
        dmin = zs_sorted[int(len(zs_sorted)*0.2)] * 0.75
        dmax = zs_sorted[int(len(zs_sorted)*0.8)] * 1.25
        
        # Spherical-specific depth sampling
        if max_d == 0:
            # For spherical: use logarithmic depth distribution
            depth_range = dmax - dmin
            log_range = np.log(dmax / dmin)
            
            # Adaptive depth planes based on scene scale
            if log_range < 1.0:  # Narrow depth range
                depth_num = 64
            elif log_range < 2.0:  # Medium range
                depth_num = 128
            elif log_range < 3.0:  # Wide range
                depth_num = 192
            else:  # Very wide range (common in 360° scenes)
                depth_num = 256
            
            depth_num = min(256, max(32, depth_num))
        else:
            depth_num = max_d
            
        dint = (dmax - dmin) / (depth_num - 1) / interval_scale
        depth_ranges[i] = (dmin, dint, depth_num, dmax)
    
    if skipped_images:
        print(f"[WARN] Skipped {len(skipped_images)} images with insufficient points:")
        for img_id, npts in skipped_images[:5]:
            print(f"  Image {img_id}: {npts} points")
    
    return depth_ranges

# ───────────────────────────────────────────────────────────────────────────────
# 6.  Enhanced pair scoring for spherical cameras (OPTIMIZED)
# ───────────────────────────────────────────────────────────────────────────────
def calc_baseline_to_depth_ratio_vectorized(ci, cj, shared_xyz):
    """
    Vectorized baseline-to-depth ratio calculation.
    MUCH faster for large point sets.
    """
    baseline = np.linalg.norm(ci - cj)
    
    if len(shared_xyz) == 0:
        return 0.0
    
    # Vectorized depth calculation
    depths = np.linalg.norm(shared_xyz - ci, axis=1)
    median_depth = np.median(depths)
    
    if median_depth < 1e-6:
        return 0.0
    
    ratio = baseline / median_depth
    
    # Score based on optimal baseline-to-depth ratio
    if 0.05 < ratio < 0.2:
        return 1.0  # Excellent
    elif 0.03 < ratio < 0.3:
        return 0.7  # Good
    elif 0.01 < ratio < 0.5:
        return 0.4  # Acceptable
    else:
        return 0.1  # Poor

def calc_score_enhanced_fast(pair, images, points3d_xyz, theta0, cam_centers, min_shared=5):
    """
    Optimized scoring with:
    1. Early termination
    2. Vectorized operations
    3. Pre-computed camera centers
    4. Direct XYZ lookup (no dict access in loop)
    """
    i, j = pair
    
    # Early check: shared points (set intersection is fast)
    shared_pids = set(images[i].point3D_ids) & set(images[j].point3D_ids)
    shared_pids = [pid for pid in shared_pids if pid != -1 and pid in points3d_xyz]
    
    if len(shared_pids) < min_shared:
        return i, j, 0.0
    
    # Pre-computed camera centers
    ci = cam_centers[i]
    cj = cam_centers[j]
    
    # Quick baseline check (too close or too far is bad)
    baseline = np.linalg.norm(ci - cj)
    if baseline < 0.01:  # Too close
        return i, j, 0.0
    
    # Get all shared point coordinates at once (vectorized)
    shared_xyz = np.array([points3d_xyz[pid] for pid in shared_pids])
    
    # 1. Base score: number of shared points
    base_score = float(len(shared_pids))
    
    # 2. Baseline-to-depth ratio score (vectorized)
    btd_score = calc_baseline_to_depth_ratio_vectorized(ci, cj, shared_xyz)
    
    if btd_score < 0.1:  # Early termination if geometry is poor
        return i, j, base_score * btd_score
    
    # 3. Triangulation angle score (vectorized)
    vi = shared_xyz - ci
    vj = shared_xyz - cj
    
    norms_i = np.linalg.norm(vi, axis=1)
    norms_j = np.linalg.norm(vj, axis=1)
    
    # Avoid division by zero
    valid_mask = (norms_i > 1e-6) & (norms_j > 1e-6)
    if not np.any(valid_mask):
        return i, j, 0.0
    
    dots = np.sum(vi[valid_mask] * vj[valid_mask], axis=1)
    cos_angles = np.clip(dots / (norms_i[valid_mask] * norms_j[valid_mask]), -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_angles))
    
    angle_75 = np.percentile(angles, 75)
    if angle_75 < theta0:
        angle_score = 0.1  # Too small
    elif angle_75 < 5.0:
        angle_score = 0.5  # Marginal
    else:
        angle_score = 1.0  # Good
    
    # Combine scores
    final_score = base_score * btd_score * angle_score
    
    return i, j, final_score

# ───────────────────────────────────────────────────────────────────────────────
# 7.  Multi-scale baseline selection with diversity constraint (OPTIMIZED)
# ───────────────────────────────────────────────────────────────────────────────
def select_diverse_multiscale_neighbors_fast(ref_idx, candidates, cam_centers, 
                                            top_k=20, diversity_threshold=0.3):
    """
    Optimized neighbor selection:
    1. Pre-sorted candidates by score
    2. Greedy selection with KD-tree for diversity checks
    3. Early termination
    """
    if not candidates:
        return []
    
    ci = cam_centers[ref_idx]
    
    # Add baseline distance to each candidate (vectorized where possible)
    candidate_indices = np.array([idx for idx, _ in candidates])
    candidate_scores = np.array([score for _, score in candidates])
    candidate_positions = np.array([cam_centers[idx] for idx, _ in candidates])
    
    # Calculate all baselines at once
    baselines = np.linalg.norm(candidate_positions - ci, axis=1)
    
    # Sort by score descending
    sort_idx = np.argsort(-candidate_scores)
    candidate_indices = candidate_indices[sort_idx]
    candidate_scores = candidate_scores[sort_idx]
    baselines = baselines[sort_idx]
    candidate_positions = candidate_positions[sort_idx]
    
    if len(candidates) <= top_k:
        return [(int(idx), float(score)) 
                for idx, score in zip(candidate_indices, candidate_scores)]
    
    # Multi-scale target baselines
    min_baseline = np.min(baselines)
    max_baseline = np.max(baselines)
    
    if max_baseline / (min_baseline + 1e-6) > 10:
        log_min = np.log(min_baseline + 1e-6)
        log_max = np.log(max_baseline + 1e-6)
        target_baselines = np.exp(np.linspace(log_min, log_max, top_k))
    else:
        target_baselines = np.linspace(min_baseline, max_baseline, top_k)
    
    # Greedy selection
    selected = []
    selected_positions = []
    used_mask = np.zeros(len(candidates), dtype=bool)
    
    for target_baseline in target_baselines:
        if len(selected) >= top_k:
            break
        
        best_idx = -1
        best_metric = -1
        
        # Find best unused candidate near this target baseline
        for local_idx in range(len(candidates)):
            if used_mask[local_idx]:
                continue
            
            # Check diversity (only if we have some selections already)
            if len(selected) >= 5:
                pos = candidate_positions[local_idx]
                # Quick check: minimum distance to any selected
                if selected_positions:
                    dists = np.linalg.norm(np.array(selected_positions) - pos, axis=1)
                    min_dist = np.min(dists)
                    if min_dist < diversity_threshold * baselines[local_idx]:
                        continue
            
            # Combined metric
            baseline_diff = abs(baselines[local_idx] - target_baseline) / (target_baseline + 1e-6)
            combined_metric = 0.7 * candidate_scores[local_idx] - 0.3 * baseline_diff
            
            if combined_metric > best_metric:
                best_metric = combined_metric
                best_idx = local_idx
        
        if best_idx >= 0:
            idx = int(candidate_indices[best_idx])
            score = float(candidate_scores[best_idx])
            selected.append((idx, score))
            selected_positions.append(candidate_positions[best_idx])
            used_mask[best_idx] = True
    
    # Fill remaining with top scores
    if len(selected) < top_k:
        for local_idx in range(len(candidates)):
            if used_mask[local_idx]:
                continue
            idx = int(candidate_indices[local_idx])
            score = float(candidate_scores[local_idx])
            selected.append((idx, score))
            if len(selected) >= top_k:
                break
    
    return selected

# ───────────────────────────────────────────────────────────────────────────────
# 8.  Multiprocessing worker functions (module-level for pickling)
# ───────────────────────────────────────────────────────────────────────────────
def _write_camera_file_worker(item, cam_dir):
    """Worker function to write a single camera file - SPHERE FORMAT"""
    i, data = item
    cam_path = os.path.join(cam_dir, f"{i:08d}_cam.txt")
    with open(cam_path, "w") as f:
        # Extrinsic (4x4)
        f.write("extrinsic\n")
        for r in range(4):
            f.write(" ".join(map(str, data['extrinsic'][r])) + "\n")
        
        f.write("\n")
        
        # Intrinsic - SPHERE format: "SPHERE\nf cx cy\n"
        f.write("intrinsic\n")
        if data.get('is_sphere', True):
            f.write("SPHERE\n")
            focal = data['focal']
            cx = data['cx']
            cy = data['cy']
            f.write(f"{focal} {cx} {cy}\n")
        else:
            # For non-sphere cameras: write 3x3 K matrix
            K = data['K_matrix']
            for r in range(3):
                f.write(" ".join(map(str, K[r])) + "\n")
        
        f.write("\n")
        
        # Depth range
        d0, dint, Nd, dmax = data['depth']
        f.write(f"{d0} {dint} {Nd} {dmax}\n")

def _copy_image_worker(i, imgs, depth_ranges, imgs_dir, out_img):
    """Worker function to copy a single image"""
    if i not in depth_ranges:
        return None
    
    img = imgs[i]
    src = os.path.join(imgs_dir, img.name)
    dst = os.path.join(out_img, f"{i:08d}.jpg")
    
    if not os.path.exists(src):
        return f"[WARN] Image not found: {src}"
    
    try:
        if not src.lower().endswith(".jpg"):
            img_data = cv2.imread(src)
            if img_data is not None:
                cv2.imwrite(dst, img_data)
            else:
                return f"[WARN] Failed to read: {src}"
        else:
            shutil.copyfile(src, dst)
    except Exception as e:
        return f"[ERROR] Failed to copy {src}: {e}"
    
    return None

def _select_for_image_worker(img_idx, depth_ranges, img_candidates, cam_centers, top_k, diversity_threshold):
    """Worker function to select neighbors for a single image"""
    if img_idx not in depth_ranges:
        return img_idx, []
    
    candidates = img_candidates.get(img_idx, [])
    selected = select_diverse_multiscale_neighbors_fast(
        img_idx, candidates, cam_centers,
        top_k=top_k,
        diversity_threshold=diversity_threshold
    )
    return img_idx, selected

# ───────────────────────────────────────────────────────────────────────────────
# 9.  Main processing routine (HIGHLY OPTIMIZED)
# ───────────────────────────────────────────────────────────────────────────────
def process_scene(args):
    dense=args.dense_folder
    sparse=os.path.join(dense,"sparse")
    imgs_dir=os.path.join(dense,"images")
    out_img=os.path.join(args.save_folder,"images")
    cam_dir=os.path.join(args.save_folder,"cams")
    os.makedirs(out_img,exist_ok=True)
    os.makedirs(cam_dir,exist_ok=True)

    # ---------- Load model ----------
    print("[INFO] Loading COLMAP model...")
    cams,imgs_raw,pts=read_model(sparse,args.model_ext)
    
    # Normalize to SPHERICAL model
    for cid, cam in cams.items():
        if cam.model == "SPHERE":
            cam = cam._replace(model="SPHERICAL")
            cams[cid] = cam
    
    # Use consistent 0-based indexing
    imgs = {i: imgs_raw[k] for i, k in enumerate(sorted(imgs_raw))}
    N = len(imgs)
    print(f"[INFO] Loaded {N} images, {len(pts)} 3D points")

    # ---------- Build intrinsics (spherical-optimized) ----------
    Kdict={}
    for cid,cam in cams.items():
        if cam.model != "SPHERICAL":
            raise ValueError(f"Expected SPHERICAL camera, got {cam.model}")
        
        # For spherical: estimate focal length from image dimensions
        _, cx, cy = cam.params[:3]
        estimated_f = cam.width / (2 * np.pi)
        K = np.array([[estimated_f, 0, cx], [0, estimated_f, cy], [0, 0, 1]])
        Kdict[cid] = K
    
    print(f"[INFO] Estimated focal length: {estimated_f:.2f} pixels")

    # ---------- Extrinsics + Pre-compute camera centers ----------
    print("[INFO] Computing camera extrinsics...")
    extr = {}
    cam_centers = {}
    for i, img in imgs.items():
        E = np.eye(4)
        E[:3,:3] = qvec2rotmat(img.qvec)
        E[:3,3] = img.tvec
        extr[i] = E
        # Pre-compute camera center for fast access
        cam_centers[i] = -(E[:3,:3].T @ E[:3,3])

    # ---------- Pre-compute points3D XYZ lookup (MAJOR SPEEDUP) ----------
    print("[INFO] Building point cloud lookup...")
    points3d_xyz = {pid: pt.xyz for pid, pt in pts.items()}

    # ---------- Depth ranges ----------
    print("[INFO] Computing depth ranges...")
    depth_ranges = compute_depth_ranges(imgs, pts, extr,
                                       args.max_d, args.interval_scale, cams)
    print(f"[INFO] Valid depth ranges for {len(depth_ranges)}/{N} images")
    if depth_ranges:
        sample_idx = next(iter(depth_ranges.keys()))
        dmin, dint, dnum, dmax = depth_ranges[sample_idx]
        print(f"[INFO] Sample depth range (image {sample_idx}): "
              f"min={dmin:.2f}, max={dmax:.2f}, planes={dnum}")

    # ---------- OPTIMIZATION: Progressive pair filtering ----------
    print("[INFO] Building spatial index with progressive filtering...")
    valid_keys = sorted(depth_ranges.keys())
    centers = np.array([cam_centers[i] for i in valid_keys])
    
    # Build KD-tree
    tree = cKDTree(centers)
    
    # Adaptive k_search based on dataset size
    if N < 100:
        k_factor = 5
    elif N < 1000:
        k_factor = 3
    else:  # Large datasets: be more selective
        k_factor = 2
    
    k_search = min(args.top_k * k_factor, len(valid_keys))
    
    # Query KD-tree (fast spatial filtering)
    print(f"[INFO] Querying {k_search} nearest neighbors per image...")
    _, nnidx = tree.query(centers, k=k_search)
    
    # Build candidate pairs (only from spatial neighbors)
    candidate_pairs = set()
    for src_idx, neighs in enumerate(nnidx):
        src = valid_keys[src_idx]
        for n in neighs:
            if n == src_idx:
                continue
            dst = valid_keys[n]
            a, b = min(src, dst), max(src, dst)
            candidate_pairs.add((a, b))
    
    print(f"[INFO] Spatial filtering: {len(candidate_pairs)} candidate pairs "
          f"(reduced from {N*(N-1)//2} possible)")

    # ---------- OPTIMIZATION: Quick pre-filter by shared points ----------
    print("[INFO] Pre-filtering pairs by shared point count...")
    filtered_pairs = []
    min_shared = max(5, args.min_shared // 2)  # Relaxed threshold for pre-filter
    
    for pair in tqdm(candidate_pairs, desc="Pre-filtering"):
        i, j = pair
        shared = len(set(imgs[i].point3D_ids) & set(imgs[j].point3D_ids))
        if shared >= min_shared:
            filtered_pairs.append(pair)
    
    print(f"[INFO] After pre-filter: {len(filtered_pairs)} pairs "
          f"(removed {len(candidate_pairs) - len(filtered_pairs)})")

    # ---------- OPTIMIZATION: Use sparse matrix instead of dense N×N ----------
    print("[INFO] Scoring pairs with enhanced metrics...")
    
    # Prepare scoring function with pre-computed data
    func = partial(calc_score_enhanced_fast, 
                   images=imgs, 
                   points3d_xyz=points3d_xyz,
                   theta0=args.theta0, 
                   cam_centers=cam_centers,
                   min_shared=args.min_shared)
    
    # Use sparse matrix (HUGE memory savings for large N)
    score_dict = {}
    
    # Process in parallel with progress bar
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.imap_unordered(func, filtered_pairs, chunksize=args.chunksize)
        
        for i, j, s in tqdm(results, total=len(filtered_pairs), desc="Scoring"):
            if s > 0:
                score_dict[(i, j)] = s
                score_dict[(j, i)] = s
    
    print(f"[INFO] Scored pairs with non-zero score: {len(score_dict) // 2}")

    # ---------- OPTIMIZATION: Build neighbor lists directly (no full matrix) ----------
    print("[INFO] Selecting diverse multi-scale neighbors...")
    view_sel = [[] for _ in range(N)]
    
    # Build candidate lists per image
    img_candidates = defaultdict(list)
    for (i, j), score in score_dict.items():
        if i in depth_ranges and j in depth_ranges:
            img_candidates[i].append((j, score))
    
    # Select neighbors in parallel (for large datasets)
    if N > 500:
        print("[INFO] Using parallel neighbor selection...")
        
        worker_func = partial(_select_for_image_worker,
                             depth_ranges=depth_ranges,
                             img_candidates=img_candidates,
                             cam_centers=cam_centers,
                             top_k=args.top_k,
                             diversity_threshold=args.diversity_threshold)
        
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(worker_func, range(N), chunksize=10)
            for img_idx, selected in results:
                view_sel[img_idx] = selected
    else:
        # Sequential for small datasets
        for i in tqdm(range(N), desc="Neighbor selection"):
            if i not in depth_ranges:
                continue
            candidates = img_candidates[i]
            view_sel[i] = select_diverse_multiscale_neighbors_fast(
                i, candidates, cam_centers,
                top_k=args.top_k,
                diversity_threshold=args.diversity_threshold
            )
    
    # Statistics
    avg_neighbors = np.mean([len(v) for v in view_sel if v])
    print(f"[INFO] Average neighbors per image: {avg_neighbors:.1f}")

    # ---------- Write camera files (batched I/O) - FIXED FORMAT ----------
    print("[INFO] Writing camera files...")
    
    # Prepare all camera data first
    cam_data = {}
    for i in range(N):
        if i not in depth_ranges:
            continue
        
        img = imgs[i]
        cam = cams[img.camera_id]
        d0, dint, Nd, dmax = depth_ranges[i]
        
        # For SPHERE cameras: use the simple f, cx, cy format
        _, cx, cy = cam.params[:3]
        estimated_f = cam.width / (2 * np.pi)  # or max(cx, cy) as in original
        
        cam_data[i] = {
            'extrinsic': extr[i],
            'is_sphere': True,
            'focal': estimated_f,
            'cx': cx,
            'cy': cy,
            'K_matrix': Kdict[cam.id],  # Keep for non-sphere fallback
            'depth': (d0, dint, Nd, dmax),
            'width': cam.width,
            'height': cam.height
        }
    
    # Write camera files using worker function
    write_func = partial(_write_camera_file_worker, cam_dir=cam_dir)
    
    if N > 100:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            list(tqdm(pool.imap_unordered(write_func, cam_data.items()),
                     total=len(cam_data), desc="Writing cameras"))
    else:
        for item in tqdm(cam_data.items(), desc="Writing cameras"):
            write_func(item)

    # ---------- Write pair.txt (FIXED: O(1) lookup instead of O(n)) ----------
    print("[INFO] Writing pair.txt...")
    with open(os.path.join(args.save_folder, "pair.txt"), "w") as f:
        valid_images = sorted(depth_ranges.keys())
        f.write(f"{len(valid_images)}\n")
        
        # Pre-compute mapping for O(1) lookup (PERFORMANCE FIX)
        valid_img_to_seq = {img_id: seq_idx for seq_idx, img_id in enumerate(valid_images)}
        
        for seq_idx, i in enumerate(valid_images):
            neighbors = [(j, int(score_val)) for j, score_val in view_sel[i] 
                        if j in valid_images and score_val > 0]
            
            f.write(f"{seq_idx}\n{len(neighbors)} ")
            for j, s in neighbors:
                j_seq_idx = valid_img_to_seq[j]  # O(1) lookup instead of O(n)
                f.write(f"{j_seq_idx} {s} ")
            f.write("\n")

    # ---------- Copy/convert images (parallel with proper error handling) ----------
    print("[INFO] Copying images...")
    
    copy_func = partial(_copy_image_worker, 
                       imgs=imgs, 
                       depth_ranges=depth_ranges,
                       imgs_dir=imgs_dir, 
                       out_img=out_img)
    
    # Parallel image copying
    with mp.Pool(processes=min(mp.cpu_count(), 4)) as pool:  # Limit to 4 for I/O
        warnings = list(tqdm(pool.imap_unordered(copy_func, range(N)),
                            total=N, desc="Copying images"))
    
    # Print any warnings
    warnings = [w for w in warnings if w is not None]
    if warnings:
        print("\n" + "\n".join(warnings[:10]))
        if len(warnings) > 10:
            print(f"... and {len(warnings) - 10} more warnings")
    
    print(f"\n{'='*70}")
    print(f"[SUCCESS] Processing complete!")
    print(f"[SUCCESS] Output saved to {args.save_folder}")
    print(f"{'='*70}")
    print(f"Statistics:")
    print(f"  - Total images: {N}")
    print(f"  - Valid images: {len(depth_ranges)}")
    print(f"  - Avg neighbors: {avg_neighbors:.1f}")
    print(f"  - Scored pairs: {len(score_dict) // 2}")
    print(f"{'='*70}")

# ───────────────────────────────────────────────────────────────────────────────
# 10.  CLI
# ───────────────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    ap=argparse.ArgumentParser(
       description="Convert COLMAP SPHERICAL cameras to ACMMP/MVSNet format "
                   "with enhanced pair selection (OPTIMIZED for 1000s of images).",
       formatter_class=argparse.RawDescriptionHelpFormatter)
    
    ap.add_argument("--dense_folder", required=True,
                    help="Folder with sparse/ & images/")
    ap.add_argument("--save_folder", required=True,
                    help="Output dir")
    ap.add_argument("--model_ext", default=".txt", choices=[".txt", ".bin"],
                    help="COLMAP model format (.bin recommended for large datasets)")
    ap.add_argument("--max_d", type=int, default=0,
                    help="Max depth planes (0=auto, uses 64-256 based on scene)")
    ap.add_argument("--interval_scale", type=float, default=1.0,
                    help="Depth interval scale factor")
    ap.add_argument("--theta0", type=float, default=1.0,
                    help="Min triangulation angle in degrees (1.0-5.0 recommended)")
    ap.add_argument("--top_k", type=int, default=20,
                    help="Max neighbors per image (15-30 recommended)")
    ap.add_argument("--min_shared", type=int, default=10,
                    help="Min shared 3D points for valid pair (10-30 recommended)")
    ap.add_argument("--chunksize", type=int, default=512,
                    help="Multiprocessing chunk size (512-2048 for large datasets)")
    ap.add_argument("--diversity_threshold", type=float, default=0.3,
                    help="Spatial diversity threshold (0.2=strict, 0.5=relaxed)")
    
    args = ap.parse_args()
    os.makedirs(args.save_folder, exist_ok=True)

    
    import time
    start_time = time.time()
    
    process_scene(args)
    
    elapsed = time.time() - start_time
    print()
    print(f"[TIMING] Total processing time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print("=" * 70)