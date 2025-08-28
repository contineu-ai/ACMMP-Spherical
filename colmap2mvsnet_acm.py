#!/usr/bin/env python
"""
COLMAP → ACMMP/MVSNet converter with:

* tqdm progress bars
* pair-pruning (top-K neighbours, min shared tracks)
* robust skip of images with no 3-D points
* binary (.bin) or text (.txt) sparse model
* FIXED: spherical camera support with garbage focal length handling

Example
-------
python3 colmap2mvsnet_acm.py \
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
            ln=f.readline();  # first line per image
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
            # read null-terminated name
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

def read_points3d_binary(path):
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
        pts =read_points3d_binary(os.path.join(sparse_dir,"points3D"+ext))
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
# 5.  Depth-range computation
# ───────────────────────────────────────────────────────────────────────────────
def compute_depth_ranges(images, points3d, extrinsic, intrinsic,
                          max_d, interval_scale, cams):
    depth_ranges = {}
    for i, img in images.items():
        zs = []
        cam_model = cams[img.camera_id].model
        
        for pid in img.point3D_ids:
            if pid < 0:
                continue
            X = np.append(points3d[pid].xyz, 1.0)
            X_cam = extrinsic[i] @ X
            
            if cam_model == "SPHERE":
                d = np.linalg.norm(X_cam[:3])  # radial depth
            else:
                d = X_cam[2]  # pinhole: use z only
                
            if d <= 0:
                continue
            zs.append(d)
        
        if len(zs) == 0:  # Skip images with no valid points
            continue
            
        zs_sorted = sorted(zs)
        dmin = zs_sorted[int(len(zs_sorted)*0.2)] * 0.75
        dmax = zs_sorted[int(len(zs_sorted)*0.8)] * 1.25
        
        # Handle depth_num calculation differently for spherical cameras
        if max_d == 0:
            if cam_model == "SPHERE":
                # For spherical cameras, don't use focal length (garbage param)
                # Use depth sampling based on scene scale instead
                depth_range = dmax - dmin
                scene_scale = dmin  # use minimum depth as scene scale reference
                
                # Heuristic: more depth planes for closer/larger scenes
                if scene_scale < 1.0:  # very close scene
                    depth_num = min(256, max(64, int(depth_range / 0.01)))
                elif scene_scale < 10.0:  # medium distance scene  
                    depth_num = min(192, max(48, int(depth_range / 0.05)))
                else:  # far scene
                    depth_num = min(128, max(32, int(depth_range / 0.1)))
            else:
                # Original pinhole calculation
                K = intrinsic[img.camera_id]
                p1 = np.array([K[0,2], K[1,2], 1])
                p2 = p1 + np.array([1, 0, 0])
                P1 = (np.linalg.inv(K) @ p1) * dmin
                P1 = np.linalg.inv(extrinsic[i][:3,:3]) @ (P1 - extrinsic[i][:3,3])
                P2 = (np.linalg.inv(K) @ p2) * dmin
                P2 = np.linalg.inv(extrinsic[i][:3,:3]) @ (P2 - extrinsic[i][:3,3])
                depth_num = int((1/dmin - 1/dmax) / (1/dmin - 1/(dmin + np.linalg.norm(P2-P1))))
        else:
            depth_num = max_d
            
        dint = (dmax - dmin) / (depth_num - 1) / interval_scale
        depth_ranges[i] = (dmin, dint, depth_num, dmax)
    
    return depth_ranges

# ───────────────────────────────────────────────────────────────────────────────
# 6.  Pair scoring util (triangulation-angle filter)
# ───────────────────────────────────────────────────────────────────────────────
def angle_between(ci,cj,p):
    return np.degrees(np.arccos(
        np.clip(np.dot(ci-p,cj-p)/
                (np.linalg.norm(ci-p)*np.linalg.norm(cj-p)), -1.0,1.0)))

def calc_shared(pair, images):
    i, j = pair
    return len(set(images[i].point3D_ids) & set(images[j].point3D_ids))

def calc_score(pair, images, points3d, theta0, extrinsic):
    i, j = pair
    shared = set(images[i].point3D_ids) & set(images[j].point3D_ids)
    if not shared:
        return i, j, 0.0
    
    # Use 0-based indexing consistently
    ci = -(extrinsic[i][:3,:3].T @ extrinsic[i][:3,3])
    cj = -(extrinsic[j][:3,:3].T @ extrinsic[j][:3,3])
    
    angs = [angle_between(ci, cj, points3d[pid].xyz)
            for pid in shared if pid != -1]
    
    if not angs: 
        return i, j, 0.0
    if np.percentile(angs, 75) < theta0:
        return i, j, 0.0
    return i, j, float(len(shared))

# ───────────────────────────────────────────────────────────────────────────────
# 7.  Main processing routine
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
    cams,imgs_raw,pts=read_model(sparse,args.model_ext)
    # FIXED: Use consistent 0-based indexing
    imgs = {i: imgs_raw[k] for i, k in enumerate(sorted(imgs_raw))}
    N = len(imgs)

    # ---------- Build intrinsics ----------
    param_map={
        'SIMPLE_PINHOLE':['f','cx','cy'],
        'PINHOLE':['fx','fy','cx','cy'],
        'SIMPLE_RADIAL':['f','cx','cy','k'],
        'RADIAL':['f','cx','cy','k1','k2'],
        'OPENCV':['fx','fy','cx','cy','k1','k2','p1','p2'],
        'OPENCV_FISHEYE':['fx','fy','cx','cy','k1','k2','k3','k4'],
        'FULL_OPENCV':['fx','fy','cx','cy','k1','k2','p1','p2',
                       'k3','k4','k5','k6'],
        'FOV':['fx','fy','cx','cy','omega'],
        'THIN_PRISM_FISHEYE':['fx','fy','cx','cy','k1','k2',
                              'p1','p2','k3','k4','sx1','sy1'],
        'SPHERE': ['f','cx','cy'],
    }
    
    Kdict={}
    for cid,cam in cams.items():
        if cam.model == "SPHERICAL":
            cam = cam._replace(model="SPHERE")
            cams[cid] = cam
        if cam.model == "SPHERE":
            # For spherical cameras, focal length is garbage, estimate from image center
            _, cx, cy = cam.params[:3]  # ignore focal length (garbage)
            # Estimate focal length from image center (simple heuristic)
            estimated_f = max(cx, cy)
            K = np.array([[estimated_f, 0, cx], [0, estimated_f, cy], [0, 0, 1]])
            Kdict[cid] = K
        else:
            keys=param_map[cam.model]
            vals=dict(zip(keys,cam.params))
            if 'f' in vals:
                vals['fx']=vals['fy']=vals['f']
            K=np.eye(3)
            K[0,0]=vals['fx']; K[1,1]=vals['fy']
            K[0,2]=vals['cx']; K[1,2]=vals['cy']
            Kdict[cid]=K

    # ---------- Extrinsics ----------
    extr={}
    for i,img in imgs.items():
        E=np.eye(4)
        E[:3,:3]=qvec2rotmat(img.qvec)
        E[:3,3]=img.tvec
        extr[i]=E

    # ---------- Depth ranges ----------
    depth_ranges=compute_depth_ranges(imgs,pts,extr,Kdict,
                                      args.max_d,args.interval_scale,cams)
    print("depth_ranges for first valid image:",next(iter(depth_ranges.items())))

    # ---------- Candidate pairs via camera-center proximity ----------
    # 1) collect only the images that survived depth_ranges
    keys = sorted(depth_ranges.keys())  # 0-based indices

    # 2) build an (M×3) array of world-space centers C_i = -(Rᵀ·t)
    centers = np.stack([
        -(extr[i][:3,:3].T @ extr[i][:3,3])
        for i in keys
    ])

    # 3) KD-tree & K-nearest lookup
    tree     = cKDTree(centers)
    k_search = args.top_k + 1   # +1 b/c query includes self
    _, nnidx = tree.query(centers, k=k_search)

    # 4) collect unique zero-based index pairs
    candidate_pairs = set()
    for src_idx, neighs in enumerate(nnidx):
        src = keys[src_idx]  # Already 0-based
        for n in neighs:
            if n == src_idx: continue   # skip self
            dst = keys[n]
            a, b = min(src, dst), max(src, dst)
            candidate_pairs.add((a, b))

    all_pairs = list(candidate_pairs)
    # now len(all_pairs) ≃ N·top_k instead of ~N²/2
    shared_cnt = [calc_shared(p, imgs) for p in all_pairs]

    top_pairs=[]
    valid = list(depth_ranges.keys())  # 0-based
    neighbor_bins={i:[] for i in valid}
    for (pair,s) in sorted(zip(all_pairs,shared_cnt),
                           key=lambda x:x[1],reverse=True):
        i,j=pair
        if s<args.min_shared: break
        if (len(neighbor_bins[i])<args.top_k and
            len(neighbor_bins[j])<args.top_k):
            neighbor_bins[i].append(pair)
            neighbor_bins[j].append(pair)
            top_pairs.append(pair)
    print(f"[INFO] Kept {len(top_pairs)} pairs "
          f"(≤{args.top_k} per image, ≥{args.min_shared} shared tracks)")

    # ---------- Score pairs (triangulation angle) ----------
    func=partial(calc_score,images=imgs,points3d=pts,
                 theta0=args.theta0,extrinsic=extr)
    score=np.zeros((N,N))
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for i,j,s in tqdm(pool.imap_unordered(func,top_pairs,
                                              chunksize=args.chunksize),
                          total=len(top_pairs)):
            score[i,j]=score[j,i]=s

    # ---------- neighbour list for each image ----------
    view_sel=[]
    for i in range(N):
        if i not in depth_ranges:
            view_sel.append([])
            continue
            
        top=np.argsort(score[i])[::-1]
        view_sel.append([(k,score[i,k]) for k in top
                         if score[i,k]>0 and k in depth_ranges][:args.top_k])

    # ---------- Write camera files ----------
    for i in range(N):
        if i not in depth_ranges:  # Skip images without valid depth ranges
            continue
            
        cam_path=os.path.join(cam_dir,f"{i:08d}_cam.txt")
        with open(cam_path,"w") as f:
            f.write("extrinsic\n")
            for r in range(4):
                f.write(" ".join(map(str,extr[i][r]))+"\n")
            
            f.write("\nintrinsic\n")
            # Look up the original COLMAP Camera entry using 0-based indexing
            img = imgs[i]
            cam = cams[img.camera_id]
            
            if cam.model == "SPHERE":
                # SPHERE format: model name + f, cx, cy
                # Focal length is garbage, estimate from image dimensions
                f.write("SPHERE\n")
                if len(cam.params) >= 3:
                    # Estimate focal length from image center assuming spherical mapping
                    _, cx, cy = cam.params[:3]  # ignore garbage focal length
                    estimated_f = max(cx, cy)  # simple heuristic based on image center
                    f.write(f"{estimated_f} {cx} {cy}\n")
                else:
                    raise ValueError(f"SPHERE camera {cam.id} has insufficient parameters: {cam.params}")
            else:
                # All other models: write full 3×3 K matrix
                K = Kdict[cam.id]
                for r in range(3):
                    f.write(" ".join(map(str, K[r])) + "\n")
            
            d0,dint,Nd,dmax=depth_ranges[i]
            f.write(f"\n{d0} {dint} {Nd} {dmax}\n")

    # ---------- Write pair.txt ----------
    with open(os.path.join(args.save_folder,"pair.txt"),"w") as f:
        valid_images = sorted(depth_ranges.keys())
        f.write(f"{len(valid_images)}\n")
        
        for idx, i in enumerate(valid_images):
            # Find neighbors for image i
            neighbors = [(j, int(score_val)) for j, score_val in view_sel[i] 
                        if j in valid_images and score_val > 0]
            
            f.write(f"{idx}\n{len(neighbors)} ")
            for j, s in neighbors:
                # Convert j to the position in valid_images list
                j_idx = valid_images.index(j)
                f.write(f"{j_idx} {s} ")
            f.write("\n")

    # ---------- Copy/convert images ----------
    for i in range(N):
        if i not in depth_ranges:  # Skip images without depth ranges
            continue
        img = imgs[i]  # Use 0-based indexing
        src = os.path.join(imgs_dir, img.name)
        dst = os.path.join(out_img, f"{i:08d}.jpg")
        if not src.lower().endswith(".jpg"):
            cv2.imwrite(dst, cv2.imread(src))
        else:
            shutil.copyfile(src, dst)

# ───────────────────────────────────────────────────────────────────────────────
# 8.  CLI
# ───────────────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    ap=argparse.ArgumentParser(
       description="Convert COLMAP sparse model to ACMMP/MVSNet format "
                   "with neighbour pruning.")
    ap.add_argument("--dense_folder",required=True,
                    help="Folder with sparse/ & images/")
    ap.add_argument("--save_folder",required=True,
                    help="Output dir")
    ap.add_argument("--model_ext",default=".txt",choices=[".txt",".bin"])
    ap.add_argument("--max_d",type=int,default=192)
    ap.add_argument("--interval_scale",type=float,default=1.0)
    ap.add_argument("--theta0",type=float,default=1.0,
                    help="Min triangulation angle (deg)")
    ap.add_argument("--top_k",type=int,default=20,
                    help="Max neighbours kept per image")
    ap.add_argument("--min_shared",type=int,default=10,
                    help="Min shared tracks to keep a pair")
    ap.add_argument("--chunksize",type=int,default=512,
                    help="mp.pool imap chunk")
    args=ap.parse_args()
    os.makedirs(args.save_folder,exist_ok=True)
    process_scene(args)