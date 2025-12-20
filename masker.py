#!/usr/bin/env python3
"""
Equirectangular Image Segmentation Masker - Production Version

Optimized TensorRT pipeline for masking objects in 360° equirectangular images.
OUTPUT: Generates standalone masks (black background) instead of overlays.
"""

import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from typing import List, Dict, Tuple
import time
from contextlib import contextmanager


class Timer:
    """Simple profiling timer"""
    
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.times: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
    
    @contextmanager
    def __call__(self, name: str):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.times[name] = self.times.get(name, 0) + elapsed
            self.counts[name] = self.counts.get(name, 0) + 1
    
    def report(self) -> str:
        if not self.times:
            return "No timing data"
        lines = ["\nTiming Report:", "-" * 50]
        total = sum(self.times.values())
        for name, t in sorted(self.times.items(), key=lambda x: -x[1]):
            pct = (t / total * 100) if total > 0 else 0
            avg = t / self.counts[name] * 1000
            lines.append(f"  {name:25s}: {t:7.2f}s ({pct:5.1f}%) avg:{avg:6.2f}ms")
        lines.append(f"  {'TOTAL':25s}: {total:7.2f}s")
        return "\n".join(lines)


class TRTSegmentation:
    """TensorRT YOLO Segmentation Engine"""
    
    FACES = ['front', 'right', 'back', 'left', 'top', 'bottom']
    
    def __init__(self, engine_path: str, conf: float = 0.25, iou: float = 0.45,
                 profile: bool = False):
        self.conf = conf
        self.iou = iou
        self.timer = Timer(profile)
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine not found: {engine_path}")
        
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        
        if not self.engine:
            raise RuntimeError(f"Failed to load: {engine_path}")
        
        self.context = self.engine.create_execution_context()
        self._setup()
        
    def _setup(self):
        """Initialize buffers and detect batch support"""
        self.max_batch = 1
        self.dynamic_batch = False
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            if shape[0] == -1:
                self.dynamic_batch = True
                self.max_batch = 6
            if self.engine.num_optimization_profiles > 0:
                try:
                    if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                        shapes = self.engine.get_tensor_profile_shape(name, 0)
                        self.max_batch = max(self.max_batch, shapes[2][0])
                except:
                    pass
        
        self.input_name = None
        self.output_names = []
        self.output_shapes = {}
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = list(self.engine.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            mode = self.engine.get_tensor_mode(name)
            
            if shape[0] == -1:
                shape[0] = self.max_batch
            
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.input_shape = tuple(shape)
                self.input_dtype = trt.nptype(dtype)
            else:
                self.output_names.append(name)
                self.output_shapes[name] = tuple(shape)
        
        self.imgsz = self.input_shape[2] if len(self.input_shape) >= 3 else 640
        
        if self.dynamic_batch:
            shape = list(self.input_shape)
            shape[0] = self.max_batch
            self.context.set_input_shape(self.input_name, shape)
            self.input_shape = tuple(shape)
            for name in self.output_names:
                self.output_shapes[name] = tuple(self.context.get_tensor_shape(name))
        
        input_size = int(np.prod(self.input_shape) * np.dtype(self.input_dtype).itemsize)
        self.d_input = cuda.mem_alloc(input_size)
        self.h_input = cuda.pagelocked_empty(self.input_shape, dtype=self.input_dtype)
        
        self.d_outputs = {}
        self.h_outputs = {}
        
        for name in self.output_names:
            shape = self.output_shapes[name]
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = int(np.prod(shape) * np.dtype(dtype).itemsize)
            self.d_outputs[name] = cuda.mem_alloc(size)
            self.h_outputs[name] = cuda.pagelocked_empty(shape, dtype=dtype)
        
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        for name in self.output_names:
            self.context.set_tensor_address(name, int(self.d_outputs[name]))
        
        self.stream = cuda.Stream()
        
        self._batch_buf = np.zeros((self.max_batch, 3, self.imgsz, self.imgsz), dtype=np.float32)
        self._pad_buf = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        self._rgb_buf = np.empty((self.imgsz, self.imgsz, 3), dtype=np.uint8)
    
    def _set_batch(self, size: int):
        if not self.dynamic_batch:
            return
        size = min(size, self.max_batch)
        shape = list(self.input_shape)
        shape[0] = size
        self.context.set_input_shape(self.input_name, shape)
        for name in self.output_names:
            self.output_shapes[name] = tuple(self.context.get_tensor_shape(name))
    
    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        h, w = img.shape[:2]
        scale = min(self.imgsz / h, self.imgsz / w)
        nh, nw = int(h * scale), int(w * scale)
        
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        
        padded = self._pad_buf.copy()
        ph, pw = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        padded[ph:ph+nh, pw:pw+nw] = resized
        
        cv2.cvtColor(padded, cv2.COLOR_BGR2RGB, dst=self._rgb_buf)
        blob = self._rgb_buf.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)
        
        return blob, {'size': (h, w), 'scale': scale, 'pad': (ph, pw)}
    
    def preprocess_batch(self, images: List[np.ndarray]) -> Tuple[np.ndarray, List[dict]]:
        batch = self._batch_buf[:len(images)]
        infos = []
        for i, img in enumerate(images):
            blob, info = self.preprocess(img)
            batch[i] = blob
            infos.append(info)
        return batch, infos
    
    def postprocess(self, outputs: dict, info: dict) -> Tuple:
        det = proto = None
        for arr in outputs.values():
            if len(arr.shape) == 4:
                proto = arr
            else:
                det = arr
        
        if det is None:
            return [], [], [], []
        
        if len(det.shape) == 3:
            det = det[0] if det.shape[0] == 1 else det
        if det.shape[0] < det.shape[1]:
            det = det.T
        
        num_coeffs = 32
        num_classes = det.shape[1] - 4 - num_coeffs
        if num_classes <= 0:
            return [], [], [], []
        
        boxes = det[:, :4]
        scores_all = det[:, 4:4+num_classes]
        coeffs = det[:, 4+num_classes:]
        
        cls = np.argmax(scores_all, axis=1)
        scores = np.max(scores_all, axis=1)
        
        mask = scores > self.conf
        boxes, scores, cls, coeffs = boxes[mask], scores[mask], cls[mask], coeffs[mask]
        
        if len(scores) == 0:
            return [], [], [], []
        
        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        
        keep = self._nms(xyxy, scores)
        xyxy, scores, cls, coeffs = xyxy[keep], scores[keep], cls[keep], coeffs[keep]
        
        masks = []
        if proto is not None and len(coeffs) > 0:
            p = proto[0] if len(proto.shape) == 4 else proto
            masks = self._gen_masks(p, coeffs, xyxy, info)
        
        ph, pw = info['pad']
        xyxy[:, [0, 2]] -= pw
        xyxy[:, [1, 3]] -= ph
        xyxy /= info['scale']
        h, w = info['size']
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h)
        
        return xyxy, scores, cls, masks
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[np.where(iou <= self.iou)[0] + 1]
        
        return np.array(keep, dtype=np.int64)
    
    def _gen_masks(self, proto: np.ndarray, coeffs: np.ndarray,
                   boxes: np.ndarray, info: dict) -> list:
        ph, pw = proto.shape[1:]
        scale = ph / self.imgsz
        
        raw = np.tensordot(coeffs, proto, axes=([1], [0]))
        raw = 1.0 / (1.0 + np.exp(-np.clip(raw, -50, 50)))
        
        boxes_p = (boxes * scale).astype(np.int32)
        boxes_p = np.clip(boxes_p, 0, [pw, ph, pw, ph])
        
        masks = []
        for m, b in zip(raw, boxes_p):
            x1, y1, x2, y2 = b
            cropped = np.zeros((ph, pw), dtype=np.float32)
            if x2 > x1 and y2 > y1:
                cropped[y1:y2, x1:x2] = m[y1:y2, x1:x2]
            
            full = cv2.resize(cropped, (self.imgsz, self.imgsz))
            
            uh = int(info['size'][0] * info['scale'])
            uw = int(info['size'][1] * info['scale'])
            pad_h, pad_w = info['pad']
            
            unpad = full[pad_h:pad_h+uh, pad_w:pad_w+uw]
            if unpad.size > 0:
                masks.append(cv2.resize(unpad, (info['size'][1], info['size'][0])))
            else:
                masks.append(np.zeros(info['size'], dtype=np.float32))
        
        return masks
    
    def infer(self, img: np.ndarray) -> Tuple:
        blob, info = self.preprocess(img)
        self.h_input[0] = blob
        
        cuda.memcpy_htod_async(self.d_input, self.h_input[:1], self.stream)
        self.context.execute_async_v3(self.stream.handle)
        for name in self.output_names:
            cuda.memcpy_dtoh_async(self.h_outputs[name], self.d_outputs[name], self.stream)
        self.stream.synchronize()
        
        outputs = {n: self.h_outputs[n].copy() for n in self.output_names}
        return self.postprocess(outputs, info)
    
    def infer_batch(self, images: List[np.ndarray]) -> List[Tuple]:
        if not self.dynamic_batch or len(images) == 1:
            return [self.infer(img) for img in images]
        
        results = []
        batch_sz = min(len(images), self.max_batch)
        self._set_batch(batch_sz)
        
        for i in range(0, len(images), batch_sz):
            batch_imgs = images[i:i+batch_sz]
            actual = len(batch_imgs)
            
            while len(batch_imgs) < batch_sz:
                batch_imgs.append(np.zeros_like(images[0]))
            
            batch, infos = self.preprocess_batch(batch_imgs)
            self.h_input[:batch_sz] = batch
            
            cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
            self.context.execute_async_v3(self.stream.handle)
            for name in self.output_names:
                cuda.memcpy_dtoh_async(self.h_outputs[name], self.d_outputs[name], self.stream)
            self.stream.synchronize()
            
            outputs = {n: self.h_outputs[n].copy() for n in self.output_names}
            
            for b in range(actual):
                batch_out = {}
                for n, arr in outputs.items():
                    if len(arr.shape) >= 3:
                        batch_out[n] = arr[b:b+1]
                results.append(self.postprocess(batch_out, infos[b]))
        
        return results
    
    def infer_faces(self, faces: Dict[str, np.ndarray]) -> Dict[str, Tuple]:
        imgs = [faces[f] for f in self.FACES if f in faces]
        names = [f for f in self.FACES if f in faces]
        results = self.infer_batch(imgs)
        return dict(zip(names, results))


class CubemapConverter:
    """Equirectangular <-> Cubemap converter"""
    
    FACES = ['front', 'right', 'back', 'left', 'top', 'bottom']
    
    def __init__(self, face_size: int = 400):
        self.size = face_size
        self._init_forward()
        self._reverse_cache = {}
    
    def _init_forward(self):
        self.fwd_maps = {}
        u = np.linspace(-1, 1, self.size, dtype=np.float32)
        v = np.linspace(-1, 1, self.size, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        
        dirs = {
            'front':  (uu, -vv, np.ones_like(uu)),
            'right':  (np.ones_like(uu), -vv, -uu),
            'back':   (-uu, -vv, -np.ones_like(uu)),
            'left':   (-np.ones_like(uu), -vv, uu),
            'top':    (uu, np.ones_like(uu), vv),
            'bottom': (uu, -np.ones_like(uu), -vv),
        }
        
        for name, (x, y, z) in dirs.items():
            norm = np.sqrt(x**2 + y**2 + z**2)
            x, y, z = x/norm, y/norm, z/norm
            theta = np.arctan2(x, z).astype(np.float32)
            phi = np.arcsin(np.clip(y, -1, 1)).astype(np.float32)
            self.fwd_maps[name] = (theta, phi)
    
    def _get_reverse(self, out_size: Tuple[int, int]) -> dict:
        if out_size in self._reverse_cache:
            return self._reverse_cache[out_size]
        
        oh, ow = out_size
        lon = np.linspace(-np.pi, np.pi, ow, dtype=np.float32)
        lat = np.linspace(np.pi/2, -np.pi/2, oh, dtype=np.float32)
        lon_g, lat_g = np.meshgrid(lon, lat)
        
        x = np.cos(lat_g) * np.sin(lon_g)
        y = np.sin(lat_g)
        z = np.cos(lat_g) * np.cos(lon_g)
        ax, ay, az = np.abs(x), np.abs(y), np.abs(z)
        
        configs = {
            'front':  ((az >= ax) & (az >= ay) & (z > 0), lambda x,y,z: (x/z, -y/z)),
            'back':   ((az >= ax) & (az >= ay) & (z < 0), lambda x,y,z: (-x/-z, -y/-z)),
            'right':  ((ax >= ay) & (ax >= az) & (x > 0), lambda x,y,z: (-z/x, -y/x)),
            'left':   ((ax >= ay) & (ax >= az) & (x < 0), lambda x,y,z: (z/-x, -y/-x)),
            'top':    ((ay >= ax) & (ay >= az) & (y > 0), lambda x,y,z: (x/y, z/y)),
            'bottom': ((ay >= ax) & (ay >= az) & (y < 0), lambda x,y,z: (x/-y, -z/-y)),
        }
        
        maps = {}
        for name, (mask, uv_fn) in configs.items():
            mx = np.zeros((oh, ow), dtype=np.float32)
            my = np.zeros((oh, ow), dtype=np.float32)
            if mask.any():
                u_v, v_v = uv_fn(x[mask], y[mask], z[mask])
                mx[mask] = (u_v + 1) * 0.5 * (self.size - 1)
                my[mask] = (v_v + 1) * 0.5 * (self.size - 1)
            maps[name] = (mx, my, mask)
        
        self._reverse_cache[out_size] = maps
        return maps
    
    def to_cubemap(self, equirect: np.ndarray) -> Dict[str, np.ndarray]:
        h, w = equirect.shape[:2]
        faces = {}
        
        for name, (theta, phi) in self.fwd_maps.items():
            cx = np.mod((theta + np.pi) / (2 * np.pi) * w, w).astype(np.float32)
            cy = np.clip((np.pi/2 - phi) / np.pi * h, 0, h-1).astype(np.float32)
            faces[name] = cv2.remap(equirect, cx, cy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        
        return faces
    
    def to_equirect(self, faces: Dict[str, np.ndarray], out_size: Tuple[int, int]) -> np.ndarray:
        oh, ow = out_size
        maps = self._get_reverse(out_size)
        
        sample = next(iter(faces.values()))
        ch = sample.shape[2] if len(sample.shape) > 2 else 1
        
        if ch > 1:
            out = np.zeros((oh, ow, ch), dtype=np.uint8)
        else:
            out = np.zeros((oh, ow), dtype=np.uint8)
        
        for name, (mx, my, mask) in maps.items():
            if name not in faces:
                continue
            face = faces[name].astype(np.uint8) if faces[name].dtype != np.uint8 else faces[name]
            sampled = cv2.remap(face, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            if ch > 1:
                for c in range(ch):
                    out[:,:,c][mask] = sampled[:,:,c][mask]
            else:
                out[mask] = sampled[mask]
        
        return out


class EquirectMasker:
    """Main pipeline for masking equirectangular images"""
    
    def __init__(self, engine: str, mask_color: Tuple[int, int, int] = (0, 0, 0),
                 conf: float = 0.15, iou: float = 0.45, batch_size: int = 16,
                 profile: bool = False):
        
        print("=" * 50)
        print("Initializing EquirectMasker")
        print("=" * 50)
        
        self.timer = Timer(profile)
        self.profile = profile
        self.color = np.array(mask_color, dtype=np.uint8)
        self.conf = conf
        
        self.model = TRTSegmentation(engine, conf=conf, iou=iou, profile=profile)
        
        max_imgs = self.model.max_batch // 6
        self.batch_size = min(batch_size, max(1, max_imgs))
        
        self._converters = {}
        
        print(f"  Engine: {engine}")
        print(f"  Max batch: {self.model.max_batch}")
        print(f"  Images/batch: {self.batch_size}")
        print(f"  Mask color: RGB{tuple(mask_color)}")
        print("=" * 50)
    
    def _converter(self, h: int) -> CubemapConverter:
        face_size = h // 2
        if face_size not in self._converters:
            self._converters[face_size] = CubemapConverter(face_size)
        return self._converters[face_size]
    
    def _apply_mask(self, img: np.ndarray, result: Tuple) -> np.ndarray:
        # CHANGED: Now initializes a black image instead of copying input
        _, _, _, masks = result
        
        # Create empty black image with same dimensions as input
        out = np.zeros_like(img)
        
        if not masks:
            return out
        
        combined = np.zeros(img.shape[:2], dtype=np.float32)
        for m in masks:
            np.maximum(combined, m, out=combined)
        
        # Apply mask color to the black image
        out[combined > self.conf] = self.color
        return out
    
    def process(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        conv = self._converter(h)
        
        with self.timer("to_cubemap"):
            faces = conv.to_cubemap(img)
        
        with self.timer("inference"):
            results = self.model.infer_faces(faces)
        
        with self.timer("apply_masks"):
            masked = {n: self._apply_mask(f, results.get(n, ([], [], [], []))) 
                      for n, f in faces.items()}
        
        with self.timer("to_equirect"):
            return conv.to_equirect(masked, (h, w))
    
    def process_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        if not images:
            return []
        
        h, w = images[0].shape[:2]
        conv = self._converter(h)
        faces_list = ['front', 'right', 'back', 'left', 'top', 'bottom']
        
        with self.timer("to_cubemap_batch"):
            all_faces = []
            all_dicts = []
            for img in images:
                faces = conv.to_cubemap(img)
                all_dicts.append(faces)
                all_faces.extend(faces[f] for f in faces_list)
        
        with self.timer("inference_batch"):
            all_results = self.model.infer_batch(all_faces)
        
        with self.timer("apply_masks_batch"):
            for i, img in enumerate(images):
                masked = {}
                for j, fname in enumerate(faces_list):
                    idx = i * 6 + j
                    masked[fname] = self._apply_mask(all_faces[idx], all_results[idx])
                all_dicts[i] = masked
        
        with self.timer("to_equirect_batch"):
            outputs = []
            for i, img in enumerate(images):
                outputs.append(conv.to_equirect(all_dicts[i], (img.shape[0], img.shape[1])))
        
        return outputs
    
    def process_folder(self, input_dir: str, output_dir: str):
        """Process folder - preserves exact filenames and extensions"""
        inp = Path(input_dir)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        files = sorted(f for f in inp.iterdir() if f.suffix.lower() in extensions)
        
        if not files:
            print(f"No images in {input_dir}")
            return 0
        
        print(f"\nProcessing {len(files)} images")
        print(f"  Input:  {input_dir}")
        print(f"  Output: {output_dir}")
        print("-" * 50)
        
        start = time.perf_counter()
        count = 0
        
        if self.batch_size > 1 and self.model.dynamic_batch:
            for i in tqdm(range(0, len(files), self.batch_size), desc="Batches"):
                batch_files = files[i:i+self.batch_size]
                
                imgs, valid = [], []
                for f in batch_files:
                    img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        imgs.append(img)
                        valid.append(f)
                
                if not imgs:
                    continue
                
                try:
                    results = self.process_batch(imgs)
                    for f, res in zip(valid, results):
                        # Keep exact same filename and extension
                        outpath = out / f.name
                        cv2.imwrite(str(outpath), res)
                    count += len(valid)
                except Exception as e:
                    print(f"\nBatch error: {e}")
        else:
            for f in tqdm(files, desc="Processing"):
                try:
                    img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                    if img is None:
                        continue
                    res = self.process(img)
                    # Keep exact same filename and extension
                    outpath = out / f.name
                    cv2.imwrite(str(outpath), res)
                    count += 1
                except Exception as e:
                    print(f"\nError {f.name}: {e}")
        
        elapsed = time.perf_counter() - start
        fps = count / elapsed if elapsed > 0 else 0
        
        print(f"\n✓ Done: {count} images in {elapsed:.1f}s ({fps:.2f} fps)")
        
        if self.profile:
            print(self.timer.report())
            print(self.model.timer.report())
        
        return count
    
    def benchmark(self, img: np.ndarray, n: int = 100) -> dict:
        print(f"\nBenchmarking ({n} iterations)...")
        
        for _ in range(5):
            self.process(img)
        
        times = []
        for _ in tqdm(range(n), desc="Benchmark"):
            t0 = time.perf_counter()
            self.process(img)
            times.append(time.perf_counter() - t0)
        
        times = np.array(times)
        r = {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'fps': 1.0 / np.mean(times),
            'p50_ms': np.percentile(times, 50) * 1000,
            'p95_ms': np.percentile(times, 95) * 1000,
        }
        
        print(f"\nResults:")
        print(f"  Mean: {r['mean_ms']:.2f}ms | FPS: {r['fps']:.2f}")
        print(f"  P50:  {r['p50_ms']:.2f}ms | P95: {r['p95_ms']:.2f}ms")
        
        return r


def main():
    import argparse
    
    p = argparse.ArgumentParser(description='Equirect Segmentation Masker')
    p.add_argument('--engine', '-e', required=True, help='TensorRT engine')
    p.add_argument('--input', '-i', required=True, help='Input path')
    p.add_argument('--output', '-o', required=True, help='Output path')
    p.add_argument('--conf', type=float, default=0.15, help='Confidence')
    p.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    p.add_argument('--color', default='0,0,0', help='Mask RGB')
    p.add_argument('--batch', type=int, default=16, help='Batch size')
    p.add_argument('--profile', action='store_true', help='Enable profiling')
    p.add_argument('--benchmark', type=int, default=0, help='Benchmark iterations')
    
    args = p.parse_args()
    color = tuple(map(int, args.color.split(',')))
    
    masker = EquirectMasker(
        engine=args.engine,
        mask_color=color,
        conf=args.conf,
        iou=args.iou,
        batch_size=args.batch,
        profile=args.profile
    )
    
    inp = Path(args.input)
    
    if inp.is_file():
        img = cv2.imread(str(inp), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Error reading {inp}")
            return
        
        if args.benchmark > 0:
            masker.benchmark(img, args.benchmark)
        else:
            res = masker.process(img)
            outp = Path(args.output)
            if outp.is_dir():
                outp = outp / inp.name
            cv2.imwrite(str(outp), res)
            print(f"✓ Saved: {outp}")
    else:
        masker.process_folder(args.input, args.output)


if __name__ == '__main__':
    main()

