#!/usr/bin/env python3
"""
Run LoGeR* reconstruction on a folder of images and export results.

Outputs:
  - predictions.pt          : Raw reconstruction data (points, poses, confidence, images)
  - world_points_full.ply   : Full-resolution colored world point cloud
  - world_points_2x.ply     : 2x subsampled colored world point cloud
  - camera_poses.ply        : Camera positions as a colored point cloud (red=start, blue=end)
  - scene.mvs               : OpenMVS interface file (camera poses + image list)

Usage:
  python run_loger.py --input /path/to/images --output /path/to/output [--fx 1500 --fy 1500] [options]

If --fx/--fy are omitted, intrinsics are auto-recovered from the predicted pointmap
using the pinhole camera model (similar to DUSt3R's approach).
"""

import os
import sys
import glob
import math
import time
import argparse
import inspect

import yaml
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from natsort import natsorted
from plyfile import PlyData, PlyElement

# LoGeR imports (run from repo root or ensure loger is on PYTHONPATH)
from loger.models.pi3 import Pi3
from loger.utils.geometry import depth_edge


def parse_args():
    parser = argparse.ArgumentParser(description="Run LoGeR* reconstruction and export PLY + MVS files")
    parser.add_argument("--input", type=str, required=True, help="Folder of input images")
    parser.add_argument("--output", type=str, required=True, help="Output folder for results")
    parser.add_argument("--ckpt", type=str, default="ckpts/LoGeR_star/latest.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="ckpts/LoGeR_star/original_config.yaml",
                        help="Path to model config YAML")
    parser.add_argument("--window_size", type=int, default=32,
                        help="Window size for sliding-window inference")
    parser.add_argument("--overlap_size", type=int, default=3,
                        help="Overlap between adjacent windows")
    parser.add_argument("--start_frame", type=int, default=0, help="First frame index")
    parser.add_argument("--end_frame", type=int, default=-1, help="Last frame index (-1 = all)")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument("--conf_percentile", type=float, default=20.0,
                        help="Remove points below this confidence percentile (0-100)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda if available)")
    # Camera intrinsics for OpenMVS export (optional — auto-estimated if omitted)
    parser.add_argument("--fx", type=float, default=None,
                        help="Focal length X in pixels (for original image resolution). Auto-estimated if omitted.")
    parser.add_argument("--fy", type=float, default=None,
                        help="Focal length Y in pixels (for original image resolution). Auto-estimated if omitted.")
    parser.add_argument("--cx", type=float, default=None,
                        help="Principal point X in pixels (default: image_width / 2)")
    parser.add_argument("--cy", type=float, default=None,
                        help="Principal point Y in pixels (default: image_height / 2)")
    parser.add_argument("--per-frame-intrinsics", action="store_true",
                        help="Estimate focal length per frame instead of shared (for zoom/multi-camera)")
    return parser.parse_args()


def find_images(folder):
    """Find images in folder, case-insensitive for jpg/jpeg/png."""
    exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    seen_real = set()
    paths = []
    for ext in exts:
        for p in glob.glob(os.path.join(folder, ext)):
            real = os.path.realpath(p)
            if real not in seen_real:
                seen_real.add(real)
                paths.append(p)
    return natsorted(paths)


def load_model(ckpt_path, config_path):
    """Load Pi3 model with config and checkpoint."""
    model_kwargs = {}
    if config_path:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        model_config = config.get("model", {})
        pi3_sig = inspect.signature(Pi3.__init__)
        valid_kwargs = {
            name for name, param in pi3_sig.parameters.items()
            if name not in {"self", "args", "kwargs"}
            and param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        for key in sorted(valid_kwargs):
            if key in model_config:
                model_kwargs[key] = model_config[key]
        print(f"Model config: {model_kwargs}")

    model = Pi3(**model_kwargs)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # Strip DDP prefix if present
    state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    print("Model loaded successfully.")
    return model, config


def load_images(image_paths, pixel_limit=255000):
    """Load and resize images to model-compatible resolution (multiple of 14)."""
    sources = []
    for p in image_paths:
        try:
            sources.append(Image.open(p).convert("RGB"))
        except Exception as e:
            print(f"Warning: could not load {p}: {e}")

    if not sources:
        return torch.empty(0)

    W_orig, H_orig = sources[0].size
    scale = math.sqrt(pixel_limit / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_t, H_t = W_orig * scale, H_orig * scale
    k, m = round(W_t / 14), round(H_t / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > W_t / H_t:
            k -= 1
        else:
            m -= 1
    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    print(f"Resizing {len(sources)} images to {TARGET_W}x{TARGET_H}")

    to_tensor = transforms.ToTensor()
    tensors = []
    for img in sources:
        resized = img.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
        tensors.append(to_tensor(resized))
    return torch.stack(tensors, dim=0)


def build_forward_kwargs(config, args):
    """Build forward pass kwargs from config and CLI args."""
    training = config.get("training_settings", {})
    model_cfg = config.get("model", {})
    se3_from_config = model_cfg.get("se3", config.get("se3", False))
    return {
        "window_size": args.window_size or training.get("window_size", -1),
        "overlap_size": args.overlap_size or training.get("overlap_size", 0),
        "reset_every": training.get("reset_every", 0),
        "num_iterations": config.get("num_iterations", 1),
        "sim3": False,
        "sim3_scale_mode": "median",
        "se3": bool(se3_from_config) or True,  # LoGeR* uses SE(3)
        "turn_off_ttt": False,
        "turn_off_swa": False,
    }


def _weighted_median(values, w, max_n=50000):
    """Confidence-weighted median with optional subsampling for speed."""
    if len(values) == 0:
        return None
    if len(values) > max_n:
        idx = np.random.default_rng(42).choice(len(values), max_n, replace=False)
        values, w = values[idx], w[idx]
    sorted_idx = np.argsort(values)
    values, w = values[sorted_idx], w[sorted_idx]
    cumw = np.cumsum(w)
    return float(values[np.searchsorted(cumw, cumw[-1] / 2.0)])


def _focal_from_frame(local_pts, conf_frame, uu, vv, H, W):
    """Estimate (fx, fy) for a single frame from local points and pixel grid."""
    X, Y, Z = local_pts[..., 0], local_pts[..., 1], local_pts[..., 2]
    valid = (Z > 1e-3) & (np.abs(X) > 1e-6) & (np.abs(Y) > 1e-6) & (conf_frame > 0.1)
    with np.errstate(divide="ignore", invalid="ignore"):
        fx_pp = uu * Z / X
        fy_pp = vv * Z / Y
    fx_v, fy_v, w = fx_pp[valid], fy_pp[valid], conf_frame[valid]
    ok_fx = (fx_v > W * 0.1) & (fx_v < W * 10)
    ok_fy = (fy_v > H * 0.1) & (fy_v < H * 10)
    fx = _weighted_median(fx_v[ok_fx], w[ok_fx])
    fy = _weighted_median(fy_v[ok_fy], w[ok_fy])
    return fx, fy


def _snap_square_pixels(fx, fy, tol=0.02):
    """If fx and fy are within tol of each other, average them (square pixels)."""
    if fx is None or fy is None:
        return fx, fy
    rel_diff = abs(fx - fy) / max(fx, fy)
    if rel_diff < tol:
        avg = (fx + fy) / 2.0
        return avg, avg
    return fx, fy


def estimate_focal_lengths(local_points, conf, shared=True):
    """Estimate fx, fy from predicted local 3D points using the pinhole model.

    For a pinhole camera with principal point at image center:
        u - cx = fx * (X / Z)
        v - cy = fy * (Y / Z)

    Args:
        local_points: (T, H, W, 3) local camera-space 3D points.
        conf: (T, H, W) or (T, H, W, 1) confidence per pixel.
        shared: If True, estimate a single (fx, fy) from all frames pooled
                together (shared camera). If False, estimate per-frame.

    Returns:
        If shared: (fx, fy) tuple at model resolution.
        If per-frame: (fx_array, fy_array) each of length T, at model resolution.
    """
    T, H, W, _ = local_points.shape
    if conf.ndim == 4:
        conf = conf.squeeze(-1)

    # Build centered pixel grid (shared across frames)
    u_centered = np.arange(W, dtype=np.float32) - (W - 1) / 2.0
    v_centered = np.arange(H, dtype=np.float32) - (H - 1) / 2.0
    uu, vv = np.meshgrid(u_centered, v_centered)  # (H, W)

    fallback = max(W, H) * 1.2

    if shared:
        # Pool all frames together for a single robust estimate
        uu_all = np.broadcast_to(uu[None], (T, H, W))
        vv_all = np.broadcast_to(vv[None], (T, H, W))
        fx, fy = _focal_from_frame(local_points, conf, uu_all, vv_all, H, W)
        if fx is None or fy is None:
            print("  Warning: estimation failed, using fallback")
            fx, fy = fallback, fallback
        fx, fy = _snap_square_pixels(fx, fy)
        return fx, fy
    else:
        # Per-frame estimation
        fx_arr, fy_arr = np.empty(T), np.empty(T)
        for t in range(T):
            fx_t, fy_t = _focal_from_frame(
                local_points[t], conf[t], uu, vv, H, W)
            if fx_t is None or fy_t is None:
                fx_t, fy_t = fallback, fallback
            fx_t, fy_t = _snap_square_pixels(fx_t, fy_t)
            fx_arr[t], fy_arr[t] = fx_t, fy_t
        return fx_arr, fy_arr


def save_points_ply(path, points, colors, conf, conf_percentile, subsample=1):
    """Save a colored point cloud to binary PLY with confidence filtering.

    Args:
        path: Output PLY file path.
        points: (N, 3) float array of XYZ positions.
        colors: (N, 3) float array of RGB in [0, 1].
        conf: (N,) float array of per-point confidence.
        conf_percentile: Remove points below this percentile (0-100).
        subsample: Keep every Nth point after filtering.
    """
    # Confidence filter
    if conf_percentile > 0:
        threshold = np.percentile(conf, conf_percentile)
        mask = conf > threshold
        points = points[mask]
        colors = colors[mask]

    # Subsample
    if subsample > 1:
        points = points[::subsample]
        colors = colors[::subsample]

    colors_u8 = np.clip(colors * 255, 0, 255).astype(np.uint8)

    vertex = np.empty(len(points), dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]
    vertex["red"] = colors_u8[:, 0]
    vertex["green"] = colors_u8[:, 1]
    vertex["blue"] = colors_u8[:, 2]

    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)
    print(f"  Saved {len(points):,} points -> {path}")


def save_camera_ply(path, camera_poses):
    """Save camera positions as a colored point cloud (red=start -> blue=end).

    Args:
        path: Output PLY file path.
        camera_poses: (T, 4, 4) array of camera-to-world transforms.
    """
    positions = camera_poses[:, :3, 3]  # (T, 3) camera centers
    T = len(positions)

    # Color gradient: red (frame 0) -> blue (last frame)
    t = np.linspace(0, 1, T).astype(np.float32)
    colors = np.zeros((T, 3), dtype=np.uint8)
    colors[:, 0] = (255 * (1 - t)).astype(np.uint8)  # R
    colors[:, 2] = (255 * t).astype(np.uint8)         # B

    vertex = np.empty(T, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertex["x"] = positions[:, 0]
    vertex["y"] = positions[:, 1]
    vertex["z"] = positions[:, 2]
    vertex["red"] = colors[:, 0]
    vertex["green"] = colors[:, 1]
    vertex["blue"] = colors[:, 2]

    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)
    print(f"  Saved {T} camera poses -> {path}")


def save_mvs(path, camera_poses, image_paths, output_folder, fx, fy, cx, cy, img_width, img_height):
    """Export scene as OpenMVS .mvs interface file.

    Args:
        path: Output .mvs file path.
        camera_poses: (T, 4, 4) numpy array, camera-to-world transforms from LoGeR.
        image_paths: List of absolute paths to the original input images.
        output_folder: Folder where the .mvs file is saved (for computing relative paths).
        fx, fy: Focal lengths in pixels (original image resolution).
        cx, cy: Principal point in pixels (original image resolution).
        img_width, img_height: Original image dimensions.
    """
    # Import saveMVSInterface from OpenMVS scripts
    openmvs_scripts = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "openMVS", "scripts", "python")
    openmvs_scripts = os.path.normpath(openmvs_scripts)
    if not os.path.isdir(openmvs_scripts):
        print(f"Warning: OpenMVS scripts not found at {openmvs_scripts}, skipping MVS export.")
        return
    sys.path.insert(0, openmvs_scripts)
    from MvsUtils import saveMVSInterface

    T = len(camera_poses)

    # Build K matrix (3x3 intrinsics)
    K = [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]]

    # Camera has identity pose relative to the platform (no offset)
    R_cam = np.eye(3).tolist()
    C_cam = [0.0, 0.0, 0.0]

    # Convert LoGeR camera-to-world poses to OpenMVS world-to-camera convention
    # LoGeR: Twc = [Rwc | twc; 0 0 0 1] (camera-to-world)
    # OpenMVS: Pose.R = world-to-camera rotation, Pose.C = camera center in world
    #   R_mvs = Rwc^T,  C_mvs = twc (column 3 of Twc)
    poses = []
    for i in range(T):
        Twc = camera_poses[i]  # (4, 4)
        Rwc = Twc[:3, :3]
        twc = Twc[:3, 3]
        R_mvs = Rwc.T  # world-to-camera rotation
        C_mvs = twc     # camera center in world coords
        poses.append({
            "R": R_mvs.tolist(),
            "C": C_mvs.tolist(),
        })

    # Build image list with relative paths from the MVS file location
    images = []
    for i, img_path in enumerate(image_paths):
        rel_path = os.path.relpath(os.path.abspath(img_path), os.path.abspath(output_folder))
        images.append({
            "name": rel_path,
            "mask_name": "",          # version > 4: required by reader
            "platform_id": 0,
            "camera_id": 0,
            "pose_id": i,
            "id": i,                  # version > 2: required by reader
            "min_depth": 0.0,         # version > 6: required by reader
            "avg_depth": 0.0,
            "max_depth": 0.0,
            "view_scores": [],
        })

    scene = {
        "stream_version": 7,
        "platforms": [{
            "name": "platform_0",
            "cameras": [{
                "name": "camera_0",
                "band_name": "RGB",
                "width": img_width,
                "height": img_height,
                "K": K,
                "R": R_cam,
                "C": C_cam,
            }],
            "poses": poses,
        }],
        "images": images,
        "vertices": [],
        "vertices_normal": [],
        "vertices_color": [],
        "lines": [],               # version > 0: required by reader
        "lines_normal": [],
        "lines_color": [],
        "transform": np.eye(4).tolist(),  # version > 1: required by reader
        "obb": {                          # version > 5: required by reader
            "rot": np.eye(3).tolist(),
            "pt_min": [0.0, 0.0, 0.0],
            "pt_max": [0.0, 0.0, 0.0],
        },
    }

    saveMVSInterface(scene, path)
    print(f"  Saved {T} cameras -> {path}")


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Validate paths
    if not os.path.isdir(args.input):
        sys.exit(f"Error: input folder not found: {args.input}")
    if not os.path.isfile(args.ckpt):
        sys.exit(f"Error: checkpoint not found: {args.ckpt}")
    if not os.path.isfile(args.config):
        sys.exit(f"Error: config not found: {args.config}")

    os.makedirs(args.output, exist_ok=True)

    # --- Find and load images ---
    image_paths = find_images(args.input)
    end_idx = args.end_frame if args.end_frame != -1 else None
    image_paths = image_paths[args.start_frame:end_idx:args.stride]
    if not image_paths:
        sys.exit("Error: no images found in input folder.")
    print(f"Found {len(image_paths)} images.")

    images_tensor = load_images(image_paths)
    if images_tensor.numel() == 0:
        sys.exit("Error: failed to load any images.")
    images_tensor = images_tensor.to(device)

    # --- Load model ---
    model, config = load_model(args.ckpt, args.config)
    model = model.eval().to(device)

    # --- Run inference ---
    forward_kwargs = build_forward_kwargs(config, args)
    print(f"Forward kwargs: {forward_kwargs}")

    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    num_frames = images_tensor.shape[0]

    print(f"Running inference on {num_frames} frames...")
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad(), torch.amp.autocast(device, enabled=(device == "cuda"), dtype=dtype):
        preds = model(images_tensor[None], **forward_kwargs)

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"Inference: {elapsed:.1f}s ({num_frames / elapsed:.1f} FPS, {elapsed / num_frames * 1000:.1f} ms/frame)")

    # --- Post-process ---
    preds["images"] = images_tensor[None].permute(0, 1, 3, 4, 2)  # B,S,H,W,C
    preds["conf"] = torch.sigmoid(preds["conf"])

    # Extract local_points before deletion (needed for intrinsics estimation)
    local_points_np = None
    if "local_points" in preds and preds["local_points"] is not None:
        local_points_np = preds["local_points"].squeeze(0).cpu().float().numpy()
        del preds["local_points"]

    # Convert to numpy, remove batch dim
    results = {
        k: v.squeeze(0).cpu().float().numpy()
        for k, v in preds.items()
        if v is not None and torch.is_tensor(v)
    }

    # --- Save .pt ---
    pt_path = os.path.join(args.output, "predictions.pt")
    print(f"Saving {pt_path}...")
    torch.save({k: torch.from_numpy(v) for k, v in results.items()}, pt_path)

    # --- Prepare flat arrays for PLY export ---
    points = results["points"]       # (T, H, W, 3)
    colors = results["images"]       # (T, H, W, 3)
    conf = results["conf"]           # (T, H, W, 1)
    camera_poses = results["camera_poses"]  # (T, 4, 4)

    points_flat = points.reshape(-1, 3)
    colors_flat = colors.reshape(-1, 3)
    conf_flat = conf.reshape(-1)

    print(f"Total points: {len(points_flat):,}")

    # --- Export PLY files ---
    print("Exporting PLY files...")

    save_points_ply(
        os.path.join(args.output, "world_points_full.ply"),
        points_flat, colors_flat, conf_flat,
        conf_percentile=args.conf_percentile,
        subsample=1,
    )

    save_points_ply(
        os.path.join(args.output, "world_points_2x.ply"),
        points_flat, colors_flat, conf_flat,
        conf_percentile=args.conf_percentile,
        subsample=2,
    )

    save_camera_ply(
        os.path.join(args.output, "camera_poses.ply"),
        camera_poses,
    )

    # --- Resolve camera intrinsics ---
    # Get original image resolution (before LoGeR's internal resize)
    first_img = Image.open(image_paths[0])
    img_width, img_height = first_img.size
    first_img.close()

    # Model's internal resolution (from the tensor)
    _, model_H, model_W, _ = results["points"].shape

    shared = not args.per_frame_intrinsics
    scale_x = img_width / model_W
    scale_y = img_height / model_H
    cx = args.cx if args.cx is not None else img_width / 2.0
    cy = args.cy if args.cy is not None else img_height / 2.0

    if args.fx is not None:
        fx = args.fx
        fy = args.fy if args.fy is not None else fx
        fx, fy = _snap_square_pixels(fx, fy)
        print(f"Using provided intrinsics: fx={fx:.1f}, fy={fy:.1f}")
    elif local_points_np is not None:
        conf_for_est = results["conf"]
        mode = "shared" if shared else "per-frame"
        print(f"Estimating intrinsics from predicted pointmap ({mode})...")
        fx_model, fy_model = estimate_focal_lengths(
            local_points_np, conf_for_est, shared=shared)
        if shared:
            fx = fx_model * scale_x
            fy = fy_model * scale_y
            fx, fy = _snap_square_pixels(fx, fy)
            print(f"  Model res: {model_W}x{model_H} -> fx_model={fx_model:.1f}, fy_model={fy_model:.1f}")
            print(f"  Original res: {img_width}x{img_height} -> fx={fx:.1f}, fy={fy:.1f}")
        else:
            fx = fx_model * scale_x  # array of length T
            fy = fy_model * scale_y
            # Snap each pair
            for i in range(len(fx)):
                fx[i], fy[i] = _snap_square_pixels(fx[i], fy[i])
            med_fx, med_fy = float(np.median(fx)), float(np.median(fy))
            print(f"  Per-frame median: fx={med_fx:.1f}, fy={med_fy:.1f}")
            print(f"  Range: fx=[{fx.min():.1f}, {fx.max():.1f}], fy=[{fy.min():.1f}, {fy.max():.1f}]")
    else:
        fx = max(img_width, img_height) * 1.2
        fy = fx
        print(f"Warning: no local_points available, using fallback fx=fy={fx:.1f}")

    # --- Export OpenMVS .mvs file ---
    # For per-frame intrinsics, use median for the single-camera MVS export
    # (OpenMVS platform model uses one K per camera, not per frame)
    fx_mvs = float(np.median(fx)) if isinstance(fx, np.ndarray) else fx
    fy_mvs = float(np.median(fy)) if isinstance(fy, np.ndarray) else fy
    print(f"Exporting OpenMVS scene (fx={fx_mvs:.1f}, fy={fy_mvs:.1f})...")
    save_mvs(
        os.path.join(args.output, "scene.mvs"),
        camera_poses, image_paths, args.output,
        fx_mvs, fy_mvs, cx, cy, img_width, img_height,
    )

    print("Done!")


if __name__ == "__main__":
    main()
