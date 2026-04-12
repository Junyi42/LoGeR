# ==============================================================================
# 1. Import necessary libraries
# ==============================================================================
import numpy as np
import yaml
import os
import pickle
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
from scipy.spatial.transform import Rotation as R
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================================================
# 2. Configuration Parameters
# ==============================================================================

for seq_folder in glob.glob("data/vbr/*_train*"):
    print(f"Processing sequence folder: {seq_folder}")

    # Dataset root directory, e.g. 'data/vbr/campus_test0'
    DATASET_ROOT = Path("data/vbr/" + os.path.basename(seq_folder))

    # Output root directory
    OUTPUT_ROOT = Path("data/vbr/" + os.path.basename(seq_folder) + "_processed_aligned")

    # Camera to process ('cam_l' or 'cam_r')
    CAMERA_NAME = 'cam_l'

    # --- End of configuration ---

    # Define paths for sensor data, timestamps, and calibration files
    CALIB_FILE = DATASET_ROOT / "vbr_calib.yaml"
    # IMU data file (contains sensor readings and quaternions)
    IMU_POSE_FILE = DATASET_ROOT / "imu/data/imu.txt"
    GT_POSE_FILE = Path(glob.glob(str(DATASET_ROOT) + "/*.txt")[0])
    # #timestamp tx ty tz qx qy qz qw

    # Camera data and timestamps
    CAMERA_DIR_NAME = "camera_left" if CAMERA_NAME == 'cam_l' else "camera_right"
    IMAGE_DIR = DATASET_ROOT / CAMERA_DIR_NAME / "data"
    CAMERA_TS_FILE = DATASET_ROOT / CAMERA_DIR_NAME / "timestamps.txt"

    # LiDAR data and timestamps
    LIDAR_DIR = DATASET_ROOT / "ouster_points" / "data"
    LIDAR_TS_FILE = DATASET_ROOT / "ouster_points" / "timestamps.txt"
    LIDAR_DTYPE_FILE = LIDAR_DIR / ".dtype.pkl"

    print("--- Configuration ---")
    print(f"Dataset root: {DATASET_ROOT.resolve()}")
    print(f"IMU pose file: {IMU_POSE_FILE.resolve()}")
    print(f"Output directory: {OUTPUT_ROOT.resolve()}")
    print(f"Processing camera: {CAMERA_NAME}")
    print("--------------------")

    # ==============================================================================
    # 3. Helper Functions
    # ==============================================================================

    def load_timestamps(file_path):
        """Load timestamps from a timestamps.txt file, supporting ISO8601 (with nanoseconds) or integer nanoseconds."""
        if not file_path.exists():
            raise FileNotFoundError(f"Timestamp file not found: {file_path}")
        lines = [ln.strip() for ln in file_path.read_text().splitlines() if ln.strip()]
        if len(lines) == 0:
            return np.empty((0,), dtype=np.int64)
        sample = lines[0]
        # If ISO8601 (contains 'T'), convert to datetime64[ns] then to int64 nanoseconds
        if 'T' in sample:
            return np.array(lines, dtype='datetime64[ns]').astype('int64')
        # Otherwise read as integer
        return np.loadtxt(file_path, dtype=np.int64)


    def load_imu_poses(file_path):
        """
        Construct poses from IMU CSV (imu/data/imu.txt) and corresponding timestamp file (imu/timestamps.txt).
        - Timestamps: read ISO8601, convert to int64 nanoseconds
        - Orientation: use quaternions from CSV (columns: quat_w, quat_x, quat_y, quat_z)
        - Translation: not available in dataset, set to zero vectors
        Returns: (timestamps_ns: np.ndarray[int64], poses: List[4x4 np.ndarray])
        """
        if not file_path.exists():
            raise FileNotFoundError(f"IMU data file not found: {file_path}")
        # Corresponding IMU timestamp file path
        imu_ts_file = file_path.parent.parent / "timestamps.txt"
        if not imu_ts_file.exists():
            raise FileNotFoundError(f"IMU timestamp file not found: {imu_ts_file}")

        # Load timestamps (nanoseconds)
        timestamps = load_timestamps(imu_ts_file)

        # Load IMU CSV, skip header
        data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Align lengths (safety check)
        min_len = min(len(timestamps), data.shape[0])
        timestamps = timestamps[:min_len]
        data = data[:min_len, :]

        # Extract quaternion columns (order: w, x, y, z) and convert to scipy format [x, y, z, w]
        quat_wxyz = data[:, 6:10]
        quat_xyzw = np.stack([quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3], quat_wxyz[:, 0]], axis=1)

        poses = []
        for q in quat_xyzw:
            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = R.from_quat(q).as_matrix()
            # Translation unknown, set to zero
            poses.append(pose_matrix)

        return timestamps, poses


    def load_gt_poses(file_path):
        """
        Load poses from GT text file. Expected format per line:
        timestamp tx ty tz qx qy qz qw
        - timestamp: if < 1e12, treat as seconds and convert to nanoseconds; otherwise treat as nanoseconds
        - q: [qx, qy, qz, qw]
        Returns: (timestamps_ns: np.ndarray[int64], poses: List[4x4 np.ndarray])
        Assumes GT provides world-to-IMU (or body) poses.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"GT pose file not found: {file_path}")
        data = np.loadtxt(file_path, comments='#')
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 8:
            raise ValueError("GT pose file format incorrect, expected at least 8 columns: timestamp tx ty tz qx qy qz qw")
        ts = data[:, 0]
        # seconds -> nanoseconds (if decimal seconds)
        if np.max(ts) < 1e12:
            ts_ns = (ts * 1e9).astype(np.int64)
        else:
            ts_ns = ts.astype(np.int64)
        t = data[:, 1:4]
        q_xyzw = data[:, 4:8]  # qx, qy, qz, qw
        poses = []
        for ti, qi in zip(t, q_xyzw):
            T = np.eye(4)
            T[:3, :3] = R.from_quat(qi).as_matrix()
            T[:3, 3] = ti
            poses.append(T)
        return ts_ns, poses


    def load_lidar_point_cloud(bin_path, dtype_path):
        """
        Load LiDAR point cloud using a corrected dtype.
        This function hardcodes the corrected dtype to fix the 3-byte padding issue.
        """
        if not bin_path.exists():
            raise FileNotFoundError(f"LiDAR bin file not found: {bin_path}")

        # We define the correct dtype directly instead of loading from .dtype.pkl,
        # which is missing padding information. Total: 32 bytes per point.
        correct_dtype = np.dtype([
            ('x', '<f4'),
            ('y', '<f4'),
            ('z', '<f4'),
            ('intensity', '<f4'),
            ('t', '<u4'),
            ('reflectivity', '<u2'),
            ('ring', 'u1'),
            ('pad', 'V3'),  # 3 bytes of padding (critical fix)
            ('ambient', '<u2'),
            ('range', '<u4')
        ])

        cloud_np = np.fromfile(bin_path, dtype=correct_dtype)
        points = np.stack([cloud_np["x"], cloud_np["y"], cloud_np["z"]], axis=1)
        return points


    def project_lidar_to_depth(points, K, D, T_cam_lidar, img_shape):
        """Project LiDAR point cloud onto the camera image plane to generate a depth map,
        and return valid points in camera coordinates."""
        height, width = img_shape
        depth_map = np.zeros(img_shape, dtype=np.float32)

        points_h = np.hstack((points, np.ones((points.shape[0], 1))))
        points_cam_h = (T_cam_lidar @ points_h.T).T
        points_cam = points_cam_h[:, :3]

        front_mask = points_cam[:, 2] > 0.1
        points_cam = points_cam[front_mask]

        if points_cam.shape[0] == 0:
            return depth_map, np.empty((0, 3), dtype=np.float32)

        projected_points, _ = cv2.projectPoints(points_cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
        projected_points = projected_points.reshape(-1, 2)

        px, py = projected_points[:, 0].astype(int), projected_points[:, 1].astype(int)
        depths = points_cam[:, 2]

        in_bounds_mask = (px >= 0) & (px < width) & (py >= 0) & (py < height)

        # Filter points within image bounds
        points_cam_in_view = points_cam[in_bounds_mask]
        px_in, py_in, depths_in = px[in_bounds_mask], py[in_bounds_mask], depths[in_bounds_mask]

        # Sort and fill depth map to handle occlusions (farther points first)
        sort_idx = np.argsort(depths_in)[::-1]
        px_sorted, py_sorted, depths_sorted = px_in[sort_idx], py_in[sort_idx], depths_in[sort_idx]

        depth_map[py_sorted, px_sorted] = depths_sorted

        return depth_map, points_cam_in_view

    # ==============================================================================
    # 4. Main Processing Logic
    # ==============================================================================

    # Create output directories
    output_pose_dir = OUTPUT_ROOT / "camera_pose"
    output_rgb_dir = OUTPUT_ROOT / "rgb"
    output_depth_dir = OUTPUT_ROOT / "depthmap"
    output_pointmap_dir = OUTPUT_ROOT / "local_pointmap"
    output_pose_dir.mkdir(parents=True, exist_ok=True)
    output_rgb_dir.mkdir(parents=True, exist_ok=True)
    output_depth_dir.mkdir(parents=True, exist_ok=True)
    output_pointmap_dir.mkdir(parents=True, exist_ok=True)

    # --- 4.1 Load calibration file ---
    with open(CALIB_FILE, 'r') as f:
        calib = yaml.safe_load(f)

    # Parse calibration parameters
    intrinsics = calib[CAMERA_NAME]['intrinsics']
    K = np.array([[intrinsics[0], 0, intrinsics[2]], [0, intrinsics[1], intrinsics[3]], [0, 0, 1]])
    D = np.array(calib[CAMERA_NAME]['distortion_coeffs'])
    img_res = calib[CAMERA_NAME]['resolution']
    img_shape = (img_res[1], img_res[0])
    # Save camera intrinsics matrix to output directory
    np.savetxt(OUTPUT_ROOT / "intrinsics.txt", K)
    T_base_cam = np.array(calib[CAMERA_NAME]['T_b'])
    T_base_lidar = np.array(calib['lidar']['T_b'])
    # When using the GT pose, the pose is actually for the lidar sensor
    T_base_imu = np.array(calib['lidar']['T_b'])
    T_cam_lidar = np.linalg.inv(T_base_cam) @ T_base_lidar

    # --- 4.2 Load all data and timestamps ---
    print("Loading timestamps and GT poses...")
    cam_ts = load_timestamps(CAMERA_TS_FILE)
    lidar_ts = load_timestamps(LIDAR_TS_FILE)
    gt_ts, gt_poses = load_gt_poses(GT_POSE_FILE)

    # Get file lists
    image_files = sorted(IMAGE_DIR.glob('*'))
    lidar_files = sorted(LIDAR_DIR.glob('*.bin'))
    print(f"Found {len(cam_ts)} camera frames, {len(lidar_ts)} LiDAR frames, {len(gt_ts)} GT poses.")

    # --- 4.3 Timestamp alignment ---
    # If RGB has more frames than LiDAR or they are not synchronized,
    # set max allowed time difference (seconds) for matching; frames exceeding the threshold are skipped
    MAX_LIDAR_CAM_DIFF_SEC = float(os.environ.get("MAX_LIDAR_CAM_DIFF_SEC", 0.01))
    MAX_LIDAR_CAM_DIFF_NS = int(MAX_LIDAR_CAM_DIFF_SEC * 1e9)
    print("Performing timestamp alignment...")
    synchronized_data = []
    # Use camera frames as reference
    for i, t_cam in enumerate(tqdm(cam_ts, desc="Aligning data")):
        # Find nearest LiDAR frame
        lidar_idx = int(np.argmin(np.abs(lidar_ts - t_cam)))
        lidar_dt_ns = int(abs(int(lidar_ts[lidar_idx]) - int(t_cam)))
        # Mark as no LiDAR if exceeding threshold
        if lidar_dt_ns > MAX_LIDAR_CAM_DIFF_NS:
            lidar_idx_use = -1
        else:
            lidar_idx_use = lidar_idx

        # Find nearest GT pose
        gt_idx = int(np.argmin(np.abs(gt_ts - t_cam)))

        synchronized_data.append({
            "cam_idx": i,
            "lidar_idx": lidar_idx_use,
            "lidar_dt_ns": lidar_dt_ns,
            "imu_idx": gt_idx,
            "T_world_imu": gt_poses[gt_idx]
        })
    print(f"Successfully aligned {len(synchronized_data)} frames.")

    # --- 4.4 Process aligned data ---
    print("Processing aligned data (parallel)...")
    def _process_one(i, data_packet):
        lidar_idx_local = int(data_packet['lidar_idx'])
        if lidar_idx_local < 0:
            # Skip frames without LiDAR
            return 1  # missing count

        frame_name_local = f"{i:06d}"

        # --- a. Process RGB image ---
        src_img_path_local = image_files[data_packet['cam_idx']]
        dst_img_path_local = output_rgb_dir / f"{frame_name_local}{src_img_path_local.suffix}"

        # --- b. Compute and save camera pose (world-to-left-camera) ---
        T_world_imu_local = data_packet['T_world_imu']
        T_world_base_local = T_world_imu_local @ np.linalg.inv(T_base_imu)
        T_world_cam_local = T_world_base_local @ T_base_cam
        pose_path_local = output_pose_dir / f"{frame_name_local}.txt"

        # Write files
        np.savetxt(pose_path_local, T_world_cam_local)
        shutil.copy(src_img_path_local, dst_img_path_local)

        # --- c. Generate depth map and local point cloud from LiDAR ---
        lidar_path_local = lidar_files[lidar_idx_local]
        points_local = load_lidar_point_cloud(lidar_path_local, LIDAR_DTYPE_FILE)
        depth_map_local, points_cam_in_view_local = project_lidar_to_depth(points_local, K, D, T_cam_lidar, img_shape)

        # Save depth map and point cloud
        depth_path_local = output_depth_dir / f"{frame_name_local}.npy"
        np.save(depth_path_local, depth_map_local)
        pointmap_path_local = output_pointmap_dir / f"{frame_name_local}.npy"
        np.save(pointmap_path_local, points_cam_in_view_local)

        return 0

    missing_lidar_count = 0
    max_workers = int(os.environ.get("VBR_WORKERS", os.cpu_count() or 8))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_one, i, data_packet) for i, data_packet in enumerate(synchronized_data)]
        with tqdm(total=len(futures), desc="Processing frames", unit="frame") as pbar:
            for fut in as_completed(futures):
                missing_lidar_count += fut.result()
                pbar.update(1)


    print(f"\nProcessing complete! All files saved to: {OUTPUT_ROOT.resolve()}")
    if missing_lidar_count > 0:
        miss_pct = 100.0 * missing_lidar_count / max(1, len(synchronized_data))
        print(f"Warning: {missing_lidar_count}/{len(synchronized_data)} frames (~{miss_pct:.2f}%) had no nearby LiDAR match and were skipped. Threshold: {MAX_LIDAR_CAM_DIFF_SEC*1000:.0f} ms")
