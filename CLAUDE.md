# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LoGeR (Long-Context Geometric Reconstruction with Hybrid Memory) processes long video streams in overlapping chunks using a hybrid memory design (TTT fast-weights + cross-attention) to produce dense 3D point clouds and camera poses. Built on top of Pi3 and LaCT.

## Environment Setup

```bash
conda create -n loger python=3.11 cmake=3.14.0
conda activate loger
pip install -r requirements.txt
```

Checkpoints go in `ckpts/LoGeR/latest.pt` and `ckpts/LoGeR_star/latest.pt` (downloaded from HuggingFace).

## Common Commands

### Demo / Inference
```bash
# Run both checkpoints on example data
bash demo_run.sh [CUDA_DEVICE] [INPUT_PATH]

# Single model run
python demo_viser.py --input data/examples/office --config ckpts/LoGeR/original_config.yaml \
    --model_name ckpts/LoGeR/latest.pt --window_size 32 --overlap_size 3
```

### Evaluation — Long Sequences (KITTI / VBR)
```bash
# Single sequence
bash eval/demo_run_longeval.sh --cuda 0 --model LoGeR --mode kitti --seq 00 --win 32

# All sequences
bash eval/run_kitti.sh
bash eval/run_vbr.sh
```

### Evaluation — Short Sequences (Relative Pose / Reconstruction)
```bash
# Relative pose on ScanNet (multi-GPU)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash eval/relpose/run_scannet.sh LoGeR \
    --num-processes 8 --port 29122 --window-size 64 --overlap-size 3

# Multi-view reconstruction on 7scenes
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash eval/mv_recon/run.sh LoGeR \
    --num-processes 8 --port 29552 --window-size 64 --overlap-size 3
```

### Benchmark Metrics (C++ tools)
```bash
cd eval/long_eval_script
g++ -o vbr_benchmark vbr_benchmark.cpp -I /usr/include/eigen3 -O3 -std=c++17
g++ -o kitti_benchmark kitti_benchmark.cpp -I /usr/include/eigen3 -O3 -std=c++17
./kitti_benchmark ../../data/kitti/dataset/poses ../../results/viser_pi3_kitti --plot
./vbr_benchmark ../../data/vbr/processed_gt ../../results/viser_pi3_vbr --plot
```

## Architecture

### Data Flow
```
Input images (T, H, W, 3)
  → DINOv2 ViT-L/14 encoder → (T, N_patches, 1024)
  → 36-layer decoder (BlockRope + TTT + cross-attn) → (T, N_patches, 2048)
  → Points head (ConvHead) → local 3D points (T, H, W, 3)
  → Camera head → SE(3) poses (T, 4, 4)
  → World transform → world points + confidence maps
```

### Core Files

| File | Purpose |
|------|---------|
| `loger/models/pi3.py` | Main Pi3 model: encoder, 36-layer decoder, output heads. `forward()` handles windowed inference with overlap stitching |
| `loger/models/pi3x.py` | Pi3X variant with multimodal (RGB + depth) encoding |
| `loger/models/ttt.py` | Test-Time Training fast-weight operators (`FastWeightGluMLPMultihead`). Inserted at even decoder layers. Updates K,V in-place using learned per-head LRs and optional Muon optimizer |
| `loger/models/layers/` | Transformer building blocks: `attention.py` (FlashAttentionRope), `block.py` (BlockRope), `transformer_head.py` (decoder + output heads), `camera_head.py` (pose regression), `pos_embed.py` (RoPE2D) |
| `demo_viser.py` | Main inference entry point. Loads model, processes video/images, exports trajectories, launches Viser 3D viewer |
| `eval/pi3_adapter.py` | Wraps Pi3/Pi3X for evaluation. Defines `Pi3SequenceOutput` dataclass (local_points, world_points, camera_poses, confidence, colors) |

### Key Concepts

- **Hybrid Memory**: Two complementary mechanisms — (1) TTT fast-weight layers at even decoder layers (0,2,...,34) accumulate implicit memory by updating weights during inference, (2) cross-attention modules at layers 10,18,26,34 provide explicit memory retrieval from register tokens.
- **Windowed Inference**: Long sequences are processed in overlapping chunks (`window_size` frames, `overlap_size` overlap). TTT K,V history is detached between windows to prevent unbounded memory growth.
- **LoGeR vs LoGeR\***: LoGeR uses `ttt_pre_norm=true`; LoGeR\* uses `se3=true` for SE(3)-constrained pose estimation. LoGeR\* evaluation scripts add the `--se3` flag.
- **Register Tokens**: 5 learnable tokens prepended to each frame's patch sequence, used for memory aggregation across the decoder.

### Evaluation Structure

- `eval/relpose/` — Relative pose estimation (ScanNet, TUM-dynamics). Uses `accelerate` for multi-GPU.
- `eval/mv_recon/` — Multi-view 3D reconstruction (7scenes). Mesh quality metrics.
- `eval/video_depth/` — Dense depth evaluation.
- `eval/long_eval_script/` — C++ benchmarks for KITTI/VBR trajectory evaluation (ATE, RPE metrics).
- `eval/datasets_preprocess/` — Dataset conversion scripts (`long_prepare_*.py` for ScanNet, TUM, Bonn).

### Trajectory Format

Output trajectories use TUM format: `timestamp tx ty tz qx qy qz qw`. KITTI GT uses 3x4 matrix format. The C++ benchmark tools handle conversion between formats.

## Model Configuration

Configs are YAML files in `ckpts/*/original_config.yaml`. Key parameters:
- `ttt_insert_after`: Which decoder layers get TTT operators (default: all even 0-34)
- `attn_insert_after`: Which layers get cross-attention (default: 10,18,26,34)
- `ttt_head_dim`: Fast-weight head dimension (512)
- `ttt_inter_multi`: Intermediate dimension multiplier (4)
- `ttt_pre_norm` / `se3`: Distinguishes LoGeR from LoGeR*

## Data Layout

- `data/examples/` — Demo data (e.g., `office/`)
- `data/scannet/`, `data/kitti/`, `data/vbr/` — Evaluation datasets (see `eval/eval.md` for download)
- `ckpts/LoGeR/` and `ckpts/LoGeR_star/` — Model weights + config
- `results/` — Evaluation output directory
