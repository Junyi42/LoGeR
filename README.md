# LoGeR: Long-Context Geometric Reconstruction with Hybrid Memory

> **Notice: This is a reimplementation of LoGeR; complete code and models will be released upon approval.**

LoGeR processes long video streams in chunks with a hybrid memory design to improve large-scale geometric reconstruction quality and consistency.

[**LoGeR: Long-Context Geometric Reconstruction with Hybrid Memory**](https://arxiv.org/abs/2603.03269) [*Junyi Zhang*](https://junyi42.github.io/), [*Charles Herrmann*](https://scholar.google.com/citations?user=LQvi5XAAAAAJ), [*Junhwa Hur*](https://hurjunhwa.github.io/), [*Chen Sun*](https://chensun.me/index.html), [*Ming-Hsuan Yang*](https://faculty.ucmerced.edu/mhyang/), [*Forrester Cole*](https://scholar.google.com/citations?user=xZRRr-IAAAAJ&hl), [*Trevor Darrell*](https://people.eecs.berkeley.edu/~trevor/), [*Deqing Sun*](https://deqings.github.io/)
| [**[Project Webpage]**](https://LoGeR-project.github.io/) | [**[arXiv]**](https://arxiv.org/abs/2603.03269)

<p align="center">
  <img src="https://loger-project.github.io/figs/fig1_teaser.png" alt="LoGeR Teaser" width="100%">
</p>

## Installation

```bash
git clone https://github.com/junyi42/LoGeR
cd LoGeR
conda create -n loger python=3.11 cmake=3.14.0
conda activate loger
pip install -r requirements.txt
```

## Checkpoint Download

Checkpoints are hosted on [Hugging Face](https://huggingface.co/Junyi42/LoGeR):


Please place files as:
- `ckpts/LoGeR/latest.pt`
- `ckpts/LoGeR_star/latest.pt`

Example commands:

```bash
wget -O ckpts/LoGeR/latest.pt "https://huggingface.co/Junyi42/LoGeR/resolve/main/LoGeR/latest.pt?download=true"
wget -O ckpts/LoGeR_star/latest.pt "https://huggingface.co/Junyi42/LoGeR/resolve/main/LoGeR_star/latest.pt?download=true"
```

## Demo

For demo usage, please directly refer to:

- [`demo_run.sh`](demo_run.sh)

## Run Reconstruction + Export

[`run_loger.py`](run_loger.py) runs LoGeR* on a folder of images and exports PLY point clouds and an [OpenMVS](https://github.com/cdcseacave/openMVS) `.mvs` scene file.

**Outputs:**
- `predictions.pt` — raw reconstruction tensors (points, poses, confidence, images)
- `world_points_full.ply` — full-resolution colored world point cloud (confidence-filtered)
- `world_points_2x.ply` — 2x subsampled colored world point cloud
- `camera_poses.ply` — camera positions as colored point cloud (red→blue = temporal order)
- `scene.mvs` — OpenMVS interface file with camera poses and image list

**Basic usage (auto-estimated intrinsics):**

```bash
python run_loger.py \
    --input /path/to/images \
    --output /path/to/output
```

**With known intrinsics:**

```bash
python run_loger.py \
    --input /path/to/images \
    --output /path/to/output \
    --fx 1500 --fy 1500
```

**Full options:**

```bash
python run_loger.py \
    --input /path/to/images \
    --output /path/to/output \
    --fx 1500 --fy 1500 \
    --cx 1200 --cy 800 \
    --ckpt ckpts/LoGeR_star/latest.pt \
    --config ckpts/LoGeR_star/original_config.yaml \
    --window_size 32 \
    --overlap_size 3 \
    --start_frame 0 \
    --end_frame -1 \
    --stride 1 \
    --conf_percentile 20
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input` | Yes | — | Folder of input images (jpg/jpeg/png, case-insensitive) |
| `--output` | Yes | — | Output folder for all results |
| `--fx` | No | auto | Focal length X in pixels (original image resolution). Auto-estimated from pointmap if omitted. |
| `--fy` | No | =fx | Focal length Y in pixels (original image resolution) |
| `--cx` | No | width/2 | Principal point X in pixels |
| `--cy` | No | height/2 | Principal point Y in pixels |
| `--ckpt` | No | `ckpts/LoGeR_star/latest.pt` | Model checkpoint path |
| `--config` | No | `ckpts/LoGeR_star/original_config.yaml` | Model config YAML |
| `--window_size` | No | 32 | Sliding window size (frames per chunk) |
| `--overlap_size` | No | 3 | Overlap between adjacent windows |
| `--start_frame` | No | 0 | First frame index |
| `--end_frame` | No | -1 (all) | Last frame index |
| `--stride` | No | 1 | Frame stride |
| `--conf_percentile` | No | 20 | Remove points below this confidence percentile |
| `--per-frame-intrinsics` | No | off | Estimate focal length per frame instead of shared (for zoom/multi-camera) |

**Notes:**
- If provided, `--fx`/`--fy` must correspond to the **original** image resolution. LoGeR internally resizes images but the MVS export uses original dimensions.
- If `--fx`/`--fy` are omitted, focal lengths are auto-recovered from the predicted 3D pointmap using the pinhole camera model (confidence-weighted median, similar to DUSt3R's approach). By default all frames are pooled together (shared camera assumption). Use `--per-frame-intrinsics` for varying zoom or multi-camera setups.
- If the estimated fx and fy are within 2% of each other, they are averaged to enforce square pixels.
- The `scene.mvs` file stores relative paths to the original images — no image copying is performed.
- To visualize the raw `.pt` output interactively: `python demo_viser.py --load /path/to/predictions.pt`

## Evaluation

For evaluation instructions, please refer to:

- [`eval/eval.md`](eval/eval.md)

## Citation

If you find our work useful, please cite:

```bibtex
@article{zhang2026loger,
  title={LoGeR: Long-Context Geometric Reconstruction with Hybrid Memory},
  author={Zhang, Junyi and Herrmann, Charles and Hur, Junhwa and Sun, Chen and Yang, Ming-Hsuan and Cole, Forrester and Darrell, Trevor and Sun, Deqing},
  journal={arXiv preprint arXiv:2603.03269},
  year={2026}
}
```

## Acknowledgments
Our code is based on [Pi3](https://github.com/yyfz/Pi3) and [LaCT](https://github.com/a1600012888/LaCT), our camera pose estimation evaluation script is based on [TTT3R](https://github.com/Inception3D/TTT3R) & [VBR](https://github.com/rvp-group/vbr-slam-benchmark), and our visualization code is based on [Viser](https://github.com/nerfstudio-project/viser). We thank the authors for their excellent work!