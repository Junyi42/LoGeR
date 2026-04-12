# Evaluation

## Datasets
Please follow [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md) and [Spann3R](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md) to download **ScanNet**, **TUM-dynamics**, **7scenes** and **Bonn** datasets.
Please download and extract all datasets under `data/` (for example, `data/scannet`, `data/kitti`).

### ScanNet
To prepare the **ScanNet** dataset, execute:
```bash
python eval/datasets_preprocess/long_prepare_scannet.py # Run from repository root; you may need to change dataset paths
```

### TUM-dynamics
To prepare the **TUM-dynamics** dataset, execute:
```bash
python eval/datasets_preprocess/long_prepare_tum.py # Run from repository root; you may need to change dataset paths
```

### Bonn
To prepare the **Bonn** dataset, execute:
```bash
python eval/datasets_preprocess/long_prepare_bonn.py # Run from repository root; you may need to change dataset paths
```

### KITTI
To download and extract **KITTI** odometry files under `data/kitti`, execute:
```bash
mkdir -p data/kitti # Run from repository root
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip -P data/kitti
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip -P data/kitti
unzip -o data/kitti/data_odometry_color.zip -d data/kitti  # RGB files
unzip -o data/kitti/data_odometry_poses.zip -d data/kitti  # Ground truth pose
```

### VBR

**Option A: Download pre-processed data from Hugging Face**
```bash
mkdir -p data  # Run from repository root
wget -c https://huggingface.co/datasets/Junyi42/vbr_processed/resolve/main/vbr_processed.tar.gz -O data/vbr_processed.tar.gz
tar -xzf data/vbr_processed.tar.gz -C data
rm -f data/vbr_processed.tar.gz
```

**Option B: Download raw data and preprocess locally**

Download the raw VBR dataset from the [official VBR devkit](https://github.com/rvp-group/vbr-devkit) and place the sequences under `data/vbr/`. Then run:
```bash
python eval/datasets_preprocess/vbr_preprocess.py  # Run from repository root
```

## Long Evaluation Launch


Run single sequence:
```bash
# KITTI example
bash eval/demo_run_longeval.sh --cuda 0 --model LoGeR --mode kitti --seq 00 --win 32

# VBR example
bash eval/demo_run_longeval.sh --cuda 0 --model LoGeR_star --mode vbr --seq campus_train1 --win 64
```

Run all benchmark sequences:
```bash
bash eval/run_kitti.sh
bash eval/run_vbr.sh
```

## Long Evaluation Metrics

Compile benchmark tools:
```bash
cd eval/long_eval_script
g++ -o vbr_benchmark vbr_benchmark.cpp -I /usr/include/eigen3 -O3 -std=c++17
g++ -o kitti_benchmark kitti_benchmark.cpp -I /usr/include/eigen3 -O3 -std=c++17
```

Run VBR benchmark:
```bash
cd eval/long_eval_script
./vbr_benchmark ../../data/vbr/processed_gt ../../results/viser_pi3_vbr --plot
```

Run KITTI benchmark:
```bash
cd eval/long_eval_script
./kitti_benchmark ../../data/kitti/dataset/poses ../../results/viser_pi3_kitti --plot
```

Notes:
- `vbr_benchmark` expects `*_es.txt` trajectories in TUM format.
- `kitti_benchmark` expects estimated trajectories in TUM format and GT in KITTI pose format.
- `--plot` generates trajectory visualizations and error dumps.



## Short Evaluation Launch

### LoGeR

Relative pose estimation on **ScanNet**:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash eval/relpose/run_scannet.sh \
  LoGeR \
  --num-processes 8 --port 29122 --window-size 64 --overlap-size 3 \
  --datasets 'scannet_s3_1000,scannet_s3_50,scannet_s3_90,scannet_s3_100,scannet_s3_150,scannet_s3_200,scannet_s3_300,scannet_s3_400,scannet_s3_500,scannet_s3_600,scannet_s3_700,scannet_s3_800,scannet_s3_900'
```

Relative pose estimation on **TUM-dynamics**:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash eval/relpose/run_tum.sh \
  LoGeR \
  --num-processes 8 --port 29122 --window-size 64 --overlap-size 3 \
  --datasets 'tum_s1_1000,tum_s1_50,tum_s1_100,tum_s1_150,tum_s1_200,tum_s1_300,tum_s1_400,tum_s1_500,tum_s1_600,tum_s1_700,tum_s1_800,tum_s1_900'
```

Multi-view reconstruction on **7scenes** (sequential sampling):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash eval/mv_recon/run.sh \
  LoGeR \
  --num-processes 8 --port 29552 --window-size 64 --overlap-size 3 \
  --max-frames "500,450,400,50,100,150,200,250,300,350"
```

Multi-view reconstruction on **7scenes** (uniform sampling):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash eval/mv_recon/run.sh \
  LoGeR \
  --num-processes 8 --port 29552 --window-size 64 --overlap-size 3 \
  --max-frames "100,500,1000" --frame-sampling uniform
```

### LoGeR*

Relative pose estimation on **ScanNet**:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash eval/relpose/run_scannet.sh \
  LoGeR_star \
  --num-processes 8 --port 29122 --window-size 64 --overlap-size 3 --se3 \
  --datasets 'scannet_s3_1000,scannet_s3_50,scannet_s3_90,scannet_s3_100,scannet_s3_150,scannet_s3_200,scannet_s3_300,scannet_s3_400,scannet_s3_500,scannet_s3_600,scannet_s3_700,scannet_s3_800,scannet_s3_900'
```

Relative pose estimation on **TUM-dynamics**:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash eval/relpose/run_tum.sh \
  LoGeR_star \
  --num-processes 8 --port 29122 --window-size 64 --overlap-size 3 --se3 \
  --datasets 'tum_s1_1000,tum_s1_50,tum_s1_100,tum_s1_150,tum_s1_200,tum_s1_300,tum_s1_400,tum_s1_500,tum_s1_600,tum_s1_700,tum_s1_800,tum_s1_900'
```

Multi-view reconstruction on **7scenes** (sequential sampling):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash eval/mv_recon/run.sh \
  LoGeR_star \
  --num-processes 8 --port 29552 --window-size 64 --overlap-size 3 --se3 \
  --max-frames "500,450,400,50,100,150,200,250,300,350"
```

Multi-view reconstruction on **7scenes** (uniform sampling):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash eval/mv_recon/run.sh \
  LoGeR_star \
  --num-processes 8 --port 29552 --window-size 64 --overlap-size 3 --se3 \
  --max-frames "100,500,1000" --frame-sampling uniform
```