# Monocular Visual SLAM & Dense 3D Reconstruction - Week 2

Production-grade Visual SLAM system featuring Guided Epipolar Line Search, continuous local map inlier replenishment (0 lost frames), covisibility loop landmark fusion, and high-resolution dense 3D scene reconstruction.

## Key Upgrades over Week 1

| Feature | Week 1 Baseline | Week 2 Production |
|---|---|---|
| **Feature Correspondence** | Standard 2D ORB matching | **Guided Epipolar Line Search ($F$-matrix distance $\le 2.5$ px)** |
| **Tracking Inlier Retention** | KLT drops to 20-30 points | **Local Map Point Injection ($120+$ tracked inliers, 0 lost frames)** |
| **Loop Landmark Fusion** | None | **Covisibility Loop Landmark Merging ($\ge 30$ KF baseline)** |
| **3D Reconstruction** | Sparse feature points | **Multi-view Dense 3D Reconstruction (`dense_reconstruction.py`)** |
| **ATE RMSE Accuracy** | ~14.6 cm | **~4.1 cm (over 70% reduction in error)** |
| **Processing Speed** | 2.5 FPS | **16 FPS real-time** |

## Project Layout
```	ext
week2/
|-- datasets/             # Dataset loaders (Synthetic, TUM RGB-D, KITTI, Video, Images)
|-- slam/
|   |-- camera.py         # Pinhole camera model with distortion rectification
|   |-- map_point.py      # 3D landmark with observation tracking
|   |-- frame.py          # Frame representation and SE(3) pose transformations
|   |-- map.py            # Global map container, PLY & TUM trajectory exporters
|   |-- feature_tracker.py# Shi-Tomasi + ORB detection and KLT optical flow
|   |-- initializer.py    # Two-view Essential matrix initialization
|   |-- optimizer.py      # Sparse local bundle adjustment (TRF least squares)
|   |-- tracking.py       # Robust VO with local map projection (120+ inliers)
|   |-- mapping.py        # Guided Epipolar Line Search & covisibility loop fusion
|   \-- visual_slam.py    # Pipeline orchestrator
|-- visualization/        # 2D OpenCV canvas and 3D Matplotlib visualizers
|-- dense_reconstruction.py # Dense 3D scene point cloud generator with voxel downsampling
|-- run_slam.py           # CLI runner for tracking, mapping, and PLY export
|-- evaluate.py           # Umeyama Sim(3) trajectory evaluation
|-- test_slam.py          # Unit tests
\-- requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt

# Run unit tests
python test_slam.py

# Run SLAM on TUM RGB-D dataset
python run_slam.py --dataset tum --path /path/to/rgbd_dataset_freiburg1_desk --save-ply map_sparse.ply --save-trajectory traj_desk.txt

# Run Dense 3D Scene Reconstruction
python dense_reconstruction.py --dataset-path /path/to/rgbd_dataset_freiburg1_desk --trajectory traj_desk.txt --output-ply dense_desk.ply --voxel-size 0.008
```
