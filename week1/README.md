# Monocular Visual SLAM - Week 1 Baseline

Monocular Visual SLAM engine built from scratch in Python and OpenCV. Estimates 6-DoF camera trajectories and sparse 3D point cloud maps from a single moving camera.

## Features Built
- **Two-View Initialization**: Essential Matrix decomposition ($E = [t]_\times R$) with RANSAC cheirality recovery and metric scale normalization.
- **Hybrid Tracking**: Pyramidal KLT optical flow for smooth frame-to-frame tracking with ORB descriptor matching fallback.
- **PnP Motion Estimation**: Perspective-n-Point pose estimation with RANSAC and Levenberg-Marquardt non-linear refinement.
- **Triangulation**: Midpoint triangulation with parallax filtering (>1.2 deg) and reprojection validation.
- **Sliding-Window Bundle Adjustment**: Trust-Region Reflective (TRF) optimization with Huber loss on the recent 5 keyframes.
- **Trajectory Evaluation**: Closed-form Umeyama Sim(3) alignment against TUM RGB-D ground truth, computing Absolute Trajectory Error (ATE RMSE).

## Project Layout
```
week1/
|-- datasets/             # Dataset loaders (Synthetic, TUM RGB-D, KITTI, Video, Images)
|-- slam/
|   |-- camera.py         # Pinhole camera model with distortion rectification
|   |-- map_point.py      # 3D landmark with observation tracking
|   |-- frame.py          # Frame representation and SE(3) pose transformations
|   |-- map.py            # Global map container, PLY & TUM trajectory exporters
|   |-- feature_tracker.py# Shi-Tomasi + ORB detection and KLT optical flow
|   |-- initializer.py    # Two-view Essential matrix initialization
|   |-- optimizer.py      # Sparse local bundle adjustment (TRF least squares)
|   |-- tracking.py       # Visual odometry and PnP-RANSAC tracking
|   |-- mapping.py        # Triangulation and map point management
|   \-- visual_slam.py    # Pipeline orchestrator
|-- visualization/        # 2D OpenCV canvas and 3D Matplotlib visualizers
|-- run_slam.py           # CLI entry point
|-- evaluate.py           # Umeyama Sim(3) trajectory evaluation
|-- test_slam.py          # Unit tests
\-- requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt

# Run test suite
python test_slam.py

# Run on synthetic benchmark
python run_slam.py --dataset synthetic

# Run on TUM RGB-D dataset
python run_slam.py --dataset tum --path /path/to/rgbd_dataset_freiburg1_desk --save-ply map.ply
```
