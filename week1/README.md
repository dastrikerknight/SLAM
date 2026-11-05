# Monocular Visual SLAM — Week 1 Prototype Baseline

Week 1 establishes the first working sparse monocular SLAM prototype. It is intended as a learning milestone and baseline for later experiments. The system estimates 6-DoF camera trajectories and builds a sparse 3D point cloud from a single moving camera using Python and OpenCV.

## Implemented Components

- **Two-view initialization** — Essential matrix decomposition with RANSAC and cheirality recovery. Reference scale normalization is applied to the monocular map; this does not recover absolute metric scale.
- **Feature tracking** — Pyramidal KLT optical flow for frame-to-frame tracking with ORB descriptor matching as a fallback.
- **PnP motion estimation** — Perspective-n-Point pose estimation with RANSAC and iterative refinement.
- **Linear triangulation** — Two-view linear triangulation with parallax checks (> 1.2 deg) and reprojection validation.
- **Sliding-window bundle adjustment** — Trust-Region Reflective (TRF) least-squares optimization with Huber loss over a local keyframe window (experimental).
- **Trajectory evaluation** — Closed-form Umeyama Sim(3) alignment against ground truth, computing Absolute Trajectory Error (ATE RMSE).

## Project Layout

```text
week1/
|-- datasets/             # Dataset loaders (Synthetic, TUM RGB-D, KITTI, Video, Images)
|-- slam/
|   |-- camera.py         # Pinhole camera model with distortion rectification
|   |-- map_point.py      # 3D landmark with observation tracking
|   |-- frame.py          # Frame representation and SE(3) pose transformations
|   |-- map.py            # Global map container, PLY & TUM trajectory exporters
|   |-- feature_tracker.py# Shi-Tomasi + ORB detection and KLT optical flow
|   |-- initializer.py    # Two-view essential matrix initialization
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

## Limitations

- Performance depends on the dataset, camera calibration, hardware, and parameter settings.
- Monocular visual SLAM has inherent scale ambiguity unless additional metric information is available.
- Tracking can fail under low texture, motion blur, lighting changes, dynamic objects, or insufficient camera motion.
- Reported FPS, inlier counts, and ATE values are dataset-specific and should not be treated as universal guarantees.
- The project is an educational/research prototype rather than a production-grade SLAM system.
- End-to-end benchmark reproducibility and broad failure-case evaluation are still incomplete.

## Development note

This project was developed and authored by me, with AI serving as a supplementary tool for documentation, debugging, idea validation, and iterative refinement. All key design decisions, system architecture, algorithms, implementation, integration, testing, and final review were completed by the author. The author assumes full responsibility for the project's functionality and outcomes.
