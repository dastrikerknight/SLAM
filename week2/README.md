# Monocular Visual SLAM and Dense Reconstruction — Week 2 Prototype Milestone

Week 2 focuses on improving robustness and adding experimental reconstruction utilities over the Week 1 baseline. The results are useful for experimentation but have not been evaluated across a broad benchmark suite.

## Experimental Improvements over Week 1

| Feature | Week 1 Baseline | Week 2 |
|---|---|---|
| **Feature correspondence** | Standard 2D ORB matching | Guided epipolar line search (F-matrix distance <= 2.5 px) |
| **Tracking inlier retention** | KLT can drop to 20–30 points | Local map point injection for more robust tracking (experimental) |
| **Loop landmark fusion** | None | Covisibility-based landmark merging at >= 30 KF baseline (experimental) |
| **3D reconstruction** | Sparse feature points | Experimental dense point cloud via `dense_reconstruction.py` |
| **ATE RMSE** | ~14.6 cm (author-reported, TUM Desk) | ~4.1 cm (author-reported, TUM Desk, one local test run) |
| **Processing speed** | ~2.5 FPS | ~16 FPS (author-reported, one local test run) |

> **Note:** The ATE and FPS figures above were observed during a particular local test run and are reported by the author. These values are not guaranteed and require reproducible benchmark scripts before being treated as validated results. Performance varies with hardware, dataset, and parameter settings.

## Project Layout

```text
week2/
|-- datasets/             # Dataset loaders (Synthetic, TUM RGB-D, KITTI, Video, Images)
|-- slam/
|   |-- camera.py         # Pinhole camera model with distortion rectification
|   |-- map_point.py      # 3D landmark with observation tracking
|   |-- frame.py          # Frame representation and SE(3) pose transformations
|   |-- map.py            # Global map container, PLY & TUM trajectory exporters
|   |-- feature_tracker.py# Shi-Tomasi + ORB detection and KLT optical flow
|   |-- initializer.py    # Two-view essential matrix initialization
|   |-- optimizer.py      # Sparse local bundle adjustment (TRF least squares)
|   |-- tracking.py       # VO with local map projection
|   |-- mapping.py        # Guided epipolar search & covisibility loop fusion
|   \-- visual_slam.py    # Pipeline orchestrator
|-- visualization/        # 2D OpenCV canvas and 3D Matplotlib visualizers
|-- dense_reconstruction.py # Experimental dense point cloud generator with voxel downsampling
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

# Run experimental dense 3D point cloud reconstruction
python dense_reconstruction.py --dataset-path /path/to/rgbd_dataset_freiburg1_desk --trajectory traj_desk.txt --output-ply dense_desk.ply --voxel-size 0.008
```

## Technical notes

- The dense reconstruction component is an experimental point-cloud pipeline. It has not been evaluated as a generally validated dense SLAM system.
- Guided epipolar search uses the fundamental matrix to constrain feature search; this improves inlier retention on well-calibrated sequences but is not robust to all conditions.
- Covisibility landmark fusion is a heuristic: it matches descriptors between spatially close keyframes when the camera revisits a region.
- The bundle adjustment window is small (5 keyframes) and is not a full global BA.

## Limitations

- Performance depends on the dataset, camera calibration, hardware, and parameter settings.
- Monocular visual SLAM has inherent scale ambiguity unless additional metric information is available.
- Tracking can fail under low texture, motion blur, lighting changes, dynamic objects, or insufficient camera motion.
- Reported FPS, inlier counts, and ATE values are dataset-specific and should not be treated as universal guarantees.
- The project is an educational/research prototype rather than a production-grade SLAM system.
- End-to-end benchmark reproducibility and broad failure-case evaluation are still incomplete.

## Development note

This project was developed and authored by me, with AI serving as a supplementary tool for documentation, debugging, idea validation, and iterative refinement. All key design decisions, system architecture, algorithms, implementation, integration, testing, and final review were completed by the author. The author assumes full responsibility for the project's functionality and outcomes.
