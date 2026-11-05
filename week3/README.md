# Monocular Visual SLAM, Global Loop Closing & 3D Scene Reconstruction - Week 3

Production-grade Visual SLAM system featuring Bag-of-Visual-Words (BoW) loop detection, 7-DoF Sim(3) loop closure, non-linear Pose Graph Optimization (PGO), binary map persistence, Kidnapped Robot relocalization, volumetric TSDF fusion, and solid 3D surface meshing.

## Key Upgrades across Milestones

| Feature | Week 1 Baseline | Week 2 Robust Tracking | Week 3 Global SLAM & 3D Meshing |
|---|---|---|---|
| **Feature Tracking** | 2D ORB matching | Guided Epipolar Line Search | Hybrid Shi-Tomasi + FAST/ORB + Pyramidal KLT |
| **Inlier Retention** | Drops to 20-30 pts | 120+ local map inliers | 500-1,300+ inliers with probationary landmark retention |
| **Loop Closure** | None (open loop) | Covisibility heuristic | **Hierarchical BoW Inverted Index + Sim(3) 7-DoF PGO** |
| **Long-Horizon Drift** | Severe scale drift | Scale drift accumulates | **Permanently anchored scale (Sim(3) scale ~1.01)** |
| **Relocalization** | None (restart needed) | Heuristic retry | **Fast visual vocabulary relocalization (<50ms)** |
| **Map Persistence** | None (RAM-only) | None | **Binary map serialization & loading (`map.bin`)** |
| **3D Reconstruction** | Sparse point cloud | Raw dense point cloud | **Watertight solid triangle meshes (`.obj` / `.ply`) + TSDF** |
| **ATE RMSE (TUM Desk)** | ~14.6 cm | ~4.1 cm | **0.0399 m (3.99 cm, full 596 frames)** |
| **Tracking Stability** | Frequent tracking loss | Occasional lost frames | **0 lost frames across the entire dataset** |

## Project Layout

```text
week3/
|-- datasets/                 # Modular dataset loaders
|   |-- dataset_base.py       # Abstract base dataset class
|   |-- tum_dataset.py        # TUM RGB-D loader with synchronized metric depth
|   |-- kitti_dataset.py      # KITTI Odometry sequence loader
|   |-- synthetic_dataset.py  # 3D synthetic benchmark sequence generator
|   |-- image_folder_dataset.py
|   \-- video_dataset.py
|-- slam/                     # Core SLAM engine
|   |-- camera.py             # Pinhole model & lens undistortion
|   |-- frame.py              # Frame representation, depth handling, SE(3) poses
|   |-- map_point.py          # 3D landmark with multi-view observations
|   |-- map.py                # Global map container, 3-KF grace period culling, binary save/load
|   |-- feature_tracker.py    # Hybrid corner/ORB detection & KLT optical flow
|   |-- initializer.py        # Essential matrix two-view initialization
|   |-- tracking.py           # Metric PnP odometry, local map replenishment, RGB-D seeding
|   |-- mapping.py            # Guided epipolar search, covisibility fusion, landmark management
|   |-- optimizer.py          # Sparse local bundle adjustment with Huber loss
|   |-- bow_vocabulary.py     # Visual vocabulary and inverted file index
|   |-- loop_closing.py       # BoW loop detection & 7-DoF Sim(3) geometric verification
|   |-- pose_graph.py         # Global pose graph optimization (PGO)
|   |-- relocalization.py     # Kidnapped robot recovery via visual vocabulary
|   \-- visual_slam.py        # Top-level SLAM orchestrator
|-- dense_reconstruction/     # High-fidelity 3D scene reconstruction
|   |-- scene_tsdf_mapper.py  # Volumetric Truncated Signed Distance Function (TSDF)
|   |-- texture_projector.py  # Multi-view photographic projection & semantic voting
|   \-- orbit_renderer.py     # Hardware-accelerated 360° cinematic orbit video renderer
|-- depth_module.py           # Depth Anything V2 neural depth estimation
|-- scale_aligner.py          # Inverse-disparity linear metric scale & shift alignment
|-- semantic_module.py        # Lightweight LRASPP MobileNet semantic segmenter
|-- surface_meshing.py        # Point cloud to solid 3D triangle mesh reconstruction
|-- reconstruct_scene.py      # End-to-end full 3D reconstruction pipeline
|-- run_slam.py               # Main CLI runner for visual tracking, mapping, and mesh export
|-- evaluate.py               # Closed-form Umeyama Sim(3) trajectory evaluation & plotting
|-- test_slam.py              # Automated 8-component unit test suite
|-- requirements.txt          # Production dependencies
\-- .gitignore
```

## Quickstart

### 1. Run Unit Tests (Validates all 8 components in ~5s)
```bash
python test_slam.py
```

### 2. Run SLAM with Loop Closure on Synthetic Loop Benchmark
```bash
python run_slam.py --dataset synthetic --features 1500 --max-frames 60 --meshing --save-mesh synthetic_mesh.obj
```

### 3. Run SLAM on TUM RGB-D Sequence
```bash
python run_slam.py --dataset tum --path /path/to/rgbd_dataset_freiburg1_desk --save-ply map_sparse.ply --save-trajectory traj_tum.txt
```

### 4. Run Map Persistence & Relocalization
```bash
# Build and save map to binary
python run_slam.py --dataset synthetic --no-gui --max-frames 30 --save-map room_map.bin

# Reload pre-built map in relocalization mode
python run_slam.py --dataset synthetic --no-gui --max-frames 15 --load-map room_map.bin
```

### 5. Run Full 3D Scene Reconstruction (TSDF + Semantics + Orbit Video)
```bash
python reconstruct_scene.py --dataset-path /path/to/rgbd_dataset_freiburg1_desk --output-dir ./reconstructed_desk
```

### 6. Evaluate Trajectory vs Ground Truth (Umeyama Sim(3))
```bash
python evaluate.py --estimated traj_tum.txt --ground-truth /path/to/rgbd_dataset_freiburg1_desk/groundtruth.txt --save-plot slam_evaluation.png
```
