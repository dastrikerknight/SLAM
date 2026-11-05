# Monocular Visual SLAM, Experimental Loop Closure, and 3D Reconstruction — Week 3

Week 3 extends the prototype with experimental visual-vocabulary retrieval, rigid loop-closure verification, pose-graph refinement, map persistence, relocalization, TSDF integration, monocular-depth completion, semantic labeling, and mesh export.

## Implemented and Experimental Features

| Feature | Status | Notes |
|---|---|---|
| Hybrid Shi-Tomasi + ORB + KLT tracking | Implemented | Core sparse tracking path |
| Essential matrix two-view initialization | Implemented | Reference scale; not inherently metric |
| PnP-RANSAC motion estimation | Implemented | 6-DoF pose estimation |
| Linear triangulation | Implemented | OpenCV linear two-view triangulation |
| Local bundle adjustment (5-KF window) | Implemented | Huber loss, TRF least squares |
| Flat k-means BoW vocabulary | Implemented | Not a hierarchical vocabulary tree |
| BoW-based loop-closure candidate retrieval | Experimental | Rigid geometric verification via PnP, not Sim(3) |
| 6-DoF pose graph refinement | Experimental | Optimizes poses only; scale is not corrected |
| Binary map serialization / loading | Implemented | Save and reload SLAM maps |
| BoW relocalization | Experimental | Candidate matching and PnP verification |
| TSDF volumetric fusion (Open3D) | Experimental | Mesh extraction does not guarantee watertight results |
| Monocular depth via Depth Anything V2 | Experimental | Relative depth; heuristically aligned to metric when SLAM landmarks are available |
| Multi-view texture projection | Experimental | Photographic color reprojection onto mesh vertices |
| Semantic label projection | Exploratory | 2D segmentation (LRASPP MobileNetV3) projected onto reconstructed geometry; not validated 3D understanding |
| Mesh export (.obj / .ply) | Implemented | Via Open3D marching cubes + optional surface meshing |
| Trajectory evaluation (Umeyama Sim(3)) | Implemented | ATE RMSE against ground truth |

## Project Layout

```text
week3/
|-- datasets/                 # Modular dataset loaders
|   |-- dataset_base.py       # Abstract base class
|   |-- tum_dataset.py        # TUM RGB-D loader with synchronized depth
|   |-- kitti_dataset.py      # KITTI Odometry loader
|   |-- synthetic_dataset.py  # Synthetic benchmark generator
|   |-- image_folder_dataset.py
|   \-- video_dataset.py
|-- slam/                     # Core SLAM engine
|   |-- camera.py             # Pinhole model & lens undistortion
|   |-- frame.py              # Frame representation, SE(3) poses
|   |-- map_point.py          # 3D landmark with multi-view observations
|   |-- map.py                # Global map, 3-KF grace period culling, binary save/load
|   |-- feature_tracker.py    # Hybrid detection & KLT optical flow
|   |-- initializer.py        # Essential matrix two-view initialization
|   |-- tracking.py           # PnP odometry, local map replenishment, RGB-D seeding
|   |-- mapping.py            # Guided epipolar search, covisibility fusion
|   |-- optimizer.py          # Local bundle adjustment with Huber loss
|   |-- bow_vocabulary.py     # Flat k-means visual vocabulary and inverted index
|   |-- loop_closing.py       # BoW loop detection & rigid geometric verification
|   |-- pose_graph.py         # 6-DoF pose graph optimization
|   |-- relocalization.py     # BoW-based relocalization with PnP verification
|   \-- visual_slam.py        # Top-level SLAM orchestrator
|-- dense_reconstruction/     # Experimental 3D reconstruction modules
|   |-- scene_tsdf_mapper.py  # Open3D TSDF volume integration
|   |-- texture_projector.py  # Keyframe color and semantic projection onto mesh
|   \-- orbit_renderer.py     # Offscreen 360-degree orbit video renderer
|-- depth_module.py           # Depth Anything V2 relative depth estimation
|-- scale_aligner.py          # Heuristic inverse-disparity scale & shift alignment
|-- semantic_module.py        # LRASPP MobileNetV3 2D semantic segmentation
|-- surface_meshing.py        # PCA-Delaunay surface mesh from point cloud
|-- reconstruct_scene.py      # End-to-end reconstruction pipeline script
|-- run_slam.py               # CLI: tracking, mapping, mesh export
|-- evaluate.py               # Umeyama Sim(3) trajectory evaluation
|-- test_slam.py              # 8-component unit test suite
|-- requirements.txt
\-- .gitignore
```

## Quickstart

### 1. Run unit tests

```bash
python test_slam.py
```

### 2. Run SLAM on the synthetic benchmark

```bash
python run_slam.py --dataset synthetic --features 1500 --max-frames 60 --meshing --save-mesh synthetic_mesh.obj
```

### 3. Run SLAM on a TUM RGB-D sequence

```bash
python run_slam.py --dataset tum --path /path/to/rgbd_dataset_freiburg1_desk --save-ply map_sparse.ply --save-trajectory traj_tum.txt
```

### 4. Test map persistence and relocalization

```bash
# Save map to binary
python run_slam.py --dataset synthetic --no-gui --max-frames 30 --save-map room_map.bin

# Reload map and relocalize
python run_slam.py --dataset synthetic --no-gui --max-frames 15 --load-map room_map.bin
```

### 5. Run the full experimental reconstruction pipeline (TSDF + depth completion + semantics)

```bash
python reconstruct_scene.py --dataset-path /path/to/rgbd_dataset_freiburg1_desk --output-dir ./reconstructed_desk
```

### 6. Evaluate trajectory against ground truth

```bash
python evaluate.py --estimated traj_tum.txt --ground-truth /path/to/rgbd_dataset_freiburg1_desk/groundtruth.txt --save-plot slam_evaluation.png
```

## Current status

- **Core sparse SLAM path** (tracking, initialization, local BA, map management) is the most mature part of the codebase and is exercised by the unit tests.
- **Loop closure and relocalization** are experimental extensions. The loop-closure implementation performs rigid geometric verification using PnP/SE(3); it does not implement a full Sim(3) optimizer and does not correct accumulated scale error.
- **TSDF fusion, monocular depth completion, semantic projection, and mesh export** are exploratory features. They work on the tested sequence but have not been systematically evaluated across different datasets, cameras, or scene types.
- The comparison table in the previous README version contained headline numbers (ATE = 3.99 cm, 0 lost frames, 56k map points) that were author-reported from one local run on the TUM Desk sequence. Those numbers are preserved in `outputs/` as a record of that run; they are not universal performance guarantees.
- A reproducible benchmark protocol covering a range of sequences and failure modes is still needed.

## Claims requiring future validation

The following reported results have not been independently benchmarked and should be treated as preliminary observations from the author's local test environment:

- Full-sequence TUM ATE RMSE
- Tracking-loss rate and robustness across sequences
- Runtime / FPS on different hardware
- Loop-closure precision and recall
- Relocalization success rate across diverse scenarios
- Mesh completeness and geometric accuracy
- Semantic label accuracy relative to a ground-truth annotation

## Technical accuracy notes

- **Loop closure** uses BoW candidate retrieval followed by rigid PnP/SE(3) verification, not a 7-DoF Sim(3) optimizer. Scale is not part of the optimization.
- **Pose graph** refines 6-DoF camera poses only. It does not optimize scale.
- **BoW vocabulary** is a flat k-means clustering of ORB descriptors, not a hierarchical vocabulary tree.
- **Monocular initialization** applies a reference-scale normalization so the initial map has a sensible depth range. This does not recover metric scale; the map is in arbitrary units unless RGB-D depth is used.
- **Triangulation** uses OpenCV's linear triangulation (`cv2.triangulatePoints`).
- **Depth Anything V2** produces relative (inverse) depth. When SLAM landmarks are available, a linear scale-and-shift alignment is applied heuristically to bring the depth into rough metric correspondence.
- **Semantic labeling** projects 2D segmentation results (LRASPP MobileNetV3, PASCAL VOC classes) onto mesh vertices via camera reprojection. This is an exploratory visualization and has not been validated as 3D semantic understanding.
- **TSDF mesh extraction** via Open3D marching cubes produces a surface mesh. The result is not guaranteed to be watertight or geometrically complete.

## Limitations

- Performance depends on the dataset, camera calibration, hardware, and parameter settings.
- Monocular visual SLAM has inherent scale ambiguity unless additional metric information is available.
- Tracking can fail under low texture, motion blur, lighting changes, dynamic objects, or insufficient camera motion.
- Reported FPS, inlier counts, and ATE values are dataset-specific and should not be treated as universal guarantees.
- The project is an educational/research prototype rather than a production-grade SLAM system.
- End-to-end benchmark reproducibility and broad failure-case evaluation are still incomplete.

## Verification status

This README reports implementation capabilities and intended workflows for the Week 3 prototype. It does not claim that every feature has been independently benchmarked or validated across datasets. Output files in `outputs/` document one specific run on the TUM Freiburg 1 Desk sequence and are provided as illustrative examples only.

## Development note

This project was developed and authored by me, with AI serving as a supplementary tool for documentation, debugging, idea validation, and iterative refinement. All key design decisions, system architecture, algorithms, implementation, integration, testing, and final review were completed by the author. The author assumes full responsibility for the project's functionality and outcomes.
