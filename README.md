# Object Distance Estimation Using Neural Networks

![Course](https://img.shields.io/badge/Course-CSCI%206366%20Neural%20Networks%20%26%20Deep%20Learning-0055A4?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![YOLOv11](https://img.shields.io/badge/Detection-YOLO11x-00FFFF?style=flat-square)
![MiDaS](https://img.shields.io/badge/Depth-MiDaS%20DPT--Large-8A2BE2?style=flat-square)
![OpenCV](https://img.shields.io/badge/Stereo-OpenCV%20SGBM-5C3EE8?style=flat-square&logo=opencv&logoColor=white)

> **Combining deep monocular depth estimation and neural object detection with classical stereo geometry to estimate metric object distances across indoor and outdoor scenes.**

---

## Project Summary
This project combines neural object detection with monocular depth estimation to identify objects in a scene and estimate their metric distance from the camera. Using pretrained **YOLO11x** for object localization and classification alongside **MiDaS** (`DPT_Large` and `MiDaS_small`) for relative inverse-depth prediction, we evaluate how neural depth estimation compares against classical stereo geometry (OpenCV `StereoSGBM` + WLS filtering) and a known-object-size perspective baseline across indoor and outdoor environments.

### Core Objectives
1. **Depth Map Generation:** Compare dense relative depth maps from transformer-based MiDaS (`DPT_Large` vs. `MiDaS_small`) against calibrated stereo disparity ($Z = \frac{f \cdot B}{d}$).
2. **Object-Level Distance Estimation:** Extract bounding boxes via `YOLO11x` and evaluate three distance estimation strategies: stereo-metric disparity, perspective known-object-size heuristics ($Z = \frac{f \cdot H_{\text{real}}}{h_{\text{px}}}$), and least-squares scaled MiDaS predictions.
3. **Failure Mode & Runtime Analysis:** Quantify valid-depth coverage, inference latency, and edge cases (occlusion, small distant objects, low-texture indoor surfaces, and non-stereo time-shifted frames).

---

## System Architecture & Pipeline

```text
Stereo / RGB Input ──┬──► [OpenCV StereoSGBM + WLS] ───► Metric Disparity (Z = f·B / d)
                     │
                     ├──► [MiDaS DPT_Large / Small] ───► Relative Inverse Depth Heatmap
                     │
                     └──► [Ultralytics YOLO11x] ───────► Bounding Boxes + Class Labels
                                                                │
                                                                ▼
                                                [Object-Level Distance Aggregation]
                                                (Stereo vs. Size Heuristic vs. Scaled MiDaS)
```

---

## Tech Stack & Tools

| Component | Tool / Model | Role in Pipeline |
| :--- | :--- | :--- |
| **Object Detection** | `Ultralytics YOLO11x` | Pretrained 2D object localization & COCO class prediction |
| **Monocular Depth** | `Intel ISL MiDaS` (`DPT_Large`, `MiDaS_small`) | Vision Transformer dense inverse-depth estimation & speed benchmarking |
| **Classical Stereo** | `OpenCV StereoSGBM` + `WLS Filter` | Rectified stereo matching and disparity refinement (`opencv-contrib-python`) |
| **Deep Learning Backend** | `PyTorch` & `torch.hub` | Hardware-accelerated inference (`CUDA` / `Apple MPS` / `CPU`) |
| **Data & Visualization** | `NumPy`, `Matplotlib`, `Pillow` | Metric scaling, quantile analysis, TURBO heatmaps, and annotated overlays |

---

## Team Members & Responsibilities

| Team Member | GitHub Handle | Primary Role & Contribution |
| :--- | :--- | :--- |
| **Ilaha Behbudova** | [@ibehbudova7711](https://github.com/ibehbudova7711) | **Data & Stereo Geometry:** Verify image pairing & camera calibration, implement `StereoSGBM`, and handle invalid disparities. |
| **Nijat Jafarov** | [@nijatjafarov](https://github.com/nijatjafarov) | **Neural Depth (MiDaS):** Integrate `DPT_Large` & `MiDaS_small`, analyze DPT architecture, and benchmark depth quality vs. latency. |
| **Laman Khudadatzada** | [@nnamanx](https://github.com/nnamanx) | **Detection & Distance:** Integrate `YOLO11x`, tune confidence/IoU thresholds, and implement box-level distance aggregation. |
| **Aysu Sarkarova** |  [@AysuSerkerova0625](https://github.com/AysuSerkerova0625) | **Evaluation & Visualization:** Design cross-method error metrics, measure valid-depth coverage, and build comparison plots. |

---

## Datasets

* **KITTI 2015 Stereo / Scene Flow Dataset**
  * **Usage:** Calibrated outdoor road scenes (`road1`, `road2`) with known camera intrinsics ($f = 721.54\text{ px}$) and stereo baseline ($B = 0.537\text{ m}$), plus a non-stereo scene-flow pair (`000020`) to test temporal mismatch failure modes.
  * **Original Source:** [KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php)
* **Custom Indoor Stereo Pairs (Nikon D3X & HD Lab)**
  * **Usage:** High-resolution indoor pairs (`lab`, `room`, `hallway`) utilizing EXIF focal-length estimation to stress-test classical stereo matching on textureless walls and evaluate single-image baselines.
  * **Reference Calibration Docs:** [OpenCV Camera Calibration and 3D Reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

---

## Repository Structure

```text
├── data/                           # Sample stereo pairs & calibration metadata
├── notebooks/
│   └── 01_initial_pipeline.ipynb   # Initial data loader, calibration checks & StereoSGBM baseline
├── outputs/                        # Generated disparity maps & preliminary figures
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## Quickstart & Setup

1. **Clone the repository:**
   ```bash
    git clone https://github.com/nnamanx/Object-Distance-Estimation-6366.git
    cd Object-Distance-Estimation-6366
   ```

2. **Install dependencies:**
   ```bash
   pip install torch torchvision ultralytics opencv-contrib-python numpy matplotlib pillow timm
   ```

3. **Run the initial prototype notebook:**
   Open `notebooks/01_initial_pipeline.ipynb` in Jupyter or VS Code and run top-to-bottom to verify dataset loading, camera intrinsics parsing, and initial `StereoSGBM` disparity maps.

---

## Roadmap & Current Status (Deliverable II)

- [x] Define project architecture, literature review, and team workload division
- [x] Curate outdoor (KITTI 2015) and indoor (Nikon D3X / Lab) test pairs and verify intrinsics
- [x] Implement initial classical stereo disparity pipeline (`StereoSGBM` + WLS filtering)
- [ ] Integrate pretrained `MiDaS` (`DPT_Large` & `MiDaS_small`) depth inference
- [ ] Integrate `YOLO11x` object detection and per-box distance aggregation
- [ ] Run full quantitative error/latency evaluation and compile final presentation
