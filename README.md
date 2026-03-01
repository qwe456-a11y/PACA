# PACA: 3D Physical Adversarial Camouflage for Aircraft in Remote Sensing

<div align="center">
  <img src="https://img.shields.io/github/stars/qwe456-a11y/PACA?style=social" alt="GitHub Stars"/>
  <img src="https://img.shields.io/github/forks/qwe456-a11y/PACA?style=social" alt="GitHub Forks"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"/>
</div>

This repository provides the official implementation of the camouflage texture generator proposed in the paper **PACA: 3D Physical Adversarial Camouflage for Aircraft in Remote Sensing under Multi-View and Varying Weather Conditions**.

## 📝 Abstract

Most existing physical adversarial attack methods for remote sensing object detectors are optimized from a single nadir viewpoint, leading to significant performance degradation under multi-view observations. Moreover, most methods directly overlay adversarial textures onto images during digital training without adequately modeling the color distribution of complex environmental backgrounds, causing severe color inconsistency between adversarial textures and their surroundings. This reduces visual realism, training stability, and environmental adaptability. To address these limitations, this paper proposes Physical full-coverage Adversarial Camouflage for Aircraft (PACA), a framework designed to achieve robust deception across multiple viewpoints and complex environmental conditions. PACA integrates differentiable neural rendering with an environment-aware texture optimization mechanism: a neural rendering module (NeuroCam) enables precise texture mapping and multi-view projection on 3D aircraft surfaces, while an environment-aware module (FIRNet) extracts and fuses environmental color features to generate visually consistent and physically realizable full-coverage adversarial textures. Extensive experiments on the high-fidelity simulated remote sensing dataset (RS-AirSim) and real-world physical prototype tests demonstrate that the proposed method effectively degrades detection performance under multi-view observations. Compared with traditional remote sensing camouflage strategies, PACA reduces the detection AP from 0.30–0.60 to below 0.02 in most white-box settings, and maintains strong attack effectiveness in real-world physical deployments, while remaining robust to viewpoint variations and complex weather conditions.

## 📝 Framework

<img width="10194" height="5625" alt="fig2" src="https://github.com/user-attachments/assets/1158810c-84d7-4f9d-be95-4c75801abab8" />
The overview of PACA. First, a 3D environment is constructed in Blender using a set of physical transformations to generate a comprehensive dataset containing aircraft images, corresponding object masks, and camera parameters. The aircraft foreground and background are separated via segmentation using the mask annotations. Environmental features are extracted from the foreground aircraft using Extraction module. Meanwhile, the aircraft's 3D model and the associated camera parameters are input into the neural rendering module to generate the adversarial image x_adv. These rendered images are then fused with the extracted features and seamlessly integrated into the background. Finally, the adversarial camouflage is optimized by designing a loss function based on the output of an object detector.

## 📥 Downloads

### Dataset & Pre-trained Models
| Platform | Link | Password |
| -------- | ---- | -------- |
| Google Drive | [Download](https://drive.google.com/file/d/1CMJsnGwUMikdbI02R8udWmgzpI_MnZFq/view?usp=sharing) | - |
| Baidu Netdisk | [Download](https://pan.baidu.com/s/1p7__8bK6Z99CisriupQpvw) | qwer |

### Pre-built Camouflage Texture Generator
The core camouflage texture generator code is available in the `src/` directory.

## 🔧 Getting Started

### Prerequisites
- Python 3.6+
- PyTorch 1.10+
- CUDA 11.3+
- neural_renderer (install guide: https://winterwindwang.github.io/2021/07/22/nerual_rendered_build.html)

### Run Command

```bash
python ./src/pacaTest.py \
  --generator \
  --datapath \
  --obj \
  --faces \
  --textures \
  --output_dir \
  --batchsize 1 \
  --detector
```

### Parameter Description

| Parameter | Description |
|-----------|-------------|
| `--generator` | Path to generator model |
| `--datapath` | Path to dataset |
| `--obj` | Path to 3D object file |
| `--faces` | Path to exterior_face file |
| `--textures` | Path to texture file |
| `--output_dir` | Output directory |
| `--batchsize` | Batch size (default: 1) |
| `--detector` | Target detector model(yolov5x) |
