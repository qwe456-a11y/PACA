# PACA: 3D Physical Adversarial Camouflage for Aircraft in Remote Sensing

[![GitHub Stars](https://img.shields.io/github/stars/qwe456-a11y/PACA?style=social)](https://github.com/qwe456-a11y/PACA/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/qwe456-a11y/PACA?style=social)](https://github.com/qwe456-a11y/PACA/network/members)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

This repository provides the official implementation of the camouflage texture generator proposed in the paper **PACA: 3D Physical Adversarial Camouflage for Aircraft in Remote Sensing under Multi-View and Varying Weather Conditions**.

## 📝 Abstract

Most existing physical adversarial attack methods for remote sensing object detectors are optimized from a single nadir viewpoint, leading to significant performance degradation under multi-view observations. Moreover, most methods directly overlay adversarial textures onto images during digital training without adequately modeling the color distribution of complex environmental backgrounds, causing severe color inconsistency between adversarial textures and their surroundings. This reduces visual realism, training stability, and environmental adaptability. To address these limitations, this paper proposes Physical full-coverage Adversarial Camouflage for Aircraft (PACA), a framework designed to achieve robust deception across multiple viewpoints and complex environmental conditions. PACA integrates differentiable neural rendering with an environment-aware texture optimization mechanism: a neural rendering module (NeuroCam) enables precise texture mapping and multi-view projection on 3D aircraft surfaces, while an environment-aware module (FIRNet) extracts and fuses environmental color features to generate visually consistent and physically realizable full-coverage adversarial textures. Extensive experiments on the high-fidelity simulated remote sensing dataset (RS-AirSim) and real-world physical prototype tests demonstrate that the proposed method effectively degrades detection performance under multi-view observations. Compared with traditional remote sensing camouflage strategies, PACA reduces the detection AP from 0.30–0.60 to below 0.02 in most white-box settings, and maintains strong attack effectiveness in real-world physical deployments, while remaining robust to viewpoint variations and complex weather conditions.

## 📥 Downloads

### Dataset & Pre-trained Models
- **Google Drive**: [Download Link](https://drive.google.com/drive/folders/your-folder-id?usp=sharing)
- **Baidu Netdisk**: [Download Link]([https://pan.baidu.com/s/your-share-code](https://pan.baidu.com/s/1p7__8bK6Z99CisriupQpvw)) (Password: qwer)

### Pre-built Camouflage Texture Generator
The core camouflage texture generator code is available in the `src/` directory of this repository.

## 🔧 Getting Started

### Prerequisites
- Python 3.6+
- PyTorch 1.10+
- CUDA 11.3+

