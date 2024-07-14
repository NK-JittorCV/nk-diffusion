# JDiffusion

## Introduction

JDiffusion is a diffusion model library for generating images or videos based on [Jittor](https://github.com/Jittor/jittor), [Jtorch](https://github.com/JITTorch/jtorch), [diffusers](https://github.com/huggingface/diffusers).

## Installation
### 0. Clone JDiffusion & Prepare Env
```bash
git clone https://github.com/JittorRepos/JDiffusion.git
#We recommend using conda to configure the Python environment.
conda create -n jdiffusion python=3.9
conda activate jdiffusion
```
### 1. Install Requirements

Our code is based on JTorch, a high-performance dynamically compiled deep learning framework fully compatible with the PyTorch interface, please install our version of library.

```bash
pip install git+https://github.com/JittorRepos/jittor
pip install git+https://github.com/JittorRepos/jtorch
pip install git+https://github.com/JittorRepos/diffusers_jittor
pip install git+https://github.com/JittorRepos/transformers_jittor
```
or just
```bash
pip install -r requirement.txt
```
### 2. Install JDiffusion
```bash
cd JDiffusion
pip install -e .
```
We also provide a [docker image](https://cg.cs.tsinghua.edu.cn/jittor/assets/docker/jdiffusion.tar) (md5:62c305028dae6e62d3dff885d5bc9294) about our environment.

### 3.Optional Requirements
 If you encounter `No module named 'cupy'`:
```bash
# Install CuPy from source
pip install cupy
# Install CuPy for cuda11.2 (Recommended, change cuda version you use)
pip install cupy-cuda112
```
