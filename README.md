# Multi-Scale Representation Learning for Heterogeneous Networks (MSRL)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

Official implementation of the paper **"Multi-Scale Representation Learning for Heterogeneous Networks via Hawkes Point Processes"**. This repository contains code for dynamic heterogeneous network representation learning with Hawkes processes and triadic closure modeling.

## 📌 Table of Contents
- [Key Features](#key-features)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results Reproduction](#results-reproduction)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## 🔥 Key Features
- **Hawkes Process Integration**: Models temporal self-excitation effects between events
- **Triadic Closure Modeling**: Captures structural evolution patterns
- **Multi-Granularity Aggregation**: Micro-meso-macro neighborhood analysis
- **Dynamic Heterogeneous Support**: Handles multiple node/edge types over time
- **Reproducible Benchmarks**: Matching results from the original paper

## 🛠 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.7+ (For GPU acceleration)
- PyTorch 2.0+

### Step-by-Step Setup
```bash
# 1. Clone repository
git clone https://github.com/LSRLSZ/MSRL.git
cd MSRL

# 2. Install dependencies (adjust CUDA version as needed)
pip install -r requirements.txt

# 3. Install PyTorch Geometric extensions
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html