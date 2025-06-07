# HyperRA

> **Hyper**visor for **R**esource **A**nalysis – a simulator framework for comparing
> GPU vs. NPU performance, power, and area on real vision & language workloads.

## Motivation

Modern edge NPUs promise high inference throughput at very low power and silicon area compared to general-purpose GPUs.  
**HyperRA** provides a convenient way to:

1. **Benchmark** a suite of vision & NLP models on your local CUDA GPU (e.g. RTX 3060).
2. **Simulate** an NPU with configurable execution units, memory banks, and communication overhead.
3. **Compare** performance (latency & throughput), power dissipation, and area utilization side-by-side.
4. **Visualize** results and generate publication-quality plots.

## Introduction

HyperRA consists of three main components:

1. **GPU PPA profiler** (`cpp/main.cpp`):  
   Loads TorchScript models (`.pt`) and real data (`real_images/`, `real_texts.txt`),  
   measures average latency, power draw (via NVML), and area estimate (via utilization).

2. **NPU simulator** (`cpp/main_npu_sim.cpp`):  
   Replays the same vision workloads through a simple model of an NPU:  
   - `NPU_EUs` execution units  
   - `NPU_MEM_BANKS` memory banks  
   - `COMM_LAT_MS` per-image communication overhead  
   Combines measured GPU idle power, dynamic power deltas, and EU utilization to  
   estimate PPA under NPU-like hardware constraints.

3. **Plotting script** (`plots.py`):  
   Reads the two CSV outputs (`results.csv` and `results_npu.csv`),  
   and produces grouped bar charts for:
   - Average Latency (ms)  
   - Δ Power (W)  
   - Throughput (items/s)  
   - Area Used (mm²)  

## System Design


![HyperRA](https://github.com/user-attachments/assets/287ad94d-ad7d-489e-9b08-c8bfed9fc9ad)






- **Models & data** (10 vision nets + text models) live in `exported/` and `real_images` / `real_texts.txt`  
- **`main_gpu.cpp`** measures real GPU PPA over 30 s windows  
- **`main_npu_sim.cpp`** injects COMM_LAT_MS delays, limits EUs & memory banks, and computes “idealized” NPU PPA  
- **Python plotting** in `plots.py` generates bar-charts across GPU vs NPU for latency, power, throughput, area

---

## 📁 Large assets

All raw images, tokenized text, and model weights (~10 GB) live on Google Drive. We keep them out of GitHub LFS.  
**Download them here before running benchmarks:**

👉 [Google Drive: HyperRA assets (10 GB)](https://drive.google.com/drive/folders/1HUCQIMzozHY_Gm4rYlnSxDs3z-IzIaA3?usp=sharing)


- **Data ingestion**  
  Images are loaded via OpenCV, texts are tokenized/pre-tokenized into Torch Tensors.  
- **GPU Profiler**  
  TorchScript models are traced on the CUDA device. NVML polls power & utilization.  
- **NPU Simulator**  
  Models run in “batches” with extra COMM_LAT_MS delays to mimic DMA overhead.  
  Idle baseline power subtracted to estimate active power.  
  Area = (EU-util × #EUs × per-EU-area) + (#banks × per-bank-area).  
- **Plotting**  
  Aggregates CSVs and produces side-by-side bar charts.

## Getting Started

### Prerequisites

- CUDA Toolkit & NVIDIA driver  
- CMake ≥ 3.10, GNU C++17  
- libtorch (download the matching version), OpenCV, NVML headers  
- Python 3.8+, pandas, matplotlib, numpy  

### Clone & Download Models

```bash
git clone https://github.com/<you>/HyperRA.git
cd HyperRA
# download 10 GB of TorchScript exports to `exported/`:
# we provide a Google Drive link to avoid large repo size:
# https://drive.google.com/drive/folders/1HUCQIMzozHY_Gm4rYlnSxDs3z-IzIaA3?usp=sharing
```
Build GPU Profiler & NPU Simulator

```
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=~/libtorch ..      # adjust libtorch path
make gpu_ppa    # builds `gpu_ppa` (main.cpp)
make npu_sim    # builds `npu_sim` (main_npu_sim.cpp)
```

Run Benchmarks

```
# GPU profiling
./gpu_ppa

# NPU simulation
./npu_sim

```


Directory Structure

```
HyperRA/
├─ cpp/
│  ├─ main.cpp              # GPU profiler
│  ├─ main_npu_sim.cpp      # NPU simulator
│  └─ CMakeLists.txt
├─ real_images/             # your dataset of JPEG/PNG (224×224)
├─ real_texts.txt           # one sentence per line
├─ exported/                # TorchScript models & token data
├─ plots.py                 # Python plotting script
├─ README.md                # ← you’re reading it!
└─ docs/
   └─ architecture.png      # simple 2D block diagram

```


License 
AbdallahAboShoaib@RA.tech 

