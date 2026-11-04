
# e2eET Skeleton Based HGR Using Data-Level Fusion

> [!WARNING]
> This repository has been archived and is no longer maintained. It is now read-only. Feel free to fork or use it as needed.

This repository holds the PyTorch/FastAI implementation for **"Skeleton Based Hand Gesture Recognition Using Data Level Fusion"**. Alternatively, this [drive folder](https://drive.google.com/drive/folders/1BvoxkRDBK86A3_oNdQrnC8TLvp4l0W9x) mirrors this repository but also contains the preprocessed .pckl files and generated .rar spatiotemporal datasets that are otherwise too large to be uploaded here.

The proposed hand gesture recognition (HGR) framework transforms the dynamic hand gesture recognition task into a static image classification task using data-level fusion and a custom ensemble tuner multi-stream CNN architecture. For technical details, please refer to the following publication(s):

> **Real-Time Hand Gesture Recognition: Integrating Skeleton-Based Data Fusion and Multi-Stream CNN**
> *Oluwaleke Umar, Maki Habib, Mohamed Moustafa* &mdash; [[arXiv Preprint](https://arxiv.org/abs/2406.15003)]
>
> **Development of a Lightweight Real-Time Application for Dynamic Hand Gesture Recognition**
> *Oluwaleke Umar, Maki Habib* &mdash; [[IEEE ICMA Conference Contribution](https://ieeexplore.ieee.org/document/10216066)]

A real-time HGR application developed based on our framework only requires video streams from any standard inbuilt PC webcam and operates with a minimal CPU and RAM footprint, as shown in the [application memory utilisation image](./docs/Application-Memory-Utilization.png). The application underscores the utility of our proposed framework for reducing the hardware requirements and computational complexity of the HGR task on a standard PC while achieving acceptable latency, frames-per-second, and classification accuracy.

<p align="center">
  <img src="./docs/Generalized-e2eET-Network-Architecture.png" style="width:560px;">
</p>

## Prerequisites

The project uses Python 3.12 and [uv](https://docs.astral.sh/uv/). Model training requires an NVIDIA GPU with a driver supporting CUDA 12.4; the real-time gesture recognition can run a trained model on CPU or GPU.

Create the environment from the repository root:

```bash
uv sync
```

The dependencies are declared in `pyproject.toml` and pinned in `uv.lock`. `uv run` finds the root project from any subdirectory.

## Repository Layout and Reproduction

- `data-pipeline/` contains preprocessing notebooks and scripts for the benchmark datasets.
- `experiments/` contains the training and evaluation code.
- `results/` contains the recorded benchmark results.
- `runs/` contains TensorBoard event archives linked to the result logs; see its [archive guide](./runs/README.md).
- `gesture-recognition/` contains the live webcam application.

The detailed instructions for [preparing datasets](./docs/REPRODUCTION.md#data-level-fusion-processing-benchmark-datasets), [training and evaluation](./docs/REPRODUCTION.md#model-training-and-evaluation), and [running the live application](./docs/REPRODUCTION.md#live-real-time-hgr-application) are in the [reproduction and usage guide](./docs/REPRODUCTION.md).

## Benchmark Results

The final evaluation used the optimal view combinations and training schedules. See `results/allDatasets-e2eEnsembleTuning-Summary.xlsx` and its [summary log](./results/allDatasets-e2eEnsembleTuning-Summary.log) for details.

<!-- ˄˅↑↓ -->
| Benchmark Dataset | Classification Accuracy (%) |
| ----------------- | :-------------------------: |
| CNR               | 97.05 <sub>**↓1.73**</sub>  |
| LMDHG             | 98.97 <sub>**↑5.16**</sub>  |
| FPHA              | 91.83 <sub>**↓4.10**</sub>  |
| SHREC2017 (14G)   | 97.86 <sub>**↑0.24**</sub>  |
| SHREC2017 (28G)   | 95.36 <sub>**↓0.47**</sub>  |
| DHG1428 (14G)     | 95.83 <sub>**↓2.27**</sub>  |
| DHG1428 (28G)     | 92.38 <sub>**↓1.82**</sub>  |
| SBUKID            | 93.96 <sub>**↓4.34**</sub>  |
<!-- ˄˅↑↓ -->

## Live Gesture Recognition

<p align="center">
  <img src="./docs/HGR-Live-Demo-Video.gif" style="width:720px;">
</p>

The webcam application recognises the seven `Swipe` gestures from DHG1428. It requires trained models, available from this [drive folder](https://drive.google.com/drive/folders/1BvoxkRDBK86A3_oNdQrnC8TLvp4l0W9x), in `./gesture-recognition/models`.

```bash
cd gesture-recognition
uv run python liveStreamHGR.py
```

For model setup, seven-class dataset generation and gesture controls, see the [live application guide](./docs/REPRODUCTION.md#live-real-time-hgr-application).

## Citation

If you find this work contained in this repository useful in your research, please cite the following publication(s) as relevant:

```bibtex
@misc{Yusuf2024rth,
  title = {Real-Time Hand Gesture Recognition: Integrating Skeleton-Based Data Fusion and Multi-Stream CNN},
  author = {Oluwaleke Yusuf and Maki Habib and Mohamed Moustafa},
  year = {2024},
  eprint = {2406.15003},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  doi = {10.3390/technologies13110484},
  url = {https://arxiv.org/abs/2406.15003},
}
```

```bibtex
@INPROCEEDINGS{Yusuf2023doa,
  author={Yusuf, Oluwaleke and Habib, Maki},
  booktitle={2023 IEEE International Conference on Mechatronics and Automation (ICMA)},
  title={Development of a Lightweight Real-Time Application for Dynamic Hand Gesture Recognition},
  year={2023},
  pages={543-548},
  doi={10.1109/ICMA57826.2023.10216066}
}
```
