# SparseMM: Head Sparsity Emerges from Visual Concept Responses in MLLMs

This repository contains PyTorch implementation for SparseMM.

[Project Page](https://cr400af-a.github.io/SparseMM/) | [arXiv Paper](https://arxiv.org/abs/2506.05344)

## Introduce SparseMM
We investigate how MLLMs process visual inputs by analyzing their attention mechanisms and reveal a surprising sparsity phenomenon: only a small subset (approximately less than 5\%) of attention heads in LLMs actively contribute to visual understanding, termed **Visual Heads**. To identify these heads efficiently, we design a training-free framework that quantifies head-level visual relevance through targeted response analysis. 

Building on this discovery, we introduce **SparseMM**, a KV-Cache optimization strategy that allocates asymmetric computation budgets to heads in LLMs based on their visual scores, leveraging **the sparity of visual heads** for accelerating the inference of MLLMs. Compared with prior KV-Cache acceleration methods that ignore the particularity of visual, SparseMM prioritizes stress and retaining visual semantics during decoding.

### Main idea of Visual Head
<p align="center" width="100%">
<img src="https://github.com/CR400AF-A/SparseMM/blob/main/assets/Visual_Head.png" alt="Visual_Head.png" width=80%>
</p>
<div>

### SparseMM for MLLM Acceleration
<p align="center" width="100%">
<img src="https://github.com/CR400AF-A/SparseMM/blob/main/assets/SparseMM.png" alt="SparseMM.png" width=50%>
</p>
<div>


## Main Results
### Results on Multi-modal Benchmarks
<p align="center" width="100%">
<img src="https://github.com/CR400AF-A/SparseMM/blob/main/assets/main_result.png" alt="main_result.png" width=90%>
</p>
<div>


### Efficiency Evaluation for SparseMM
<p align="center" width="100%">
<img src="https://github.com/CR400AF-A/SparseMM/blob/main/assets/efficiency.png" alt="efficiency.png" width=50%>
</p>
<div>


### Visualization of Visual Head
<p align="center" width="100%">
<img src="https://github.com/CR400AF-A/SparseMM/blob/main/assets/viz.png" alt="viz.png" width=80%>
</p>
<div>


## Get Started

### Install
1. Clone this repository:
```bash 
git clone https://github.com/CR400AF-A/SparseMM.git
cd SparseMM
```

2. Init your environment
```bash
conda create -n sparsemm python=3.10 -y
conda activate sparsemm
```

3. Install packages

Compile CUDA code for Flatten Cache Storage. If you encounter a CUDA compile error, please check your [GPU Virtual Architecture](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list) and [GPU Feature](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list), then change the corresponding compile flag in [csrc/build.py](https://github.com/CR400AF-A/SparseMM/blob/main/csrc/build.py#L20)
```bash
pip install packaging torch==2.5.1
pip uninstall ninja && pip cache purge && pip install ninja --no-cache-dir
cd csrc && make
cd ..
```

Install other packages
```bash
pip install -e .
pip install flash-attn==2.4.1 --no-build-isolation # currently only support FlashAttention
pip install qwen-vl-utils
```

4. Install lmms-eval for evaluation
```bash
cd lmms-eval
pip install -e .
cd ..
```


### Chase Visual Head
1. download dataset:
```bash
huggingface-cli download --repo-type dataset --resume-download nnethercott/synthdog-en-detection --local-dir /path/to/datasets/synthdog-en-detection

huggingface-cli download --repo-type dataset --resume-download detection-datasets/coco --local-dir /path/to/datasets/coco
```

2. process dataset:
```bash
python3 scripts/chase_visual_head/process_data.py
python3 scripts/chase_visual_head/process_data_coco.py
```

3. chase visual head:
```bash 
bash scripts/chase_visual_head/llava.sh
bash scripts/chase_visual_head/llava_coco.sh
bash scripts/chase_visual_head/qwen.sh
```

### Eval
```bash
bash scripts/eval/llava.sh
bash scripts/eval/mistral.sh
bash scripts/eval/qwen.sh
```

### Viz
```bash
bash scripts/others/viz.sh
```

### Speed and Memory
```bash
bash scripts/others/speed_and_memory.sh
```


## Citation

If you found this repository useful, please consider citing:

```
@article{wang2025sparsemm,
  title={SparseMM: Head Sparsity Emerges from Visual Concept Responses in MLLMs},
  author={Wang, Jiahui and Liu, Zuyan and Rao, Yongming and Lu, Jiwen},
  journal={arXiv preprint arXiv:2506.05344},
  year={2025}
}
```

## Acknowledgement

- Our codebase is conducted on [AdaKV](https://github.com/FFY0/AdaKV) and [PyramidKV](https://github.com/Zefan-Cai/KVCache-Factory).

- Thanks to [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) team, for building such a useful evaluation system!
