# Graph Information Propagation Network (GripNet) Model

This repository contains a PyTorch implementation of GripNet, as well as eight datasets and experiments on link prediction and node classification. The description of model and the results can be found in our paper:

[GripNet: Graph Information Propagation on Supergraph for Heterogeneous Graphs](https://www.sciencedirect.com/science/article/pii/S0031320322004538), Hao Xu, Shengqi Sang, Peizhen Bai, Ruike Li, Laurence Yang, Haiping Lu (Pattern Recognition 2022)

## GripNet and Baselines

GripNet is an effective and efficient framework to learn node representations on **heterogeneous graphs** (or Knowledge Graphs) for multi-relational **link prediction**, and **node classification**, when there is only a type of node/edge related to the task. It is also a natural framework for graph-like **data integration** (i.e. integrating multiple datasets).

We provide the implementations of GripNet in the root directory, and those of baselines:

- TransE, RotatE, ComplEx, DistMult and RGCN on link prediction (LP) in `baselines/LP_baselines/`, and
- GCN, GAT, RGCN and GANN on node classification (NC) in `baselines/NC_baselines/`.

Each model directory contains a bash script, which gives examples to run models. You can explore different model structures and hyperparameter settings by changing input parameters or code directly. It takes three steps to run these scripts.

### Step 1: Installation

(Prerequisites): Before installing `gripnet`, PyTorch (torch>=1.4.0) and PyTorch Geometric (torch_geometric<2.0) are required to be installed matching your hardware.

Install `gripnet` from source:
```bash
git clone https://github.com/NYXFLOWER/GripNet.git
cd GripNet
pip install .
```

### Step 2: Dataset Preparation

We constructed eight datasets for the experiments: three link prediction datasets (pose-0/1/2) and five node classification datasets (aminer and freebase-a/b/c/d). 

The datasets need to be downloaded into the corresponding directories with the provided links and unzipped:

- `.`: https://www.dropbox.com/s/hnt3v5890qozbtx/datasets.zip
- `./baselines/`: https://www.dropbox.com/s/wieca61m7jw2zqv/datasets_baselines.zip

Or, prepare the datasets using the following commands:

```bash
wget https://www.dropbox.com/s/hnt3v5890qozbtx/datasets.zip
unzip datasets.zip

cd baselines/
wget https://www.dropbox.com/s/wieca61m7jw2zqv/datasets_baselines.zip
```

Additionally, the raw data and code for constructing these datasets are available to download via: https://www.dropbox.com/s/41e43exro113pc9/data.zip

### Step 3: run scripts

To run a given experiment, execute our bash scripts as follows:

#### Run GripNet demo:

```bash
bash run.sh
```

#### Run baseline demo:

```bash
bash baselines/run_lp.sh	# link prediction
bash baselines/run_nc.sh	# node classification
```

## Citation

Please consider citing our paper below if you find GripNet or this code useful to your research.

```latex
@article{xu2022gripnet,
    title={GripNet: Graph Information Propagation on Supergraph for Heterogeneous Graphs},
    author={Xu, Hao and Sang, Shengqi and Bai, Peizhen and Li, Ruike and Yang, Laurence and Lu, Haiping},
    journal={Pattern Recognition},
    pages={108973},
    year={2022},
    publisher={Elsevier}
}
```
