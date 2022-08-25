# Graph Information Propagation Network (GripNet) Model
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

This repository contains a PyTorch implementation of GripNet, as well as eight datasets and experiments on link prediction and node classification. The description of model and the results can be found in our paper:

[GripNet: Graph Information Propagation on Supergraph for Heterogeneous Graphs](https://www.sciencedirect.com/science/article/pii/S0031320322004538), Hao Xu, Shengqi Sang, Peizhen Bai, Ruike Li, Laurence Yang, Haiping Lu (Pattern Recognition 2022)

## GripNet and Baselines

GripNet is an effective and efficient framework to learn node representations on **heterogeneous graphs** (or Knowledge Graphs) for multi-relational **link prediction**, and **node classification**, when there is only a type of node/edge related to the task. It is also a natural framework for graph-like **data integration** (i.e. integrating multiple datasets).

We provide the implementations of GripNet in the root directory, and those of baselines:

- TransE, RotatE, ComplEx, DistMult and RGCN on link prediction (LP) in `baselines/LP_baselines/`, and
- GCN, GAT, and RGCN on node classification (NC) in `baselines/NC_baselines/`.

Each model directory contains a bash script, which gives examples to run models. You can explore different model structures and hyperparameter settings by changing input parameters or code directly. It takes three steps to run these scripts.

### Step 1: Installation

(Prerequisites): Before installing `gripnet`, PyTorch (torch>=1.4.0) and PyTorch Geometric (torch_geometric<2.0) are required to be installed matching your hardware.

Install `gripnet` from source:
```bash
git clone https://github.com/NYXFLOWER/GripNet.git
cd GripNet
pip install .
```

### Step 2: Dataset preparation

We constructed eight datasets for the experiments: three link prediction datasets (pose-0/1/2) and five node classification datasets (aminer and freebase-a/b/c/d).

The datasets need to be downloaded into the corresponding directories with the provided links and unzipped:

- `.`: https://www.dropbox.com/s/hnt3v5890qozbtx/datasets.zip
- `./baselines/`: https://www.dropbox.com/s/wieca61m7jw2zqv/datasets_baselines.zip

Or, prepare the datasets using the following commands:

```bash
wget https://www.dropbox.com/s/hnt3v5890qozbtx/datasets.zip
unzip datasets.zip && rm datasets.zip

cd baselines/
wget https://www.dropbox.com/s/g81hgxnewi7br8d/datasets_baselines.zip
unzip datasets_baselines.zip && rm datasets_baselines.zip
```

Additionally, the raw data and code for constructing these datasets are available to download via: https://www.dropbox.com/s/41e43exro113pc9/data.zip

### Step 3: Running scripts

To run a given experiment, execute our bash scripts as follows:

#### Run GripNet demo:

```bash
bash run.sh
```

#### Run baseline demo:

```bash
cd baselines
bash run_lp.sh	# link prediction
bash run_nc.sh	# node classification
```

Note that argument descriptions are provided in these bash scripts.

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

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/sangsq"><img src="https://avatars.githubusercontent.com/u/16742808?v=4?s=100" width="100px;" alt=""/><br /><sub><b>sangsq</b></sub></a><br /><a href="https://github.com/NYXFLOWER/GripNet/commits?author=sangsq" title="Code">💻</a> <a href="#ideas-sangsq" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="http://www.peizhenbai.me"><img src="https://avatars.githubusercontent.com/u/67964033?v=4?s=100" width="100px;" alt=""/><br /><sub><b>peizhenbai</b></sub></a><br /><a href="https://github.com/NYXFLOWER/GripNet/commits?author=pz-white" title="Code">💻</a> <a href="https://github.com/NYXFLOWER/GripNet/commits?author=pz-white" title="Tests">⚠️</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!