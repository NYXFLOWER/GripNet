# Graph Information Propagation Network (GripNet) Model

This repository contains the official implementation of GripNet, as well as eight datasets and experiments on link prediction and node classification. The description of model and the results can be found in our paper:

[GripNet: Graph Information Propagation on Supergraph for Heterogeneous Graphs](https://doi.org/10.1016/j.patcog.2022.108973), Hao Xu, Shengqi Sang, Peizhen Bai, Ruike Li, Laurence Yang, Haiping Lu (Pattern Recognition, 2023)

🍺 **Update August 2022**: *Check out [this work](https://doi.org/10.1145/3511808.3557676) by Haiping Lu et al. (CIKM, 2022) from the PyKale team. In the [`pykale`](https://github.com/pykale/pykale) library, the structure and interface of GripNet implementation are improved, which makes it more convenient to construct GripNet models applied to knowledge graphs with high heterogeneity.*

## GripNet and Baselines

GripNet is an effective and efficient framework to learn node representations on **heterogeneous graphs** (or Knowledge Graphs) for multi-relational **link prediction**, and **node classification**, when there is only a type of node/edge related to the task. It is also a natural framework for graph-like **data integration** (i.e. integrating multiple datasets).

We provide the implementations of GripNet in the root directory, and those of baselines:

- TransE, RotatE, ComplEx, DistMult and RGCN on link prediction (LP) in `baselines/LP_baselines/`, and
- GCN, GAT, and RGCN on node classification (NC) in `baselines/NC_baselines/`.

Each model directory contains a bash script, which gives examples to run models. You can explore different model structures and hyperparameter settings by changing input parameters or code directly. 

It takes three steps to run these scripts.

### Step 1: Installation

All models in this repository are built on top of the [`PyTorch`](https://pytorch.org/get-started/locally/) and [`PyG`](https://github.com/pyg-team/pytorch_geometric). Before installing the `gripnet` package, `torch>=1.4.0` and `torch_geometric<2.0` are required to be installed matching your hardware.

Then, install the `gripnet` from source:
```bash
git clone https://github.com/NYXFLOWER/GripNet.git
cd GripNet
pip install .
```

### Step 2: Dataset preparation

We constructed eight datasets for the experiments: three link prediction datasets (pose-0/1/2) and five node classification datasets (aminer and freebase-a/b/c/d).

The datasets need to be downloaded to the corresponding directories with the provided links and unzipped:

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

Additionally, the raw data and code for constructing these datasets are available to download using: 
```bash
wget https://www.dropbox.com/s/41e43exro113pc9/data.zip 
```
We collect the data from the [BioSNAP](https://snap.stanford.edu/biodata/index.html), [AminerAcademicNetwork](https://dl.acm.org/doi/10.1145/1401890.1402008), and [Freebase](https://ieeexplore.ieee.org/document/9300240) databases.

### Step 3: Running scripts

We provide descriptions of arguments in these bash scripts.
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

🧋 **Note when not using GPUs**: *We use the [`pytorch_memlab`](https://github.com/Stonesjtu/pytorch_memlab) package by default to evaluate the GPU memory usage during training. If you are trining GripNet models on CPUs only, please find and comment all lines of `@profile` in the code. For example, comment Line 112 in `GripNet-pose.py`*:

https://github.com/NYXFLOWER/GripNet/blob/43022286290ae10f1d615520cc8ef37320e41953/GripNet-pose.py#L109-L113

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
    <td align="center"><a href="http://www.peizhenbai.me"><img src="https://avatars.githubusercontent.com/u/67964033?v=4?s=100" width="100px;" alt=""/><br /><sub><b>peizhenbai</b></sub></a><br /><a href="https://github.com/NYXFLOWER/GripNet/commits?author=pz-white" title="Code">💻</a> <a href="https://github.com/NYXFLOWER/GripNet/commits?author=pz-white" title="Tests">⚠️</a> <a href="#data-pz-white" title="Data">🔣</a></td>
    <td align="center"><a href="https://github.com/Lyric-19"><img src="https://avatars.githubusercontent.com/u/55618685?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Lyric-19</b></sub></a><br /><a href="https://github.com/NYXFLOWER/GripNet/issues?q=author%3ALyric-19" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/O3Ol"><img src="https://avatars.githubusercontent.com/u/46882376?v=4?s=100" width="100px;" alt=""/><br /><sub><b>O3Ol</b></sub></a><br /><a href="https://github.com/NYXFLOWER/GripNet/issues?q=author%3AO3Ol" title="Bug reports">🐛</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
