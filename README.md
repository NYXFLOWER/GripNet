# MIP arch for integreting multiple graph

Node Representation Learning and Interpreting via Multi-graph Information Propagation

### 1. Motivation

Sometimes our data is naturally presented in the form of multiple interconnected graphs, like diseas-protein-drug graphs. How to learn explainable node representation when the data is a set of interconnected graphs is the core of this project. 

In other words, our **task** is to learn meaningful node representation for the downstream machine task, like link prediction and node classification.

use more info and get better performance. Not easy.  with the size of data increasing, two models may have different improvement.

### 2. Model = predictor(encoder, decoder) + explainer

#### 2.1 Graph Representation

Given a set of node $N = \{{N}^1_{n_1}, {N}^2_{n_2}, ..., {N}^m_{n_m}\}$, $m$ is node type, $n_m\in \mathcal{R}$ is the number of node for the corresponding node type. 

Each graph $G^m = \{(n_1, n_2, r)\}$, $n_1, n_2 \in N^m_{n_m}$, contains undirected edges labelled by edge type $r$, represented as a set of symmetric adjacency matrix $\{A^r_m\}_r \in A$. 

A set of nonsymmetric adjacency matrix $B = \{ B^{m_1, m_2}|m_1, m_2 \in m , m_1 \neq m_2\}$ denotes the directed subgraph between $G^{m_1}$ and $G^{m_2}$.

Key properties:

-   Each graph contains only one node type, and the edges are undirected.
-   Graph interconnection edges are directed
-   No isolated points on all the graphs
-   Nodes and edges can have multiple labels

#### 2.2 MIP Encoder

Learn the vector embeddings for each type of nodes $\{Z_m\}_m$ with GCNs.

Key Idea for information propagation: 

-   directed passed from one graph to another
-   undirected passed among nodes on a graph.

#### 2.3 Decoder

-   Node Classification: inner product kernel with shared class-specified vectors $R_r$
    -   $\hat{Y}_r = \textbf{Z} \textbf{R}_r$
-   Link Prediction:
    -   $\hat{Y}_r = \textbf{Z}_1 \textbf{R}_r \textbf{Z}_2$

#### 2.4 Explainer

The explainer contains two sets of adjacency matrix $\hat{A}$ and $\hat{B}$, which have the same non-zero elements as $A$ and $B$, but now with tunnable values $\hat{A^r_m}_{ji}=\hat{A^r_m}_{ij}\in (0,1]$ and $\hat{B}^{m_1, m_2}_{ji}=\hat{B}^{m_1, m_2}_{ij}\in (0,1]$.

Given a trained predictor and a set of instances we want to explain $S = \{s_n\}_n$. Set all parameters in the predictor untunabl. 

The loss function is:

$$
\begin{align}
loss =& \sum_n \log(1-P(s_n)) \\
+& \alpha (\sum_{r i j}\hat{A^r}_{ij} + \sum_{m i j}\hat{B}^{m_1, m_2}_{ij}) \\
+& \beta (\sum_{rij} \hat{A^r}_{ij}(1-\hat{A^r}_{ij}) + \sum_{ij} \hat{B}^{m_1, m_2}_{mij}(1-\hat{B}^{m_1, m_2}_{ij}))
\end{align}
$$

-   The term $(1)$ encourages P to take a value close to $1$. It can be replaced with any functions that decrease sharply near 1.
-   The term $(2)$ encourages the size of subgraph given by $\hat{A^r_m}$ and $\hat{B}^{m_1, m_2}$ to be as small as possible. It can be replace by any monotonically increase function.
-   The term $(3)$ encourages $\hat{A^r_m}_{ji}$ and $\hat{B}^{m_1, m_2}$ to be either $0$ or $1$. It can be replaced with any function that is $0$ at the points $x=0$ and $x=1$.

### 3. Evaluation

#### 3.1 Dataset and Task


| Dataset            | Task                             | Graph (Node)             | Size                                        |
| ------------------ | -------------------------------- | ------------------------ | ------------------------------------------- |
| POSE[^1]           | Multi-relational link prediction | 19k Protein +  284 Drug  | 1.4m p-p; 18k p-d; 4.6m d-d; 861 d-d label. |
| DTD [^2]           | Binary link prediction           | Disease + Protein + Drug | (To be processed)                           |
| (To be determined) | Multi-label node classification  |                          |                                             |
| (To be determined) | Binary node classification       |                          |                                             |

#### 3.2 Preformance with Pre-trained Embedding

-   Is the pre-training embeddings more accurate than the original?
-   Is the pre-training embeddings have a better interpretability?
-   Can the pre-training perform well when the data becomes more complex?

#### 3.3 Does the learned representation make sense?

-   Explain the prediction for different datasets (choose by auprc etc.)



[^1]: Sub-dataset of DECAGON dataset

[^2]: Chen H, Li J. Modeling Relational Drug-Target-Disease Interactions via Tensor Factorization with Multiple Web Sources. InThe World Wide Web Conference 2019 May 13 (pp. 218-227). https://dl.acm.org/doi/abs/10.1145/3308558.3313476



