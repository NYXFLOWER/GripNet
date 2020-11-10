## GripNet and Baselines

We provide the implementation of GripNet in the `.\GripNet` directory, and the code for baselines:

-   TransE, RotatE, ComplEx, DistMult and RGCN on link prediction (LP) in `.\LP_baselines`
-    GCN, GAT, RGCN and GANN on node classification (NC) in `.\NC_baselines`

Each model directory contains a `run.sh` file which gives examples for run models. You can explore different GripNet structures and hyperparameter settings by changing input parameters or code directly.

## Environment Requirment

Make the following packages are installed and meet the requirments.

-   Python >= 3.7
-   PyTorch >= 1.4.0 and its corresponding PyTorch Geometric version
-   Pytorch-memlab == 0.2.1
-   Numpy == 1.19.1
-   Pandas >= 0.24.2
-   Scikit-learn >= 0.21.2
-   Scipy >= 1.3.0

## Data for training and testing

Before running the GripNet and baseline models, please download the data with links provided below, put them into the corresponding data directories and unzip the file. 

-   `.\GripNet\data`: https://www.dropbox.com/s/4ctqy829zrf1wr1/data_gripnet.zip
-   `.\LP_baselines\data` : https://www.dropbox.com/s/1dpae6eiwtnyxwd/data_LP.zip
-   `.\NC_baselines\data` : https://www.dropbox.com/s/i2lsrlpaqsf9bbz/data_NC.zip

