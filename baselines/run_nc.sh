#!/usr/bin/env bash

# -i input dataset file
# -o output directory
# -n output file name

# GAT model
python NC_baselines/GAT.py -i datasets_baselines/aminer.pt -o out/aminer/ -n gat-32.txt
python NC_baselines/GAT.py -i datasets_baselines/freebase-a.pt -o out/freebase-a/ -n gat-32.txt
python NC_baselines/GAT.py -i datasets_baselines/freebase-b.pt -o out/freebase-b/ -n gat-32.txt
python NC_baselines/GAT.py -i datasets_baselines/freebase-c.pt -o out/freebase-c/ -n gat-32.txt
python NC_baselines/GAT.py -i datasets_baselines/freebase-d.pt -o out/freebase-d/ -n gat-32.txt

# GCN model
python NC_baselines/GCN_MLP.py -i datasets_baselines/aminer.pt -o out/aminer/ -n gcn-32.txt
python NC_baselines/GCN_MLP.py -i datasets_baselines/freebase-a.pt -o out/freebase-a/ -n gcn-32.txt
python NC_baselines/GCN_MLP.py -i datasets_baselines/freebase-b.pt -o out/freebase-b/ -n gcn-32.txt
python NC_baselines/GCN_MLP.py -i datasets_baselines/freebase-c.pt -o out/freebase-c/ -n gcn-32.txt
python NC_baselines/GCN_MLP.py -i datasets_baselines/freebase-d.pt -o out/freebase-d/ -n gcn-32.txt

# RGCN model
python NC_baselines/RGCN_MLP.py -i datasets_baselines/aminer.pt -o out/aminer/ -n rgcn-32.txt
python NC_baselines/RGCN_MLP.py -i datasets_baselines/freebase-a.pt -o out/freebase-a/ -n rgcn-32.txt
python NC_baselines/RGCN_MLP.py -i datasets_baselines/freebase-b.pt -o out/freebase-b/ -n rgcn-32.txt
python NC_baselines/RGCN_MLP.py -i datasets_baselines/freebase-c.pt -o out/freebase-c/ -n rgcn-32.txt
python NC_baselines/RGCN_MLP.py -i datasets_baselines/freebase-d.pt -o out/freebase-d/ -n rgcn-32.txt
