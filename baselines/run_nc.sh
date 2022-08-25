#!/usr/bin/env bash

python NC_baselines/gat.py -i ../datasets_databases/aminer.pt -o result/aminer/gat-32.txt
python NC_baselines/gat.py -i ../datasets_databases/freebase-a.pt -o result/freebase-a/gat-32.txt
python NC_baselines/gat.py -i ../datasets_databases/freebase-b.pt -o result/freebase-b/gat-32.txt
python NC_baselines/gat.py -i ../datasets_databases/freebase-c.pt -o result/freebase-c/gat-32.txt
python NC_baselines/gat.py -i ../datasets_databases/freebase-d.pt -o result/freebase-d/gat-32.txt

python NC_baselines/rgcn_mlp.py -i ../datasets_databases/aminer.pt -o result/aminer/rgcn-32.txt
python NC_baselines/rgcn_mlp.py -i ../datasets_databases/freebase-a.pt -o result/freebase-a/rgcn-32.txt
python NC_baselines/rgcn_mlp.py -i ../datasets_databases/freebase-b.pt -o result/freebase-b/rgcn-32.txt
python NC_baselines/rgcn_mlp.py -i ../datasets_databases/freebase-c.pt -o result/freebase-c/rgcn-32.txt
python NC_baselines/rgcn_mlp.py -i ../datasets_databases/freebase-d.pt -o result/freebase-d/rgcn-32.txt
