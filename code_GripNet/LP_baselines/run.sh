#!/usr/bin/env bash

python baselines.py 0 TransE 2 24
python baselines.py 0 DistMult 2 24
python baselines.py 0 ComplEx 2 24
python baselines.py 0 RotatE 2 24
python rgcn_pose.py 0 RGCN 2 24

