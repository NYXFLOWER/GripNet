#!/usr/bin/env bash

python grip-pose.py 0 2 16
python grip_freebase_b.py 100 512 64 64 64 128 64 32 0
python grip_freebase_c_d.py 100 256 128 128 256 128 128 128 128 128 32