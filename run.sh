#!/usr/bin/env bash

# on Pose-0/1/2
# python GripNet-pose.py {epoch} {dataset_idx} {use_checkout}
python GripNet-pose.py 50 0 1
python GripNet-pose.py 50 1 0
python GripNet-pose.py 50 2 1

# on aminer
# python Gripnet-aminer.py {epoch} {split_idx}
python GripNet-aminer.py 200 1

# on freebase-a/b/c/d
# python Gripnet-freebase-{dataset_idx}.py {epoch} {split_idx}
python GripNet-freebase-a.py 100 2
python GripNet-freebase-b.py 100 2

python GripNet-freebase-c.py 30 2
python GripNet-freebase-d.py 30 2
