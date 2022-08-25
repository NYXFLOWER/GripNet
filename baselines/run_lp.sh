#!/usr/bin/env bash
# Link prediction on Pose-0/1/2 dataset

# inproved DistMult and RGCN
# python dmt_pose.py {epoch} {dataset_idx}
python LP_baselines/dmt_pose.py 100 0

python LP_baselines/rgcn_pose.py 100 0

# Other baselines
# python TransE_DistMult_ComplEx_RotatE.py {epoch} {dataset_idx} {model}
python LP_baselines/TransE_DistMult_ComplEx_RotatE.py 100 0 TransE
TransE_DistMult_ComplEx_RotatE.py 100 0 DistMult
TransE_DistMult_ComplEx_RotatE.py 100 0 ComplEx
TransE_DistMult_ComplEx_RotatE.py 100 0 RotatE
