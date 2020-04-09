import torch
import matplotlib.pyplot as plt
import numpy as np

mip = torch.load('out/pose/100[64, 32, 16]-32-16-16-[48, 16]-record.pt')
auroc_mip = [mip.test_out[i][1] for i in range(100)]

dmt = torch.load('out/pose_dmt_all/100-32-0.01-record.pt')
auroc_dmt = [dmt.test_out[i][1] for i in range(100)]

dmt_org = torch.load("out/pose_dmt_org/100-32-0.01-record.pt")
auroc_dmt_org = [dmt_org.test_out[i][1] for i in range(100)]

rgcn_org = torch.load('/Users/nyxfer/Docu/MIP/out/pose_rgcn_org/100-64-32-16-16-0.01-record.pt')
auroc_rgcn_org = [rgcn_org.test_out[i][1] for i in range(100)]

rgcn = torch.load('out/pose_rgcn_all/100-64-32-16-16-0.01-record.pt')
auroc_rgcn = [rgcn.test_out[i][1] for i in range(100)]

plt.figure(1)

plt.subplot(121)
x = np.array(range(100), dtype=int) + 1
plt.plot(x, auroc_mip, label='MIP', color='y')
plt.plot(x, auroc_dmt_org, label='DistMult', color=(1, 0, 1))
plt.plot(x, auroc_rgcn_org, label='RGCN', color=(0, 1, 1))
plt.plot(x, auroc_dmt, label='DistMult++', color='r')
plt.plot(x, auroc_rgcn, label='RGCN++', color='b')
plt.title('AUROC by epoch')
plt.legend()
plt.grid()

plt.subplot(122)
tmp = [i for i in (x*21) if i < 1000]
plt.plot(tmp, auroc_mip[:len(tmp)], label='MIP', color='y')
tmp = [i for i in (x*11.7) if i < 1000]
plt.plot(tmp, auroc_dmt[:len(tmp)], label='DistMult++', color='r')
plt.plot(tmp, auroc_dmt_org[:len(tmp)], label='DistMult', color=(1, 0, 1))

tmp = [i for i in (x*59.6) if i < 1000]
plt.plot(tmp, auroc_rgcn[:len(tmp)], label='RGCN++', color='b')
plt.plot(tmp, auroc_rgcn_org[:len(tmp)], label='RGCN', color=(0, 1, 1))

plt.title('AUROC by time')
plt.legend()
plt.grid()

plt.savefig('./fig/pose_auroc.png')
plt.show()





