import torch
import matplotlib.pyplot as plt
import numpy as np

mip = torch.load('out/pose-1/100[64, 32, 16]-32-16-16-[48, 16]-record.pt')
auroc_mip = [mip.test_out[i][1] for i in range(100)]

dmt = torch.load('out/multi-rela/pose_dmt_all/100-32-0.01-record.pt')
auroc_dmt = [dmt.test_out[i][1] for i in range(100)]

dmt_org = torch.load("out/multi-rela/pose_dmt_org/100-32-0.01-record.pt")
auroc_dmt_org = [dmt_org.test_out[i][1] for i in range(100)]

rgcn_org = torch.load('out/multi-rela/pose_rgcn_org/100-64-32-16-16-0.01-record.pt')
auroc_rgcn_org = [rgcn_org.test_out[i][1] for i in range(100)]

rgcn = torch.load('out/multi-rela/pose_rgcn_all/100-64-32-16-16-0.01-record.pt')
auroc_rgcn = [rgcn.test_out[i][1] for i in range(100)]

fig = plt.figure(1, figsize=(10, 3))

plt.subplot(121)
x = np.array(range(100), dtype=int) + 1
plt.plot(x, auroc_mip, label='GripNet', color='y')
plt.plot(x, auroc_dmt_org, label='DistMult', color=(1, 0, 1))
plt.plot(x, auroc_rgcn_org, label='RGCN', color=(0, 1, 1))
plt.plot(x, auroc_dmt, label='DistMult++', color='r')
plt.plot(x, auroc_rgcn, label='RGCN++', color='b')
plt.title('AUROC by epoch')
plt.ylim((0.45, 0.95))
# plt.legend()
plt.grid()

ax = plt.subplot(122)
tmp = [i for i in (x*21) if i < 1000]
plt.plot(tmp, auroc_mip[:len(tmp)], label='$GripNet$', color='y')
tmp = [i for i in (x*11.7) if i < 1000]
plt.plot(tmp, auroc_dmt[:len(tmp)], label='$DistMult^{TNG}$', color='r')
plt.plot(tmp, auroc_dmt_org[:len(tmp)], label='$DistMult$', color=(1, 0, 1))

tmp = [i for i in (x*59.6) if i < 1000]
plt.plot(tmp, auroc_rgcn[:len(tmp)], label='$RGCN^{TNG}$', color='b')
plt.plot(tmp, auroc_rgcn_org[:len(tmp)], label='$RGCN$', color=(0, 1, 1))

plt.title('AUROC by time (s)')
plt.ylim((0.45, 0.95))
# plt.legend()
plt.grid()

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.7))
# plt.subplot(133)
# plt.scatter(19.3, 12.22, label='GripNet', color='y', marker='P')
# plt.scatter(59.6, 10.92, label='RGCN', color=(1, 0, 1), marker='D')
# plt.scatter(59.6, 10.94, label='RGCN++', color='b', marker='*')
# plt.scatter(11.7, 9.3, label='DistMult', color=(0, 1, 1), marker='o')
# plt.scatter(11.7, 9.31, label='DistMult++', color='r', marker='1')
# plt.title('Peak GPU memory usage (GB) by time (s)')

plt.savefig('./fig/pose_auroc1.png')
plt.show()





