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

fig = plt.figure(1, figsize=(15, 3))

plt.subplot(141)
x = np.array(range(100), dtype=int) + 1
plt.plot(x, auroc_mip, label='GripNet', color='black')
plt.plot(x, auroc_dmt_org, label='DistMult', color=(1, 0, 1))
plt.plot(x, auroc_rgcn_org, label='RGCN', color='c')
plt.plot(x, auroc_dmt, label='DistMult++', color='r')
plt.plot(x, auroc_rgcn, label='RGCN++', color='b')
# plt.title('a. AUROC by epoch on PoSE-1')
plt.ylim((0.45, 0.95))
# plt.legend()
plt.grid()
plt.xlabel('Training epochs')
plt.ylabel('AUROC')

ax = plt.subplot(142)
tmp = [i for i in (x*21) if i < 1000]
plt.plot(tmp, auroc_mip[:len(tmp)], label='$GripNet$', color='black')
tmp = [i for i in (x*11.7) if i < 1000]
plt.plot(tmp, auroc_dmt[:len(tmp)], label='$DistMult^{TNG}$', color='r')
plt.plot(tmp, auroc_dmt_org[:len(tmp)], label='$DistMult$', color=(1, 0, 1))

tmp = [i for i in (x*59.6) if i < 1000]
plt.plot(tmp, auroc_rgcn[:len(tmp)], label='$RGCN^{TNG}$', color='b')
plt.plot(tmp, auroc_rgcn_org[:len(tmp)], label='$RGCN$', color='c')

# plt.title('b. AUROC by time (s) on PoSE-1')
plt.ylim((0.45, 0.95))
# plt.legend()
plt.grid()
plt.xlabel('Training time (seconds)')
plt.ylabel('AUROC')

# Shrink current axis by 20%
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width, box.height])

# Put a legend to the right of the current axis
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)

plt.subplot(143)

c = ['y', 'g', 'c', 'black']
row_head = ['GCN', 'GAT', 'RGCN', 'GripNet']
# col_head = ['Freebase-a', 'Freebase-b', 'Freebase-c', 'Freebase-d']
# col_head = np.array([12556, 848032, 1044814, 3458696])
col_head = [1, 2, 3, 4]
data = np.array([[0.309, 0.430, 0.508, 0.574],
                 [0.341, 0.454, 0.498, 0.564],
                 [0.300, 0.365, 0.464, 0.506],
                 [0.300, 0.476, 0.563, 0.597]])
for i in range(len(row_head)):
    plt.plot(col_head, data[i], label=row_head[i], color=c[i], alpha=0.8)
# plt.legend()
# plt.title('c. Micro-F1 by data integration')
plt.grid()
plt.xlabel('Number of node types')
plt.ylabel('Micro-F1')
plt.xticks(col_head)
# plt.xscale('log')

plt.subplot(144)
col_head = np.array([14989, 354961, 457504, 1100400])
row_head = ['GCN', 'GAT', 'RGCN', 'GripNet']

for i in range(len(row_head)):
    plt.plot(col_head, data[i], label=row_head[i], color=c[i], alpha=0.8)
# plt.legend()
# plt.title('c. Micro-F1 by data integration')
plt.grid()
plt.xscale('log')
plt.xlabel('Number of nodes')
plt.ylabel('Micro-F1')

plt.savefig('./fig/pose_auroc1.png')
plt.show()





