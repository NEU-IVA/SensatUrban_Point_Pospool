import numpy as np
from helper_ply import read_ply, write_ply
import matplotlib.pyplot as plt

plt.imshow(bev[0, 0:3, :, :].permute(1, 2, 0).cpu().numpy())
plt.imshow(bev[0, 3:, :, :].permute(1, 2, 0).cpu().numpy())
plt.show()

bev_ = torch.reshape(bev, (-1, 4))
bev_ = torch.index_select(bev_, 0, bind_idx)
bev__ = np.hstack([p.F.cpu().numpy(), bev_])
write_ply("bind.ply", bev__, ['x', 'y', 'z', 'r', 'g', 'b'])
