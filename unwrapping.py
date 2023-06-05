import math
import torch


height = 180
Height = 176
width = 240
dataset = 'real'

if dataset is 'real':
    fx = 190.6186
    fy = 200.7526
    cx = 116.8406
    cy = 101.4767
    para100 = [-0.7427,  -0.016, -0.006, 0.0012, -0.0007, 0.0002, 0.000154, -0.00018]
elif dataset is 'synthetic':
    fx = 210.0
    fy = 210.0
    cx = 120.0
    cy = 90.0
    para100 = [-0.7427,  -0.016, 0,  0,  0, 0, 0.000154, -0.00018]
else:
    raise NotImplementedError


def Unwrapping(self, img1, Depth, isFlatten):
    phi = torch.atan2(img1[:, 3, ...] - img1[:, 1, ...], img1[:, 0, ...] - img1[:, 2, ...]).squeeze()
    depth20 = Depth / 1000
    dis = 1.5 * phi / (2 * math.pi)
    dis[dis < 0] += 1.5
    T_Error = para100[7] * self.T
    depth100 = dis + para100[0] + para100[1] * dis + para100[2] * torch.cos(4 * self.k100 * dis) \
               + para100[3] * torch.sin(4 * self.k100 * dis) + para100[4] * torch.cos(8 * self.k100 * dis) \
               + para100[5] * torch.sin(8 * self.k100 * dis) + T_Error + para100[6] * self.R_array

    pA = depth20 / 7.5
    pB = depth100 / 1.5
    e = pA * self.MB - pB * self.MA
    nB = (self.K0 * torch.round(e)) % self.MB
    dep_mean = depth100 + 1.5 * nB * (para100[1] + 1)

    if isFlatten:
        H, W = torch.meshgrid(torch.arange(Height).cuda(), torch.arange(width).cuda())
        depth = (2 * dep_mean - 0.0007).true_divide(torch.sqrt(((W - cx) / fx) ** 2 + ((H - cy) / fy) ** 2 + 1)
                + torch.sqrt((0.007 / dep_mean - (W - cx) / fx) ** 2 + ((H - cy) / fy) ** 2 + 1))
        depth[Depth == 0] = 0
        depth *= 1000
    else:
        dep_mean[Depth == 0] = 0
        depth = dep_mean * 1000

    return depth

