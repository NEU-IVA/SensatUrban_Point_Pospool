from typing import Tuple
import torch
import torch.nn as nn
import torchsparse.nn as spnn
from torchpack.utils.config import configs
from torchsparse import PointTensor, SparseTensor
from torchpack import distributed as dist

from core.dataset.sensatUrban_crop import SensatUrban_crop
from core.modules.utils import initial_voxelize


class CBN2d(nn.Module):
    """ Conv2d + BacthNorm2d + NoLinear
    """

    def __init__(
            self, in_channels, out_channels: int, kernel_size=5, stride=1, padding=2, bias=True, no_linear=nn.ReLU(),
            bn=True):
        super(CBN2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        self.with_bn = bn
        if self.with_bn == True:
            self.bn = nn.BatchNorm2d(out_channels)

        self.no_linear = no_linear

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.with_bn:
            outputs = self.bn(outputs)

        if isinstance(self.no_linear, nn.Module):
            outputs = self.no_linear(outputs)

        return outputs


class GlobalContextPooling(nn.Module):
    def __init__(self, grid_size, spare_in_channels, bev_in_channels, gcp_depth=[6, 6], gcp_width=[128, 256]) -> None:
        super(GlobalContextPooling, self).__init__()

        self._grid_size = grid_size
        self.spare_in_channels = spare_in_channels

        fusion_in_channels = spare_in_channels * grid_size[-1] + bev_in_channels

        layer_in_channel = fusion_in_channels

        self.fusion_conv = CBN2d(layer_in_channel, gcp_width[0], kernel_size=1, padding=0)

        self.conv_encoder = nn.Sequential(
            CBN2d(gcp_width[0], gcp_width[0], stride=1, kernel_size=1, no_linear=None, bn=False, padding=0),
            *[CBN2d(gcp_width[0], gcp_width[0]) for c in range(gcp_depth[0] - 1)]
        )
        layer_in_channel = gcp_width[0]

        self.down_layer = CBN2d(layer_in_channel, gcp_width[1], stride=2)

        self.conv_decoder = nn.Sequential(
            CBN2d(gcp_width[1], gcp_width[1], stride=1, kernel_size=1, no_linear=None, bn=False, padding=0),
            *[CBN2d(gcp_width[1], gcp_width[1]) for c in range(gcp_depth[1] - 1)]
        )

        self.up_layer = nn.Sequential(
            nn.ConvTranspose2d(gcp_width[1], gcp_width[1], kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(gcp_width[1]),
            nn.ReLU()
        )

        self.out_layer = CBN2d(gcp_width[1] + gcp_width[0], layer_in_channel, kernel_size=1, padding=0)

    def _sparse2dense(self, sparse_voxel: SparseTensor) -> torch.Tensor:
        """
        Return:
            torch.Tensor: with shape(B, C, H, W)
        """
        weight, height, depth = self._grid_size[:3]

        feat = sparse_voxel.F
        coord = sparse_voxel.C
        x, y, z, b = coord.unbind(-1)

        batch = b.max() + 1

        dense_map = feat.new_zeros([batch, height, weight, depth, feat.shape[-1]]).view(-1, feat.shape[-1])
        dense_indice = b * (weight * height * depth) + y * (weight * depth) + x * depth + z
        dense_map[dense_indice.long()] = feat

        dense_map = dense_map.view(batch, height, weight, depth * feat.shape[-1])
        dense_map = dense_map.permute(0, 3, 1, 2)

        return dense_map

    def _dense2sparse(self, dense_map: torch.Tensor, sparse_voxel: SparseTensor) -> SparseTensor:
        weight, height, depth = self._grid_size[:3]

        coord = sparse_voxel.C
        x, y, z, b = coord.unbind(-1)

        dense_indice = b * (weight * height) + y * weight + x
        dense_map = dense_map.permute(0, 2, 3, 1)  # [B, H, W, C]
        dense_map = dense_map.view(-1, dense_map.shape[-1])

        sparse_feat = dense_map[dense_indice.long()]
        sparse_voxel.F = sparse_feat
        return sparse_voxel

    def _gcpblock(self, dense_map: torch.Tensor, bev_map: torch.Tensor):
        assert dense_map.shape[0] == bev_map.shape[0], 'batch size must be same'
        bev_map = torch.nn.functional.interpolate(bev_map, size=self._grid_size[:2], mode='bilinear',
                                                  align_corners=True)
        assert bev_map.shape[2:] == dense_map.shape[2:]

        fusion_map = torch.cat([dense_map, bev_map], dim=1)
        fusion_map = self.fusion_conv(fusion_map)

        out_encoder = self.conv_encoder(fusion_map)
        out_decoder = self.down_layer(out_encoder)
        out_decoder = self.conv_decoder(out_decoder)
        out_decoder = self.up_layer(out_decoder)

        out = torch.cat([out_encoder, out_decoder], dim=1)

        out = self.out_layer(out)
        return out

    def forward(self, sparse_voxel: SparseTensor, bev_feat_batch: torch.Tensor) -> Tuple[SparseTensor, torch.Tensor]:
        """ GCP模块的forward函数
            Args:
                sparse_voxel(Tensor):
                    稀疏体素类型，里面有两个参数C和F，C是int型的tensor，维度为[N,4]分别是xyz和batch_idx号，代表体素的空间编号
                    F是float类型的tensor，维度为[N,C]，代表特征
                bev_feat_batch(Tensor):
                    BEV分支的特征，维度为[B,C,W,H]

            Returns:
                fused_sparse_voxel(SparseTensor):
                    融合后的稀疏体素，shape与输入保持一致
                fused_dense_feat(Tensor):
                    融合后的BEV特征，shape与输入保持一致
        """
        # bev_feat_batch = bev_feat_batch.permute(0, 1, 3, 2).contiguous()

        dense_map = self._sparse2dense(sparse_voxel)
        fusion_dense_map = self._gcpblock(dense_map, bev_feat_batch)
        sparse_voxel = self._dense2sparse(fusion_dense_map, sparse_voxel)

        # fusion_dense_map = fusion_dense_map.permute(0, 1, 3, 2).contiguous()
        return sparse_voxel, fusion_dense_map


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    configs.load("../../configs/default.yaml", recursive=True)
    model = GlobalContextPooling([500, 500, 20], 3, 3).cuda()

    datasets = SensatUrban_crop(num_points=configs.dataset.num_points, voxel_size=configs.train.pres,
                                dataset_root=configs.dataset.root)
    dataflow = {}

    for split in datasets:
        sampler = torch.utils.data.distributed.DistributedSampler(
            datasets[split],
            num_replicas=dist.size(),
            rank=dist.rank(),
            shuffle=(split == 'train'))
        dataflow[split] = torch.utils.data.DataLoader(
            datasets[split],
            batch_size=1,
            sampler=sampler,
            num_workers=configs.workers_per_gpu,
            pin_memory=True,
            collate_fn=datasets[split].collate_fn)

    for batch_idx, feed_dict in enumerate(dataflow['train']):
        # _inputs = {}
        # for key, value in feed_dict.items():
        #     print(key)
        #     if key != 'pc_num':
        #         print(key)
        #         _inputs[key] = value.cuda()
        #     else:
        #         print("continue")
        #         continue
        inputs = {'lidar': feed_dict['lidar'].cuda(), 'bev': feed_dict['bev'].cuda(), 'alt': feed_dict['alt'].cuda(),
                  'pc_num': feed_dict['pc_num'].numpy().tolist()}

        print("net have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))

        x = inputs['lidar']
        p = PointTensor(x.F, x.C.float())
        v0 = initial_voxelize(p, configs.train.pres, configs.train.vres)
        b = inputs['bev'] / 255.0
        b = b.permute(0, 3, 1, 2)
        y = model(v0, b)

        print(y)
        exit(0)
