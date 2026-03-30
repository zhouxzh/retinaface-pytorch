import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.models._utils as _utils

from retinaface.blocks import FPN, MobileNetV1, SSH


class ClassHead(nn.Module):
    def __init__(self, in_channels=512, num_anchors=2):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, in_channels=512, num_anchors=2):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, in_channels=512, num_anchors=2):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 10, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):
    def __init__(self, cfg=None, pretrained=False, mode='train'):
        super().__init__()
        if cfg['backbone_source'] == 'custom':
            backbone = MobileNetV1()
            if pretrained:
                checkpoint = torch.load(cfg['pretrained_path'], map_location=torch.device('cpu'))
                new_state_dict = {}
                for key, value in checkpoint['state_dict'].items():
                    new_state_dict[key[7:]] = value
                backbone.load_state_dict(new_state_dict)
            self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        elif cfg['backbone_source'] == 'timm':
            self.body = timm.create_model(
                cfg['name'],
                pretrained=pretrained,
                features_only=True,
                out_indices=cfg['out_indices'],
            )
        else:
            raise ValueError(f"Unsupported backbone config: {cfg['name']}")

        self.fpn = FPN(cfg['in_channels_list'], cfg['out_channel'])
        self.ssh1 = SSH(cfg['out_channel'], cfg['out_channel'])
        self.ssh2 = SSH(cfg['out_channel'], cfg['out_channel'])
        self.ssh3 = SSH(cfg['out_channel'], cfg['out_channel'])
        self.ClassHead = self._make_class_head(fpn_num=3, in_channels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, in_channels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, in_channels=cfg['out_channel'])
        self.mode = mode

    def _make_class_head(self, fpn_num=3, in_channels=64, anchor_num=2):
        return nn.ModuleList([ClassHead(in_channels, anchor_num) for _ in range(fpn_num)])

    def _make_bbox_head(self, fpn_num=3, in_channels=64, anchor_num=2):
        return nn.ModuleList([BboxHead(in_channels, anchor_num) for _ in range(fpn_num)])

    def _make_landmark_head(self, fpn_num=3, in_channels=64, anchor_num=2):
        return nn.ModuleList([LandmarkHead(in_channels, anchor_num) for _ in range(fpn_num)])

    def forward(self, inputs):
        features = self.body(inputs)
        pyramid = self.fpn(features)
        feature1 = self.ssh1(pyramid[0])
        feature2 = self.ssh2(pyramid[1])
        feature3 = self.ssh3(pyramid[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.mode == 'train':
            return bbox_regressions, classifications, ldm_regressions
        return bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions