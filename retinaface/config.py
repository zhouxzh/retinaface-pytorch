import copy

import timm


BASE_CFG = {
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'train_image_size': 840,
}

DEFAULT_BACKBONE = 'mobilenet'
DEFAULT_WEIGHT_PATH = 'weights/Retinaface_mobilenet0.25.pth'
TIMM_BACKBONE_PREFIXES = ('resnet', 'mobilenet')

CUSTOM_BACKBONES = {
    'mobilenet': {
        'name': 'mobilenet0.25',
        'backbone_source': 'custom',
        'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
        'in_channels_list': [64, 128, 256],
        'out_channel': 64,
        'pretrained_path': 'weights/mobilenetV1X0.25_pretrain.tar',
    },
}

BACKBONE_ALIASES = {
    'mobilenet0.25': 'mobilenet',
    'mobilenet_v1_025': 'mobilenet',
}


def normalize_backbone_name(backbone_name):
    if not backbone_name:
        return DEFAULT_BACKBONE
    normalized = backbone_name.strip().lower()
    return BACKBONE_ALIASES.get(normalized, normalized)


def get_default_weight_path(backbone_name):
    normalized = normalize_backbone_name(backbone_name)
    if normalized == 'mobilenet':
        return DEFAULT_WEIGHT_PATH
    return f'weights/retinaface_{normalized}.pth'


def _build_custom_backbone_cfg(backbone_name):
    spec = copy.deepcopy(CUSTOM_BACKBONES[backbone_name])
    cfg = copy.deepcopy(BASE_CFG)
    cfg.update(spec)
    cfg['backbone_name'] = backbone_name
    return cfg


def _build_timm_backbone_cfg(backbone_name):
    probe_model = timm.create_model(backbone_name, pretrained=False, features_only=True)
    feature_channels = list(probe_model.feature_info.channels())
    if len(feature_channels) < 3:
        raise ValueError(f'主干 {backbone_name} 的特征层不足 3 个，无法用于 RetinaFace FPN。')

    out_indices = tuple(range(len(feature_channels) - 3, len(feature_channels)))
    in_channels_list = feature_channels[-3:]
    out_channel = 64 if in_channels_list[-1] <= 256 else 256

    cfg = copy.deepcopy(BASE_CFG)
    cfg.update(
        {
            'name': backbone_name,
            'backbone_name': backbone_name,
            'backbone_source': 'timm',
            'out_indices': out_indices,
            'in_channels_list': in_channels_list,
            'out_channel': out_channel,
        }
    )
    return cfg


def get_backbone_cfg(backbone_name):
    normalized = normalize_backbone_name(backbone_name)
    if normalized in CUSTOM_BACKBONES:
        return _build_custom_backbone_cfg(normalized)

    if normalized.startswith(TIMM_BACKBONE_PREFIXES):
        return _build_timm_backbone_cfg(normalized)

    raise ValueError(
        '当前只支持自定义 mobilenet，以及 timm 中以 resnet 或 mobilenet 开头的骨架。'
    )


cfg_mnet = get_backbone_cfg('mobilenet')
cfg_re50 = get_backbone_cfg('resnet50')

