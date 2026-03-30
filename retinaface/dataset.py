import multiprocessing
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from datasets import Image as HFImage
from datasets import load_dataset
from huggingface_hub import snapshot_download
from PIL import Image
from torch.utils.data import DataLoader


SPLIT_ALIASES = {
    'train': ('train',),
    'val': ('val', 'validation'),
}
TRAIN_COLUMNS = ['image', 'bboxes', 'landmarks']
DEFAULT_PREFETCH_FACTOR = 4
DEFAULT_REPO_ID = 'zhouxzh/retinaface_widerface'
DEFAULT_DOWNLOAD_ROOT = Path('data')


def preprocess_input(image):
    image -= np.array((104, 117, 123), np.float32)
    return image


def resolve_split_name(dataset_dict, canonical_split):
    for candidate in SPLIT_ALIASES[canonical_split]:
        if candidate in dataset_dict:
            return candidate
    return None


def collect_split_parquet_files(root_dir, split_name):
    nested_dir = root_dir / split_name
    if nested_dir.exists():
        nested_files = sorted(nested_dir.glob('*.parquet'))
        if nested_files:
            return nested_files

    top_level_files = sorted(root_dir.glob(f'{split_name}*.parquet'))
    if top_level_files:
        return top_level_files

    return []


def load_retinaface_parquet_dataset(repo_id, download_root=DEFAULT_DOWNLOAD_ROOT, cache_dir=None):
    return load_retinaface_parquet_dataset_with_cache(repo_id, download_root=download_root, cache_dir=cache_dir)


def resolve_download_dir(repo_id, download_root):
    download_root = Path(download_root)
    download_root.mkdir(parents=True, exist_ok=True)
    return download_root / repo_id.split('/')[-1]


def download_dataset_to_data_dir(repo_id, download_root=DEFAULT_DOWNLOAD_ROOT):
    target_dir = resolve_download_dir(repo_id, download_root)
    snapshot_dir = Path(
        snapshot_download(
        repo_id=repo_id,
        repo_type='dataset',
        allow_patterns=['*.parquet', 'README.md', '*.json', 'dataset_infos.json'],
        )
    )

    target_dir.mkdir(parents=True, exist_ok=True)
    for source_path in snapshot_dir.rglob('*'):
        if not source_path.is_file():
            continue
        relative_path = source_path.relative_to(snapshot_dir)
        destination_path = target_dir / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
    return target_dir


def resolve_cache_dir(dataset_root, cache_dir=None):
    if cache_dir:
        return Path(cache_dir)
    return None


def load_retinaface_parquet_dataset_with_cache(repo_id, download_root=DEFAULT_DOWNLOAD_ROOT, cache_dir=None):
    dataset_root = download_dataset_to_data_dir(repo_id, download_root=download_root)
    data_files = {}
    resolved_cache_dir = resolve_cache_dir(dataset_root, cache_dir)
    if resolved_cache_dir is not None:
        resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    train_files = collect_split_parquet_files(dataset_root, 'train')
    val_files = collect_split_parquet_files(dataset_root, 'val')
    validation_files = collect_split_parquet_files(dataset_root, 'validation')

    if train_files:
        data_files['train'] = [str(path) for path in train_files]
    if validation_files:
        data_files['validation'] = [str(path) for path in validation_files]
    elif val_files:
        data_files['val'] = [str(path) for path in val_files]

    if 'train' not in data_files:
        raise FileNotFoundError(f'下载后的数据集中未找到 train parquet 文件: {dataset_root}')

    dataset_dict = load_dataset(
        'parquet',
        data_files=data_files,
        cache_dir=str(resolved_cache_dir) if resolved_cache_dir is not None else None,
    )
    train_columns = dataset_dict['train'].column_names
    required_columns = [column for column in TRAIN_COLUMNS if column in train_columns]
    if len(required_columns) != len(TRAIN_COLUMNS):
        raise ValueError(
            f'train split 缺少训练所需字段，需要 {TRAIN_COLUMNS}，实际只有 {train_columns}'
        )

    dataset_dict = dataset_dict.select_columns(required_columns)
    dataset_dict = dataset_dict.cast_column('image', HFImage())
    return dataset_dict, dataset_root, resolved_cache_dir


def get_landmark_value(landmarks, key, index, default=-1.0):
    values = landmarks.get(key)
    if values is None or index >= len(values):
        return np.float32(default)
    return np.float32(values[index])


def build_annotation_array(example):
    bboxes = example['bboxes']
    landmarks = example['landmarks']
    total_boxes = len(bboxes['x'])
    annotations = np.zeros((total_boxes, 15), dtype=np.float32)

    if total_boxes == 0:
        return annotations

    for index in range(total_boxes):
        x = np.float32(bboxes['x'][index])
        y = np.float32(bboxes['y'][index])
        w = np.float32(bboxes['w'][index])
        h = np.float32(bboxes['h'][index])
        x1 = get_landmark_value(landmarks, 'x1', index)
        y1 = get_landmark_value(landmarks, 'y1', index)
        x2 = get_landmark_value(landmarks, 'x2', index)
        y2 = get_landmark_value(landmarks, 'y2', index)
        x3 = get_landmark_value(landmarks, 'x3', index)
        y3 = get_landmark_value(landmarks, 'y3', index)
        x4 = get_landmark_value(landmarks, 'x4', index)
        y4 = get_landmark_value(landmarks, 'y4', index)
        x5 = get_landmark_value(landmarks, 'x5', index)
        y5 = get_landmark_value(landmarks, 'y5', index)

        annotations[index, 0] = x
        annotations[index, 1] = y
        annotations[index, 2] = x + w
        annotations[index, 3] = y + h
        annotations[index, 4] = x1
        annotations[index, 5] = y1
        annotations[index, 6] = x2
        annotations[index, 7] = y2
        annotations[index, 8] = x3
        annotations[index, 9] = y3
        annotations[index, 10] = x4
        annotations[index, 11] = y4
        annotations[index, 12] = x5
        annotations[index, 13] = y5
        annotations[index, 14] = -1.0 if x1 < 0 else 1.0

    return annotations


class RetinaFaceTrainTransform:
    def __init__(self, img_size):
        self.img_size = img_size

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, targets, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4):
        iw, ih = image.size
        h, w = input_shape
        box = targets.copy()

        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 3.25)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data = np.array(image, np.uint8)
        color_scale = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue_channel, sat_channel, val_channel = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        lookup = np.arange(0, 256, dtype=color_scale.dtype)
        lut_hue = ((lookup * color_scale[0]) % 180).astype(dtype)
        lut_sat = np.clip(lookup * color_scale[1], 0, 255).astype(dtype)
        lut_val = np.clip(lookup * color_scale[2], 0, 255).astype(dtype)
        image_data = cv2.merge((cv2.LUT(hue_channel, lut_hue), cv2.LUT(sat_channel, lut_sat), cv2.LUT(val_channel, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2, 4, 6, 8, 10, 12]] = box[:, [0, 2, 4, 6, 8, 10, 12]] * nw / iw + dx
            box[:, [1, 3, 5, 7, 9, 11, 13]] = box[:, [1, 3, 5, 7, 9, 11, 13]] * nh / ih + dy
            if flip:
                box[:, [0, 2, 4, 6, 8, 10, 12]] = w - box[:, [2, 0, 6, 4, 8, 12, 10]]
                box[:, [5, 7, 9, 11, 13]] = box[:, [7, 5, 9, 13, 11]]

            center_x = (box[:, 0] + box[:, 2]) / 2
            center_y = (box[:, 1] + box[:, 3]) / 2
            box = box[np.logical_and(np.logical_and(center_x > 0, center_y > 0), np.logical_and(center_x < w, center_y < h))]

            box[:, 0:14][box[:, 0:14] < 0] = 0
            box[:, [0, 2, 4, 6, 8, 10, 12]][box[:, [0, 2, 4, 6, 8, 10, 12]] > w] = w
            box[:, [1, 3, 5, 7, 9, 11, 13]][box[:, [1, 3, 5, 7, 9, 11, 13]] > h] = h

            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        if len(box) > 0:
            box[:, 4:-1][box[:, -1] == -1] = 0
            box[:, [0, 2, 4, 6, 8, 10, 12]] /= w
            box[:, [1, 3, 5, 7, 9, 11, 13]] /= h
        return image_data, box.astype(np.float32, copy=False)

    def transform_example(self, example):
        image = example['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')

        target = build_annotation_array(example)
        image_data, target = self.get_random_data(image, target, [self.img_size, self.img_size])
        image_tensor = torch.from_numpy(
            np.transpose(preprocess_input(np.asarray(image_data, dtype=np.float32)), (2, 0, 1)).copy()
        )
        target_tensor = torch.from_numpy(target.copy())
        return image_tensor, target_tensor

    def __call__(self, examples):
        if isinstance(examples['image'], list):
            images = []
            targets = []
            for index in range(len(examples['image'])):
                sample = {key: value[index] for key, value in examples.items()}
                image_tensor, target_tensor = self.transform_example(sample)
                images.append(image_tensor)
                targets.append(target_tensor)
            return {'image': images, 'target': targets}

        image_tensor, target_tensor = self.transform_example(examples)
        return {'image': image_tensor, 'target': target_tensor}


class RetinaFaceEvalTransform:
    def __init__(self, img_size):
        self.img_size = img_size

    def resize_with_letterbox(self, image, targets, input_shape):
        iw, ih = image.size
        h, w = input_shape
        scale = min(w / float(iw), h / float(ih))
        nw = max(int(iw * scale), 1)
        nh = max(int(ih * scale), 1)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        resized_image = image.resize((nw, nh), Image.BICUBIC)
        canvas = Image.new('RGB', (w, h), (128, 128, 128))
        canvas.paste(resized_image, (dx, dy))
        image_data = np.array(canvas, np.uint8)
        box = targets.copy()

        if len(box) > 0:
            box[:, [0, 2, 4, 6, 8, 10, 12]] = box[:, [0, 2, 4, 6, 8, 10, 12]] * scale + dx
            box[:, [1, 3, 5, 7, 9, 11, 13]] = box[:, [1, 3, 5, 7, 9, 11, 13]] * scale + dy

            box[:, 0:14][box[:, 0:14] < 0] = 0
            box[:, [0, 2, 4, 6, 8, 10, 12]][box[:, [0, 2, 4, 6, 8, 10, 12]] > w] = w
            box[:, [1, 3, 5, 7, 9, 11, 13]][box[:, [1, 3, 5, 7, 9, 11, 13]] > h] = h

            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        if len(box) > 0:
            box[:, 4:-1][box[:, -1] == -1] = 0
            box[:, [0, 2, 4, 6, 8, 10, 12]] /= w
            box[:, [1, 3, 5, 7, 9, 11, 13]] /= h

        return image_data, box.astype(np.float32, copy=False)

    def transform_example(self, example):
        image = example['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')

        target = build_annotation_array(example)
        image_data, target = self.resize_with_letterbox(image, target, [self.img_size, self.img_size])
        image_tensor = torch.from_numpy(
            np.transpose(preprocess_input(np.asarray(image_data, dtype=np.float32)), (2, 0, 1)).copy()
        )
        target_tensor = torch.from_numpy(target.copy())
        return image_tensor, target_tensor

    def __call__(self, examples):
        if isinstance(examples['image'], list):
            images = []
            targets = []
            for index in range(len(examples['image'])):
                sample = {key: value[index] for key, value in examples.items()}
                image_tensor, target_tensor = self.transform_example(sample)
                images.append(image_tensor)
                targets.append(target_tensor)
            return {'image': images, 'target': targets}

        image_tensor, target_tensor = self.transform_example(examples)
        return {'image': image_tensor, 'target': target_tensor}


def detection_collate(batch):
    images = []
    targets = []
    for sample in batch:
        image = sample['image']
        target = sample['target']
        if target.numel() == 0:
            continue
        images.append(image)
        targets.append(target)

    if not images:
        return torch.empty((0, 3, 0, 0), dtype=torch.float32), []

    return torch.stack(images, dim=0), targets


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_worker_start_method(requested_method):
    if requested_method in {None, 'default'}:
        return None

    if requested_method == 'auto':
        available_methods = multiprocessing.get_all_start_methods()
        for candidate in ('forkserver', 'spawn', 'fork'):
            if candidate in available_methods:
                return candidate
        return None

    return requested_method


def create_dataloader(dataset, batch_size, shuffle, drop_last, num_workers, pin_memory, seed, prefetch_factor, worker_start_method):
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataloader_kwargs = {
        'dataset': dataset,
        'shuffle': shuffle,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': drop_last,
        'collate_fn': detection_collate,
        'generator': generator,
    }

    if num_workers > 0:
        dataloader_kwargs['persistent_workers'] = True
        dataloader_kwargs['prefetch_factor'] = prefetch_factor
        dataloader_kwargs['worker_init_fn'] = seed_worker
        resolved_start_method = resolve_worker_start_method(worker_start_method)
        if resolved_start_method is not None:
            dataloader_kwargs['multiprocessing_context'] = resolved_start_method

    return DataLoader(**dataloader_kwargs)


def create_train_val_dataloaders(
    repo_id,
    img_size,
    batch_size,
    download_root=DEFAULT_DOWNLOAD_ROOT,
    val_batch_size=None,
    num_workers=0,
    pin_memory=False,
    seed=3407,
    prefetch_factor=DEFAULT_PREFETCH_FACTOR,
    worker_start_method='auto',
    cache_dir=None,
):
    dataset_dict, dataset_root, resolved_cache_dir = load_retinaface_parquet_dataset_with_cache(
        repo_id,
        download_root=download_root,
        cache_dir=cache_dir,
    )
    train_split = resolve_split_name(dataset_dict, 'train')
    val_split = resolve_split_name(dataset_dict, 'val')
    if train_split is None:
        raise ValueError(f'未找到 train split，可用 splits: {list(dataset_dict.keys())}')

    val_batch_size = val_batch_size or batch_size
    train_dataset = dataset_dict[train_split].with_transform(RetinaFaceTrainTransform(img_size))
    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        prefetch_factor=prefetch_factor,
        worker_start_method=worker_start_method,
    )

    val_dataset = None
    val_loader = None
    if val_split is not None:
        val_dataset = dataset_dict[val_split].with_transform(RetinaFaceEvalTransform(img_size))
        val_loader = create_dataloader(
            dataset=val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed + 1,
            prefetch_factor=prefetch_factor,
            worker_start_method=worker_start_method,
        )

    print(
        f'使用 Hugging Face 远程数据集训练: repo={repo_id}, local_dir={dataset_root} '
        f'(cache={resolved_cache_dir if resolved_cache_dir is not None else "huggingface-default"}, train={len(train_dataset)}, val={len(val_dataset) if val_dataset is not None else 0})'
    )
    return dataset_root, train_dataset, train_loader, val_dataset, val_loader


def create_train_dataloader(
    repo_id,
    img_size,
    batch_size,
    download_root=DEFAULT_DOWNLOAD_ROOT,
    num_workers=0,
    pin_memory=False,
    seed=3407,
    prefetch_factor=DEFAULT_PREFETCH_FACTOR,
    worker_start_method='auto',
    cache_dir=None,
):
    dataset_dict, dataset_root, resolved_cache_dir = load_retinaface_parquet_dataset_with_cache(
        repo_id,
        download_root=download_root,
        cache_dir=cache_dir,
    )
    train_split = resolve_split_name(dataset_dict, 'train')
    if train_split is None:
        raise ValueError(f'未找到 train split，可用 splits: {list(dataset_dict.keys())}')

    train_dataset = dataset_dict[train_split].with_transform(RetinaFaceTrainTransform(img_size))
    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        prefetch_factor=prefetch_factor,
        worker_start_method=worker_start_method,
    )
    print(
        f'使用 Hugging Face 远程数据集训练: repo={repo_id}, local_dir={dataset_root} '
        f'(cache={resolved_cache_dir if resolved_cache_dir is not None else "huggingface-default"}, train={len(train_dataset)})'
    )
    return train_dataset, train_loader