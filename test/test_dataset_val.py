import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from datasets import load_dataset

from retinaface.dataset import collect_split_parquet_files, download_dataset_to_data_dir


DEFAULT_REPO_ID = 'zhouxzh/retinaface_widerface'
DEFAULT_DOWNLOAD_ROOT = 'data'
DEFAULT_OUTPUT_DIR = 'outputs/val_dataset_check'


def load_val_dataset(repo_id, download_root='data', cache_dir=''):
    dataset_root = download_dataset_to_data_dir(repo_id, download_root=download_root)
    validation_files = collect_split_parquet_files(dataset_root, 'validation')
    val_files = collect_split_parquet_files(dataset_root, 'val')
    if validation_files:
        split_name = 'validation'
        data_files = {'validation': [str(path) for path in validation_files]}
    elif val_files:
        split_name = 'val'
        data_files = {'val': [str(path) for path in val_files]}
    else:
        raise FileNotFoundError(f'下载后的数据集中未找到 val/validation parquet 文件: {dataset_root}')

    dataset_dict = load_dataset('parquet', data_files=data_files, cache_dir=cache_dir or None)
    dataset = dataset_dict[split_name]
    required_columns = {'image', 'bboxes'}
    missing_columns = sorted(required_columns - set(dataset.column_names))
    if missing_columns:
        raise ValueError(f'val split 缺少必要字段: {missing_columns}，实际字段为 {dataset.column_names}')
    return dataset, dataset_root, split_name


def draw_bboxes(sample):
    image = sample['image']
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_array = np.array(image, dtype=np.uint8)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    bboxes = sample['bboxes']
    total_boxes = len(bboxes['x'])
    for index in range(total_boxes):
        x = int(round(float(bboxes['x'][index])))
        y = int(round(float(bboxes['y'][index])))
        w = int(round(float(bboxes['w'][index])))
        h = int(round(float(bboxes['h'][index])))
        x2 = x + w
        y2 = y + h
        cv2.rectangle(image_array, (x, y), (x2, y2), (0, 255, 0), 2)

    label = f'bboxes={total_boxes}'
    cv2.putText(image_array, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return image_array


def build_output_name(sample, sample_index):
    image_path = sample.get('image_path')
    if image_path:
        safe_name = image_path.replace('/', '__')
        return f'{sample_index:02d}_{safe_name}'
    return f'{sample_index:02d}_sample.jpg'


def main():
    parser = argparse.ArgumentParser(description='随机抽取 HF val 数据集样本并绘制 bbox')
    parser.add_argument('--repo_id', type=str, default=DEFAULT_REPO_ID, help='Hugging Face 数据集仓库名')
    parser.add_argument('--download_root', type=str, default=DEFAULT_DOWNLOAD_ROOT, help='远程数据集下载到本地的根目录')
    parser.add_argument('--cache_dir', type=str, default='', help='datasets cache 目录；默认使用 Hugging Face 默认缓存位置')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, help='可视化结果保存目录')
    parser.add_argument('--num_samples', type=int, default=10, help='随机抽样数量，默认 10')
    parser.add_argument('--seed', type=int, default=3407, help='随机种子，默认 3407')
    args = parser.parse_args()

    dataset, dataset_root, split_name = load_val_dataset(
        repo_id=args.repo_id,
        download_root=args.download_root,
        cache_dir=args.cache_dir,
    )
    if len(dataset) == 0:
        raise ValueError('val 数据集为空，无法进行可视化校验。')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_count = min(args.num_samples, len(dataset))
    rng = random.Random(args.seed)
    sampled_indices = sorted(rng.sample(range(len(dataset)), sample_count))

    print(f'dataset_root={dataset_root}')
    print(f'split={split_name}')
    print(f'total_samples={len(dataset)}')
    print(f'sampled_indices={sampled_indices}')
    print(f'output_dir={output_dir.resolve()}')

    for output_index, dataset_index in enumerate(sampled_indices, start=1):
        sample = dataset[dataset_index]
        visualized_image = draw_bboxes(sample)
        output_name = build_output_name(sample, output_index)
        output_path = output_dir / output_name
        cv2.imwrite(str(output_path), visualized_image)
        print(f'saved: index={dataset_index} -> {output_path}')


if __name__ == '__main__':
    main()