import argparse
import os

import cv2
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from retinaface.evaluate import EvalParquetDataset, collate_eval_batch, load_eval_dataset, run_detector_on_batch
from retinaface.inference import Retinaface


DEFAULT_REPO_ID = 'zhouxzh/retinaface_widerface'
DEFAULT_DOWNLOAD_ROOT = 'data'
DEFAULT_OUTPUT_DIR = 'outputs/predict_eval_dataset_check'
MEAN_BGR = np.array((104, 117, 123), dtype=np.float32)


def inverse_preprocess_input(image_chw):
    image_hwc = image_chw.transpose(1, 2, 0).copy()
    image_hwc += MEAN_BGR
    return np.clip(image_hwc, 0, 255).astype(np.uint8)


def compute_resize_params(image_shape, input_shape, letterbox_image_enabled):
    image_height, image_width = image_shape
    input_height, input_width = input_shape
    if letterbox_image_enabled:
        scale = min(input_width / image_width, input_height / image_height)
        resized_width = int(image_width * scale)
        resized_height = int(image_height * scale)
        pad_x = (input_width - resized_width) // 2
        pad_y = (input_height - resized_height) // 2
        return {'scale_x': scale, 'scale_y': scale, 'pad_x': pad_x, 'pad_y': pad_y}
    return {
        'scale_x': input_width / image_width,
        'scale_y': input_height / image_height,
        'pad_x': 0,
        'pad_y': 0,
    }


def map_box_to_input_view(box, resize_params):
    x1, y1, x2, y2 = box
    return (
        int(round(x1 * resize_params['scale_x'] + resize_params['pad_x'])),
        int(round(y1 * resize_params['scale_y'] + resize_params['pad_y'])),
        int(round(x2 * resize_params['scale_x'] + resize_params['pad_x'])),
        int(round(y2 * resize_params['scale_y'] + resize_params['pad_y'])),
    )


def map_point_to_input_view(x_coord, y_coord, resize_params):
    return (
        int(round(x_coord * resize_params['scale_x'] + resize_params['pad_x'])),
        int(round(y_coord * resize_params['scale_y'] + resize_params['pad_y'])),
    )


def draw_gt_annotations(image_bgr, sample, resize_params=None):
    bboxes = sample['bboxes']
    total_boxes = len(bboxes['x'])
    for index in range(total_boxes):
        x1 = float(bboxes['x'][index])
        y1 = float(bboxes['y'][index])
        x2 = x1 + float(bboxes['w'][index])
        y2 = y1 + float(bboxes['h'][index])
        if resize_params is not None:
            x1, y1, x2, y2 = map_box_to_input_view((x1, y1, x2, y2), resize_params)
        else:
            x1 = int(round(x1))
            y1 = int(round(y1))
            x2 = int(round(x2))
            y2 = int(round(y2))
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    landmarks = sample.get('landmarks') or {}
    landmark_pairs = [('x1', 'y1'), ('x2', 'y2'), ('x3', 'y3'), ('x4', 'y4'), ('x5', 'y5')]
    visibility_keys = ['v1', 'v2', 'v3', 'v4', 'v5']
    for box_index in range(total_boxes):
        for point_index, (x_key, y_key) in enumerate(landmark_pairs):
            x_values = landmarks.get(x_key, [])
            y_values = landmarks.get(y_key, [])
            visibility_values = landmarks.get(visibility_keys[point_index], [])
            if box_index >= len(x_values) or box_index >= len(y_values):
                continue
            x_coord = float(x_values[box_index])
            y_coord = float(y_values[box_index])
            if x_coord < 0 or y_coord < 0:
                continue
            if visibility_values and box_index < len(visibility_values) and float(visibility_values[box_index]) <= 0:
                continue
            if resize_params is not None:
                point = map_point_to_input_view(x_coord, y_coord, resize_params)
            else:
                point = (int(round(x_coord)), int(round(y_coord)))
            cv2.circle(image_bgr, point, 2, (0, 255, 255), -1)


def draw_pred_annotations(image_bgr, results, resize_params=None):
    for detection in results:
        x1, y1, x2, y2, score = detection[:5]
        if resize_params is not None:
            x1, y1, x2, y2 = map_box_to_input_view((float(x1), float(y1), float(x2), float(y2)), resize_params)
        else:
            x1 = int(round(float(x1)))
            y1 = int(round(float(y1)))
            x2 = int(round(float(x2)))
            y2 = int(round(float(y2)))
        cv2.rectangle(
            image_bgr,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            image_bgr,
            f'{float(score):.3f}',
            (x1, max(15, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        for point_index in range(5):
            landmark_x = detection[5 + point_index * 2]
            landmark_y = detection[6 + point_index * 2]
            if resize_params is not None:
                point = map_point_to_input_view(float(landmark_x), float(landmark_y), resize_params)
            else:
                point = (int(round(float(landmark_x))), int(round(float(landmark_y))))
            cv2.circle(
                image_bgr,
                point,
                2,
                (255, 0, 255),
                -1,
            )


def add_panel_title(image_bgr, title):
    cv2.rectangle(image_bgr, (0, 0), (image_bgr.shape[1], 44), (20, 20, 20), -1)
    cv2.putText(image_bgr, title, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return image_bgr


def build_comparison_canvas(original_view, model_input_view):
    target_height = max(original_view.shape[0], model_input_view.shape[0])

    def pad_to_height(image_bgr, height):
        if image_bgr.shape[0] == height:
            return image_bgr
        pad_bottom = height - image_bgr.shape[0]
        return cv2.copyMakeBorder(image_bgr, 0, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(32, 32, 32))

    original_view = pad_to_height(original_view, target_height)
    model_input_view = pad_to_height(model_input_view, target_height)
    separator = np.full((target_height, 24, 3), 32, dtype=np.uint8)
    comparison = np.concatenate([original_view, separator, model_input_view], axis=1)
    legend = np.full((52, comparison.shape[1], 3), 24, dtype=np.uint8)
    cv2.putText(legend, 'GT bbox: green   GT landmarks: yellow   Pred bbox: red   Pred landmarks: magenta', (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return np.concatenate([comparison, legend], axis=0)


def debug_predict_eval_dataset(detector, eval_dataset, output_dir, batch_size, num_workers, num_samples=10, seed=3407):
    if len(eval_dataset) == 0:
        raise ValueError('val 数据集为空，无法进行可视化校验。')

    sample_count = min(num_samples, len(eval_dataset))
    random_generator = np.random.default_rng(seed)
    sampled_indices = sorted(random_generator.choice(len(eval_dataset), size=sample_count, replace=False).tolist())
    sampled_dataset = eval_dataset.select(sampled_indices)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    prediction_dataset = EvalParquetDataset(
        sampled_dataset,
        input_shape=(detector.input_shape[0], detector.input_shape[1]),
        letterbox_image_enabled=detector.letterbox_image,
    )
    dataloader = DataLoader(
        prediction_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=detector.cuda,
        collate_fn=collate_eval_batch,
    )

    anchors = detector.anchors
    if detector.cuda:
        anchors = anchors.cuda()

    input_shape = np.array([detector.input_shape[0], detector.input_shape[1]], dtype=np.float32)
    detector.net.eval()
    saved_paths = []
    sample_offset = 0
    with torch.no_grad():
        for images, metadata_batch in tqdm.tqdm(dataloader, desc='Debug predict_eval_dataset'):
            batch_results = run_detector_on_batch(detector, anchors, input_shape, images, metadata_batch)
            for batch_index, (image_tensor, metadata, results) in enumerate(zip(images, metadata_batch, batch_results)):
                sample = sampled_dataset[sample_offset + batch_index]
                image = sample['image']
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                original_view = cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
                model_input_view = inverse_preprocess_input(image_tensor.cpu().numpy())
                model_input_view = cv2.cvtColor(model_input_view, cv2.COLOR_RGB2BGR)

                resize_params = compute_resize_params(
                    image_shape=original_view.shape[:2],
                    input_shape=(detector.input_shape[0], detector.input_shape[1]),
                    letterbox_image_enabled=detector.letterbox_image,
                )

                draw_gt_annotations(original_view, sample)
                draw_pred_annotations(original_view, results)
                draw_gt_annotations(model_input_view, sample, resize_params=resize_params)
                draw_pred_annotations(model_input_view, results, resize_params=resize_params)

                add_panel_title(original_view, 'Original image view')
                add_panel_title(model_input_view, 'Model input view (inverse preprocess)')
                comparison = build_comparison_canvas(original_view, model_input_view)
                safe_name = metadata['image_path'].replace('/', '__')
                output_path = os.path.join(output_dir, safe_name)
                cv2.imwrite(output_path, comparison)
                saved_paths.append(output_path)
            sample_offset += len(metadata_batch)

    return {
        'output_dir': output_dir,
        'sampled_indices': sampled_indices,
        'saved_paths': saved_paths,
    }


def main():
    parser = argparse.ArgumentParser(description='可视化校验 evaluate.predict_eval_dataset 的预测结果')
    parser.add_argument('--backbone', type=str, required=True, help='主干网络名称')
    parser.add_argument('--weights', type=str, required=True, help='权重文件路径')
    parser.add_argument('--repo_id', type=str, default=DEFAULT_REPO_ID, help='Hugging Face 数据集仓库名')
    parser.add_argument('--download_root', type=str, default=DEFAULT_DOWNLOAD_ROOT, help='远程数据集下载到本地的根目录')
    parser.add_argument('--cache_dir', type=str, default='', help='datasets cache 目录；默认使用 Hugging Face 默认缓存位置')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, help='可视化结果保存目录')
    parser.add_argument('--num_samples', type=int, default=10, help='随机抽样数量，默认 10')
    parser.add_argument('--seed', type=int, default=3407, help='随机种子，默认 3407')
    parser.add_argument('--batch_size', type=int, default=8, help='调试批次大小，默认 8')
    parser.add_argument('--num_workers', type=int, default=0, help='调试 DataLoader worker 数，默认 0')
    parser.add_argument('--confidence', type=float, default=0.5, help='预测置信度阈值，默认 0.5')
    parser.add_argument('--nms_iou', type=float, default=0.45, help='NMS IoU 阈值，默认 0.45')
    args = parser.parse_args()

    eval_dataset, dataset_root, split_name = load_eval_dataset(
        repo_id=args.repo_id,
        download_root=args.download_root,
        cache_dir=args.cache_dir,
    )
    detector = Retinaface(
        backbone=args.backbone,
        model_path=args.weights,
        confidence=args.confidence,
        nms_iou=args.nms_iou,
    )
    debug_info = debug_predict_eval_dataset(
        detector=detector,
        eval_dataset=eval_dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    print(f'dataset_root={dataset_root}')
    print(f'split={split_name}')
    print(f'output_dir={debug_info["output_dir"]}')
    print(f'sampled_indices={debug_info["sampled_indices"]}')
    for output_path in debug_info['saved_paths']:
        print(f'saved: {output_path}')


if __name__ == '__main__':
    main()