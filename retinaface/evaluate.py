import os
import csv
import glob
from datetime import datetime

import cv2
import numpy as np
import torch
import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from retinaface.dataset import collect_split_parquet_files, download_dataset_to_data_dir
from retinaface.inference import Retinaface, letterbox_image, preprocess_input
from retinaface.postprocess import decode, decode_landm, non_max_suppression, retinaface_correct_boxes


def normalize_backbone_tag(backbone):
    return backbone.replace('/', '__').replace(' ', '_')


def infer_backbone_from_weights(weights_path):
    weight_name = os.path.splitext(os.path.basename(weights_path))[0]
    normalized = weight_name.lower()
    candidates = [
        ('retinaface_', ''),
        ('retinaface-', ''),
        ('retinaface', ''),
    ]

    for prefix, replacement in candidates:
        if normalized.startswith(prefix):
            normalized = replacement + normalized[len(prefix):]
            break

    normalized = normalized.lstrip('_-')
    for suffix in ('_best', '-best', '_final', '-final', '_last', '-last'):
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)]
            break
    if normalized == 'mobilenet0.25':
        return 'mobilenet'
    return normalized


def ensure_parent_dir(file_path):
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def append_results_csv(csv_path, rows):
    ensure_parent_dir(csv_path)
    fieldnames = ['backbone', 'easy_ap', 'medium_ap', 'hard_ap', 'mAP', 'weights', 'prediction_dir', 'log_path']
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def collect_weight_paths(weights_dir='weights'):
    pattern = os.path.join(weights_dir, '*.pth')
    weight_paths = [path for path in sorted(glob.glob(pattern)) if os.path.isfile(path)]
    if not weight_paths:
        raise FileNotFoundError(f'在目录 {weights_dir} 下没有找到 .pth 文件。')
    return weight_paths


def intersect(box_a, box_b):
    count_a = np.shape(box_a)[0]
    count_b = np.shape(box_b)[0]
    max_xy = np.minimum(np.tile(np.expand_dims(box_a[:, 2:], 1), (1, count_b, 1)), np.tile(np.expand_dims(box_b[:, 2:], 0), (count_a, 1, 1)))
    min_xy = np.maximum(np.tile(np.expand_dims(box_a[:, :2], 1), (1, count_b, 1)), np.tile(np.expand_dims(box_b[:, :2], 0), (count_a, 1, 1)))
    inter = np.maximum(max_xy - min_xy, np.zeros_like(max_xy - min_xy))
    return inter[:, :, 0] * inter[:, :, 1]


def bbox_overlaps(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = np.tile(np.expand_dims(((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])), 1), [1, np.shape(box_b)[0]])
    area_b = np.tile(np.expand_dims(((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])), 0), [np.shape(box_a)[0], 1])
    union = area_a + area_b - inter
    return inter / union


def norm_score(pred):
    max_score = 0
    min_score = 1
    for _, event_data in pred.items():
        for _, boxes in event_data.items():
            if len(boxes) == 0:
                continue
            min_score = min(np.min(boxes[:, -1]), min_score)
            max_score = max(np.max(boxes[:, -1]), max_score)
    diff = max_score - min_score
    if diff <= 0:
        for _, event_data in pred.items():
            for _, boxes in event_data.items():
                if len(boxes) == 0:
                    continue
                boxes[:, -1] = 1
        return
    for _, event_data in pred.items():
        for _, boxes in event_data.items():
            if len(boxes) == 0:
                continue
            boxes[:, -1] = (boxes[:, -1] - min_score) / diff


def image_eval(pred, gt, ignore, iou_thresh):
    predicted = pred.copy()
    ground_truth = gt.copy()
    pred_recall = np.zeros(predicted.shape[0])
    recall_list = np.zeros(ground_truth.shape[0])
    proposal_list = np.ones(predicted.shape[0])
    predicted[:, 2] = predicted[:, 2] + predicted[:, 0]
    predicted[:, 3] = predicted[:, 3] + predicted[:, 1]
    ground_truth[:, 2] = ground_truth[:, 2] + ground_truth[:, 0]
    ground_truth[:, 3] = ground_truth[:, 3] + ground_truth[:, 1]
    overlaps = bbox_overlaps(predicted[:, :4], ground_truth)

    for index in range(predicted.shape[0]):
        gt_overlap = overlaps[index]
        max_overlap = gt_overlap.max()
        max_idx = gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[index] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1
        pred_recall[index] = len(np.where(recall_list == 1)[0])
    return pred_recall, proposal_list


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    for index in range(thresh_num):
        thresh = 1 - (index + 1) / thresh_num
        selected = np.where(pred_info[:, 4] >= thresh)[0]
        if len(selected) == 0:
            pr_info[index, 0] = 0
            pr_info[index, 1] = 0
        else:
            selected = selected[-1]
            positives = np.where(proposal_list[:selected + 1] == 1)[0]
            pr_info[index, 0] = len(positives)
            pr_info[index, 1] = pred_recall[selected]
    return pr_info


def dataset_pr_info(thresh_num, pr_curve, count_face):
    dataset_curve = np.zeros((thresh_num, 2))
    for index in range(thresh_num):
        if pr_curve[index, 0] > 0:
            dataset_curve[index, 0] = pr_curve[index, 1] / pr_curve[index, 0]
        if count_face > 0:
            dataset_curve[index, 1] = pr_curve[index, 1] / count_face
    return dataset_curve


def voc_ap(rec, prec):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = np.maximum(mpre[index - 1], mpre[index])
    changing_points = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[changing_points + 1] - mrec[changing_points]) * mpre[changing_points + 1])


def build_gt_boxes(example):
    bboxes = example['bboxes']
    if not bboxes['x']:
        return np.zeros((0, 4), dtype=np.float32)
    gt_boxes = np.stack(
        [
            np.asarray(bboxes['x'], dtype=np.float32),
            np.asarray(bboxes['y'], dtype=np.float32),
            np.asarray(bboxes['w'], dtype=np.float32),
            np.asarray(bboxes['h'], dtype=np.float32),
        ],
        axis=1,
    )
    return gt_boxes


def evaluation(pred, eval_dataset, iou_thresh=0.5):
    norm_score(pred)
    settings = ['easy', 'medium', 'hard']
    setting_keys = ['easy_keep_indices', 'medium_keep_indices', 'hard_keep_indices']
    aps = []
    thresh_num = 1000

    for setting_id in range(3):
        keep_key = setting_keys[setting_id]
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2)).astype('float')
        for example in tqdm.tqdm(eval_dataset, desc=f'Processing {settings[setting_id]}'):
            event_name = example['event']
            image_name = os.path.splitext(os.path.basename(example['image_path']))[0]
            pred_info = pred.get(event_name, {}).get(image_name, np.zeros((0, 5), dtype=np.float32))
            gt_boxes = build_gt_boxes(example)
            keep_indices = np.asarray(example.get(keep_key, []), dtype=np.int64)
            count_face += len(keep_indices)
            if len(gt_boxes) == 0 or len(pred_info) == 0:
                continue
            ignore = np.zeros(gt_boxes.shape[0])
            if len(keep_indices) != 0:
                ignore[keep_indices] = 1
            pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_thresh)
            pr_curve += img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)

        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)
        aps.append(voc_ap(pr_curve[:, 1], pr_curve[:, 0]))

    return {
        'easy_ap': float(aps[0]),
        'medium_ap': float(aps[1]),
        'hard_ap': float(aps[2]),
        'iou_thresh': float(iou_thresh),
    }


class EvalParquetDataset(Dataset):
    def __init__(self, eval_dataset, input_shape, letterbox_image_enabled=True):
        self.eval_dataset = eval_dataset
        self.input_shape = input_shape
        self.letterbox_image_enabled = letterbox_image_enabled

    def __len__(self):
        return len(self.eval_dataset)

    def __getitem__(self, index):
        sample = self.eval_dataset[index]
        image = sample['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_array = np.array(image, dtype=np.float32)
        image_height, image_width = image_array.shape[:2]

        if self.letterbox_image_enabled:
            processed_image = letterbox_image(image_array, [self.input_shape[1], self.input_shape[0]])
        else:
            processed_image = cv2.resize(image_array, (self.input_shape[1], self.input_shape[0]))

        image_tensor = torch.from_numpy(preprocess_input(processed_image).transpose(2, 0, 1).copy()).float()
        metadata = {
            'event': sample['event'],
            'image_path': sample['image_path'],
            'image_shape': np.array([image_height, image_width], dtype=np.float32),
        }
        return image_tensor, metadata


def collate_eval_batch(batch):
    images = torch.stack([sample[0] for sample in batch], dim=0)
    metadata = [sample[1] for sample in batch]
    return images, metadata


def save_prediction_txt(save_folder, relative_image_path, results):
    save_name = os.path.join(save_folder, os.path.splitext(relative_image_path)[0] + '.txt')
    dirname = os.path.dirname(save_name)
    os.makedirs(dirname, exist_ok=True)
    with open(save_name, 'w') as handle:
        handle.write(os.path.basename(save_name)[:-4] + '\n')
        handle.write(str(len(results)) + '\n')
        for box in results:
            x = int(box[0])
            y = int(box[1])
            w = int(box[2]) - int(box[0])
            h = int(box[3]) - int(box[1])
            handle.write(f'{x} {y} {w} {h} {box[4]} \n')


def convert_predictions_to_eval_format(results):
    if len(results) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    eval_results = np.zeros((len(results), 5), dtype=np.float32)
    eval_results[:, 0] = results[:, 0]
    eval_results[:, 1] = results[:, 1]
    eval_results[:, 2] = results[:, 2] - results[:, 0]
    eval_results[:, 3] = results[:, 3] - results[:, 1]
    eval_results[:, 4] = results[:, 4]
    return eval_results


def run_detector_on_batch(detector, anchors, input_shape, images, metadata_batch):
    if detector.cuda:
        images = images.cuda(non_blocking=True)

    loc_batch, conf_batch, landms_batch = detector.net(images)
    batch_results = []
    for batch_index, metadata in enumerate(metadata_batch):
        boxes = decode(loc_batch[batch_index], anchors, detector.cfg['variance'])
        conf = conf_batch[batch_index][:, 1:2]
        landms = decode_landm(landms_batch[batch_index], anchors, detector.cfg['variance'])
        boxes_conf_landms = torch.cat([boxes, conf, landms], dim=-1)
        boxes_conf_landms = non_max_suppression(boxes_conf_landms, detector.confidence, detector.nms_iou)
        if len(boxes_conf_landms) <= 0:
            results = np.zeros((0, 15), dtype=np.float32)
        else:
            image_shape = metadata['image_shape']
            if detector.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, input_shape, image_shape)
            scale = np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]], dtype=np.float32)
            scale_for_landmarks = np.array(
                [
                    image_shape[1], image_shape[0], image_shape[1], image_shape[0],
                    image_shape[1], image_shape[0], image_shape[1], image_shape[0],
                    image_shape[1], image_shape[0],
                ],
                dtype=np.float32,
            )
            boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
            boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks
            results = boxes_conf_landms
        batch_results.append(results)
    return batch_results


def predict_eval_dataset(detector, eval_dataset, save_folder, batch_size, num_workers):
    prediction_dataset = EvalParquetDataset(
        eval_dataset,
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

    predictions = {}
    input_shape = np.array([detector.input_shape[0], detector.input_shape[1]], dtype=np.float32)
    detector.net.eval()
    with torch.no_grad():
        for images, metadata_batch in tqdm.tqdm(dataloader, desc='Evaluating HF val split'):
            batch_results = run_detector_on_batch(detector, anchors, input_shape, images, metadata_batch)
            for metadata, results in zip(metadata_batch, batch_results):
                event_predictions = predictions.setdefault(metadata['event'], {})
                image_name = os.path.splitext(os.path.basename(metadata['image_path']))[0]
                event_predictions[image_name] = convert_predictions_to_eval_format(results)
                save_prediction_txt(save_folder, metadata['image_path'], results)

    return predictions


def load_eval_dataset(repo_id, download_root='data', cache_dir=None):
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
    eval_dataset = dataset_dict[split_name].sort('image_path')
    required_columns = {'image', 'image_path', 'event'}
    missing_columns = sorted(required_columns - set(eval_dataset.column_names))
    if missing_columns:
        raise ValueError(f'评估 split 缺少必要字段: {missing_columns}，实际字段为 {eval_dataset.column_names}')
    return eval_dataset, dataset_root, split_name


def run_batch_evaluate(
    backbone='',
    weights='',
    weights_dir='weights',
    csv_path='./outputs/evaluate_results.csv',
    repo_id='zhouxzh/retinaface_widerface',
    download_root='data',
    cache_dir='',
    batch_size=16,
    num_workers=8,
):
    if weights:
        if not os.path.isfile(weights):
            raise FileNotFoundError(f'评估权重不存在: {weights}')
        weight_paths = [weights]
    else:
        weight_paths = collect_weight_paths(weights_dir)

    eval_dataset, dataset_root, eval_split_name = load_eval_dataset(repo_id, download_root=download_root, cache_dir=cache_dir)
    results = []
    failures = []

    for weights_path in weight_paths:
        current_backbone = backbone or infer_backbone_from_weights(weights_path)
        try:
            backbone_tag = normalize_backbone_tag(current_backbone)
            weight_tag = os.path.splitext(os.path.basename(weights_path))[0]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_root = os.path.join('./outputs/evaluate', backbone_tag, weight_tag, timestamp)
            save_folder = os.path.join(save_root, 'predictions')
            log_path = os.path.join(save_root, 'evaluate.log')

            os.makedirs(save_folder, exist_ok=True)
            detector = Retinaface(backbone=current_backbone, model_path=weights_path, confidence=0.5, nms_iou=0.45)
            pred = predict_eval_dataset(detector, eval_dataset, save_folder, batch_size=batch_size, num_workers=num_workers)
            metrics = evaluation(pred, eval_dataset)
            mean_ap = (metrics['easy_ap'] + metrics['medium_ap'] + metrics['hard_ap']) / 3.0

            log_lines = [
                '==================== Evaluation Summary ====================',
                f'timestamp: {timestamp}',
                f'backbone: {current_backbone}',
                f'backbone_name: {detector.cfg["name"]}',
                f'backbone_source: {detector.cfg["backbone_source"]}',
                f'in_channels_list: {detector.cfg["in_channels_list"]}',
                f'out_channel: {detector.cfg["out_channel"]}',
                f'train_image_size: {detector.cfg["train_image_size"]}',
                f'weights: {os.path.abspath(weights_path)}',
                f'save_root: {os.path.abspath(save_root)}',
                f'prediction_dir: {os.path.abspath(save_folder)}',
                f'log_path: {os.path.abspath(log_path)}',
                f'dataset_repo_id: {repo_id}',
                f'dataset_root: {os.path.abspath(dataset_root)}',
                f'dataset_split: {eval_split_name}',
                f'image_count: {len(eval_dataset)}',
                f'batch_size: {batch_size}',
                f'num_workers: {num_workers}',
                f'confidence: {detector.confidence}',
                f'nms_iou: {detector.nms_iou}',
                f'input_shape: {detector.input_shape}',
                f'letterbox_image: {detector.letterbox_image}',
                f'cuda: {detector.cuda}',
                f'easy_ap: {metrics["easy_ap"]}',
                f'medium_ap: {metrics["medium_ap"]}',
                f'hard_ap: {metrics["hard_ap"]}',
                f'iou_thresh: {metrics["iou_thresh"]}',
                '============================================================',
            ]
            with open(log_path, 'w') as handle:
                handle.write('\n'.join(log_lines) + '\n')

            results.append(
                {
                    'backbone': current_backbone,
                    'easy_ap': f"{metrics['easy_ap']:.10f}",
                    'medium_ap': f"{metrics['medium_ap']:.10f}",
                    'hard_ap': f"{metrics['hard_ap']:.10f}",
                    'mAP': f"{mean_ap:.10f}",
                    'weights': os.path.abspath(weights_path),
                    'prediction_dir': os.path.abspath(save_folder),
                    'log_path': os.path.abspath(log_path),
                }
            )
            print(
                f"[EVAL] backbone={current_backbone} easy_ap={metrics['easy_ap']:.6f} "
                f"medium_ap={metrics['medium_ap']:.6f} hard_ap={metrics['hard_ap']:.6f} mAP={mean_ap:.6f}"
            )
        except Exception as error:
            failures.append((weights_path, current_backbone, str(error)))
            print(f"[EVAL][FAILED] weights={weights_path} backbone={current_backbone} error={error}")

    results.sort(key=lambda item: item['backbone'])
    append_results_csv(csv_path, results)
    print(f'批量评估结果已写入: {os.path.abspath(csv_path)}')

    if failures:
        failure_lines = '\n'.join(
            f'- weights={path}, backbone={backbone}, error={message}'
            for path, backbone, message in failures
        )
        raise RuntimeError(f'以下权重评估失败：\n{failure_lines}')

    return results