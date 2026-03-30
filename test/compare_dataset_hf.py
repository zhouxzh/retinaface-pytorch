import argparse
import hashlib
import json
import numbers
import struct
from pathlib import Path

from datasets import Image, load_dataset
from huggingface_hub import snapshot_download
from PIL import Image as PILImage
from scipy.io import loadmat


DEFAULT_REPO_ID = "zhouxzh/retinaface_widerface"
DEFAULT_LOCAL_ROOT = Path("./data")
DEFAULT_DOWNLOAD_ROOT = Path("./data")
RAW_SOURCE_ROOT = Path("./data/widerface")
GROUND_TRUTH_ROOT = Path("./widerface_evaluate/ground_truth")
DEFAULT_SPLITS = ("train", "val")
SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("val", "validation"),
}


def to_float32(value):
    return struct.unpack("f", struct.pack("f", float(value)))[0]


def parse_args():
    parser = argparse.ArgumentParser(description="测试 Hugging Face 数据集仓库是否可正常加载，并与本地数据做一致性校验")
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID, help="Hugging Face 数据集仓库名")
    parser.add_argument("--local_root", default=str(DEFAULT_LOCAL_ROOT), help="本地数据根目录，默认从 ./data 下自动探测 parquet 数据集")
    parser.add_argument("--download_root", default=str(DEFAULT_DOWNLOAD_ROOT), help="将 Hugging Face 数据集下载到本地的根目录，默认 ./data")
    parser.add_argument("--sample_count", type=int, default=20, help="快速模式下每个 split 抽样校验的样本数")
    parser.add_argument("--check_all", action="store_true", help="对全部样本做完整一致性校验")
    return parser.parse_args()


def load_remote_dataset(repo_id, download_root):
    download_dir = download_dataset_to_data_dir(repo_id, Path(download_root))
    print(f"[1/4] 从 Hugging Face 下载并加载数据集: {repo_id}")
    dataset = load_downloaded_dataset(download_dir)
    print(f"  - 远端实际 splits: {list(dataset.keys())}")
    print(f"  - 远端数据已下载到: {download_dir}")
    return dataset


def download_dataset_to_data_dir(repo_id, download_root):
    download_root.mkdir(parents=True, exist_ok=True)
    target_dir = download_root / repo_id.split("/")[-1]
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target_dir),
        allow_patterns=["*.parquet", "README.md", "*.json", "dataset_infos.json"],
    )
    return target_dir


def load_downloaded_dataset(download_dir):
    data_files = {}
    train_files = collect_split_parquet_files(download_dir, "train")
    validation_files = collect_split_parquet_files(download_dir, "validation")
    val_files = collect_split_parquet_files(download_dir, "val")

    if train_files:
        data_files["train"] = [str(path) for path in train_files]
    if validation_files:
        data_files["validation"] = [str(path) for path in validation_files]
    elif val_files:
        data_files["val"] = [str(path) for path in val_files]

    if not data_files:
        raise FileNotFoundError(f"下载目录下没有找到 parquet 文件: {download_dir}")

    return load_dataset("parquet", data_files=data_files)


def collect_split_parquet_files(root_dir, split_name):
    nested_dir = root_dir / split_name
    if nested_dir.exists():
        nested_files = sorted(nested_dir.glob("*.parquet"))
        if nested_files:
            return nested_files

    top_level_files = sorted(root_dir.glob(f"{split_name}*.parquet"))
    if top_level_files:
        return top_level_files

    return []


def load_local_dataset(local_root, excluded_roots=None):
    local_root = resolve_local_root(Path(local_root), excluded_roots=excluded_roots)
    train_files = collect_local_parquet_files(local_root, "train")
    val_files = collect_local_parquet_files(local_root, "val")
    print(f"[2/4] 加载本地 parquet 数据: {local_root}")
    print(f"  - 本地 train parquet: {[path.name for path in train_files]}")
    print(f"  - 本地 val parquet: {[path.name for path in val_files]}")
    return load_dataset(
        "parquet",
        data_files={
            "train": [str(path) for path in train_files],
            "val": [str(path) for path in val_files],
        },
    )


def try_load_local_parquet_dataset(local_root, excluded_roots=None):
    try:
        return load_local_dataset(local_root, excluded_roots=excluded_roots)
    except FileNotFoundError:
        return None


def resolve_local_root(requested_root, excluded_roots=None):
    excluded_roots = {path.resolve() for path in (excluded_roots or [])}

    if not requested_root.exists():
        raise FileNotFoundError(f"本地数据目录不存在: {requested_root}")

    if requested_root.resolve() not in excluded_roots and has_required_parquet_layout(requested_root):
        return requested_root

    candidate_roots = [path for path in sorted(requested_root.iterdir()) if path.is_dir()]
    for candidate in candidate_roots:
        if candidate.resolve() in excluded_roots:
            continue
        if has_required_parquet_layout(candidate):
            print(f"  - 从 {requested_root} 自动探测到本地数据目录: {candidate}")
            return candidate

    raise FileNotFoundError(
        f"在 {requested_root} 下没有找到包含 train/val parquet 的数据集目录"
    )


def has_required_parquet_layout(root_dir):
    train_dir = root_dir / "train"
    val_dir = root_dir / "val"
    return (
        train_dir.exists()
        and val_dir.exists()
        and any(train_dir.glob("*.parquet"))
        and any(val_dir.glob("*.parquet"))
    )


def collect_local_parquet_files(local_root, split):
    split_dir = local_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"本地 split 目录不存在: {split_dir}")

    parquet_files = sorted(split_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"本地 split 目录下没有 parquet 文件: {split_dir}")
    return parquet_files


def sha256_of_file(file_path):
    with open(file_path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def empty_landmarks():
    return {
        "x1": [], "y1": [], "x2": [], "y2": [], "x3": [], "y3": [],
        "x4": [], "y4": [], "x5": [], "y5": [],
        "v1": [], "v2": [], "v3": [], "v4": [], "v5": [],
    }


def empty_bboxes():
    return {"x": [], "y": [], "w": [], "h": []}


def parse_train_label_file():
    label_path = RAW_SOURCE_ROOT / "train" / "label.txt"
    if not label_path.exists():
        raise FileNotFoundError(f"训练标注不存在: {label_path}")

    train_records = {}
    current_relative_path = None
    current_annotations = []
    with open(label_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if current_relative_path is not None:
                    train_records[current_relative_path] = current_annotations
                current_relative_path = line[2:]
                current_annotations = []
            else:
                current_annotations.append([float(value) for value in line.split()])
    if current_relative_path is not None:
        train_records[current_relative_path] = current_annotations
    return train_records


def build_train_raw_example(relative_path, annotations):
    image_path = RAW_SOURCE_ROOT / "train" / "images" / relative_path
    with PILImage.open(image_path) as image:
        width, height = image.size

    bboxes = empty_bboxes()
    landmarks = empty_landmarks()
    blur = []
    for values in annotations:
        bboxes["x"].append(to_float32(values[0]))
        bboxes["y"].append(to_float32(values[1]))
        bboxes["w"].append(to_float32(values[2]))
        bboxes["h"].append(to_float32(values[3]))
        landmarks["x1"].append(to_float32(values[4]))
        landmarks["y1"].append(to_float32(values[5]))
        landmarks["x2"].append(to_float32(values[7]))
        landmarks["y2"].append(to_float32(values[8]))
        landmarks["x3"].append(to_float32(values[10]))
        landmarks["y3"].append(to_float32(values[11]))
        landmarks["x4"].append(to_float32(values[13]))
        landmarks["y4"].append(to_float32(values[14]))
        landmarks["x5"].append(to_float32(values[16]))
        landmarks["y5"].append(to_float32(values[17]))
        landmarks["v1"].append(to_float32(values[6]))
        landmarks["v2"].append(to_float32(values[9]))
        landmarks["v3"].append(to_float32(values[12]))
        landmarks["v4"].append(to_float32(values[15]))
        landmarks["v5"].append(to_float32(values[18]))
        blur.append(to_float32(values[19]) if len(values) > 19 else to_float32(-1.0))

    return {
        "image": {"bytes": None, "path": str(image_path)},
        "image_path": relative_path,
        "event": Path(relative_path).parts[0],
        "split": "train",
        "width": width,
        "height": height,
        "bboxes": bboxes,
        "landmarks": landmarks,
        "blur": blur,
        "easy_keep_indices": [],
        "medium_keep_indices": [],
        "hard_keep_indices": [],
    }


def matlab_indices_to_zero_based(indices):
    if len(indices) == 0:
        return []
    return [int(index) - 1 for index in indices.reshape(-1).tolist()]


def build_val_raw_index():
    gt_mat = loadmat(GROUND_TRUTH_ROOT / "wider_face_val.mat")
    hard_mat = loadmat(GROUND_TRUTH_ROOT / "wider_hard_val.mat")
    medium_mat = loadmat(GROUND_TRUTH_ROOT / "wider_medium_val.mat")
    easy_mat = loadmat(GROUND_TRUTH_ROOT / "wider_easy_val.mat")

    facebox_list = gt_mat["face_bbx_list"]
    event_list = gt_mat["event_list"]
    file_list = gt_mat["file_list"]
    easy_gt_list = easy_mat["gt_list"]
    medium_gt_list = medium_mat["gt_list"]
    hard_gt_list = hard_mat["gt_list"]

    val_records = {}
    for event_index in range(len(event_list)):
        event_name = str(event_list[event_index][0][0])
        image_entries = file_list[event_index][0]
        event_face_boxes = facebox_list[event_index][0]
        event_easy = easy_gt_list[event_index][0]
        event_medium = medium_gt_list[event_index][0]
        event_hard = hard_gt_list[event_index][0]

        for image_index in range(len(image_entries)):
            image_stem = str(image_entries[image_index][0][0])
            relative_path = f"{event_name}/{image_stem}.jpg"
            image_path = RAW_SOURCE_ROOT / "val" / "images" / relative_path
            with PILImage.open(image_path) as image:
                width, height = image.size

            boxes = event_face_boxes[image_index][0].astype("float32")
            bboxes = empty_bboxes()
            for box in boxes:
                bboxes["x"].append(to_float32(box[0]))
                bboxes["y"].append(to_float32(box[1]))
                bboxes["w"].append(to_float32(box[2]))
                bboxes["h"].append(to_float32(box[3]))

            val_records[relative_path] = {
                "image": {"bytes": None, "path": str(image_path)},
                "image_path": relative_path,
                "event": event_name,
                "split": "val",
                "width": width,
                "height": height,
                "bboxes": bboxes,
                "landmarks": empty_landmarks(),
                "blur": [],
                "easy_keep_indices": matlab_indices_to_zero_based(event_easy[image_index][0]),
                "medium_keep_indices": matlab_indices_to_zero_based(event_medium[image_index][0]),
                "hard_keep_indices": matlab_indices_to_zero_based(event_hard[image_index][0]),
            }
    return val_records


def build_raw_reference_indexes():
    print("[2/4] 本地未找到独立 parquet，对比原始 WiderFace 数据")
    return {
        "train": parse_train_label_file(),
        "val": build_val_raw_index(),
    }


def resolve_split_name(dataset_dict, canonical_split):
    for candidate in SPLIT_ALIASES[canonical_split]:
        if candidate in dataset_dict:
            return candidate
    return None


def ensure_remote_images_decode(dataset_dict):
    print("[3/4] 检查远端数据集图片字段是否可正常解码")
    for split in DEFAULT_SPLITS:
        resolved_split = resolve_split_name(dataset_dict, split)
        if resolved_split is None or len(dataset_dict[resolved_split]) == 0:
            raise AssertionError(f"远端 split 缺失或为空: {split}, available={list(dataset_dict.keys())}")
        sample = dataset_dict[resolved_split][0]
        image = sample["image"]
        if not hasattr(image, "size"):
            raise AssertionError(f"远端 split={resolved_split} 的 image 字段未解码成图片对象")
        print(
            f"  - {split} -> {resolved_split}: 首条样本图片可解码, "
            f"size={image.size}, image_path={sample['image_path']}"
        )


def with_raw_image(dataset_dict):
    converted = {}
    for split, dataset in dataset_dict.items():
        if "image" in dataset.column_names:
            converted[split] = dataset.cast_column("image", Image(decode=False))
        else:
            converted[split] = dataset
    return converted


def normalize_numbers(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return to_float32(value)
    if isinstance(value, list):
        return [normalize_numbers(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_numbers(item) for key, item in value.items()}
    return value


def image_sha256(image_field):
    image_bytes = image_field.get("bytes")
    image_path = image_field.get("path")

    if image_bytes is not None:
        return hashlib.sha256(image_bytes).hexdigest()

    if image_path:
        with open(image_path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    raise AssertionError("image 字段既没有 bytes 也没有 path")


def build_signature(example):
    payload = {
        "image_sha256": image_sha256(example["image"]),
        "image_path": example["image_path"],
        "event": example["event"],
        "split": example["split"],
        "width": int(example["width"]),
        "height": int(example["height"]),
        "bboxes": normalize_numbers(example["bboxes"]),
        "landmarks": normalize_numbers(example["landmarks"]),
        "blur": normalize_numbers(example["blur"]),
        "easy_keep_indices": normalize_numbers(example["easy_keep_indices"]),
        "medium_keep_indices": normalize_numbers(example["medium_keep_indices"]),
        "hard_keep_indices": normalize_numbers(example["hard_keep_indices"]),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sample_indices(length, sample_count):
    if length <= sample_count:
        return list(range(length))
    if sample_count <= 1:
        return [0]

    step = (length - 1) / float(sample_count - 1)
    indices = []
    for i in range(sample_count):
        index = int(round(i * step))
        if not indices or index != indices[-1]:
            indices.append(index)
    return indices


def compare_examples(local_example, remote_example, split, index):
    local_sig = build_signature(local_example)
    remote_sig = build_signature(remote_example)
    if local_sig != remote_sig:
        raise AssertionError(
            f"split={split} index={index} 数据不一致: "
            f"local_image_path={local_example['image_path']}, remote_image_path={remote_example['image_path']}"
        )


def compare_split_fast(local_dataset, remote_dataset, split, sample_count):
    print(f"  - {split}: 进行抽样一致性校验, sample_count={sample_count}")
    indices = sample_indices(len(local_dataset), sample_count)
    for index in indices:
        compare_examples(local_dataset[index], remote_dataset[index], split, index)


def build_signature_map(dataset, split):
    signature_map = {}
    for index, example in enumerate(dataset):
        image_path = example["image_path"]
        if image_path in signature_map:
            raise AssertionError(f"split={split} 存在重复 image_path: {image_path}")
        signature_map[image_path] = build_signature(example)
        if (index + 1) % 2000 == 0:
            print(f"    已处理 {index + 1} 条 {split} 样本")
    return signature_map


def compare_split_all(local_dataset, remote_dataset, split):
    print(f"  - {split}: 进行全量一致性校验")
    local_map = build_signature_map(local_dataset, split)
    remote_map = build_signature_map(remote_dataset, split)

    local_keys = set(local_map)
    remote_keys = set(remote_map)
    if local_keys != remote_keys:
        missing_remote = sorted(local_keys - remote_keys)[:10]
        missing_local = sorted(remote_keys - local_keys)[:10]
        raise AssertionError(
            f"split={split} 样本集合不一致: missing_remote={missing_remote}, missing_local={missing_local}"
        )

    for image_path in sorted(local_keys):
        if local_map[image_path] != remote_map[image_path]:
            raise AssertionError(f"split={split} 样本内容不一致: {image_path}")


def compare_remote_with_raw_source(remote_dataset_dict, raw_indexes, sample_count, check_all):
    print("[4/4] 校验远端数据与原始 WiderFace 数据是否一致")
    for split in DEFAULT_SPLITS:
        remote_split = resolve_split_name(remote_dataset_dict, split)
        if remote_split is None:
            raise AssertionError(f"远端缺少 split: {split}, available={list(remote_dataset_dict.keys())}")

        remote_dataset = remote_dataset_dict[remote_split]
        raw_index = raw_indexes[split]
        print(f"  - {split}: remote({remote_split})={len(remote_dataset)} 条, raw={len(raw_index)} 条")
        if len(remote_dataset) != len(raw_index):
            raise AssertionError(f"split={split} 样本数量不一致")

        if check_all:
            indices = list(range(len(remote_dataset)))
        else:
            indices = sample_indices(len(remote_dataset), sample_count)
            print(f"  - {split}: 进行抽样一致性校验, sample_count={sample_count}")

        for index in indices:
            remote_example = remote_dataset[index]
            image_path = remote_example["image_path"]
            if image_path not in raw_index:
                raise AssertionError(f"split={split} 原始数据缺少样本: {image_path}")

            raw_value = raw_index[image_path]
            if split == "train":
                local_example = build_train_raw_example(image_path, raw_value)
            else:
                local_example = raw_value
            compare_examples(local_example, remote_example, split, index)


def compare_datasets(local_dataset_dict, remote_dataset_dict, sample_count, check_all):
    print("[4/4] 校验远端数据与本地数据是否一致")
    for split in DEFAULT_SPLITS:
        local_split = resolve_split_name(local_dataset_dict, split)
        remote_split = resolve_split_name(remote_dataset_dict, split)
        if local_split is None:
            raise AssertionError(f"本地缺少 split: {split}, available={list(local_dataset_dict.keys())}")
        if remote_split is None:
            raise AssertionError(f"远端缺少 split: {split}, available={list(remote_dataset_dict.keys())}")

        local_dataset = local_dataset_dict[local_split]
        remote_dataset = remote_dataset_dict[remote_split]

        print(
            f"  - {split}: local({local_split})={len(local_dataset)} 条, "
            f"remote({remote_split})={len(remote_dataset)} 条"
        )
        if len(local_dataset) != len(remote_dataset):
            raise AssertionError(f"split={split} 样本数量不一致")

        if check_all:
            compare_split_all(local_dataset, remote_dataset, split)
        else:
            compare_split_fast(local_dataset, remote_dataset, split, sample_count)


def main():
    args = parse_args()
    remote_download_dir = Path(args.download_root) / args.repo_id.split("/")[-1]

    remote_dataset = load_remote_dataset(args.repo_id, args.download_root)
    ensure_remote_images_decode(remote_dataset)

    remote_raw = with_raw_image(remote_dataset)
    local_dataset = try_load_local_parquet_dataset(args.local_root, excluded_roots={remote_download_dir})
    if local_dataset is not None:
        local_raw = with_raw_image(local_dataset)
        compare_datasets(local_raw, remote_raw, args.sample_count, args.check_all)
    else:
        raw_indexes = build_raw_reference_indexes()
        compare_remote_with_raw_source(remote_raw, raw_indexes, args.sample_count, args.check_all)

    print("\n校验通过")
    print(f"- 仓库可正常 load_dataset: {args.repo_id}")
    print("- 图片字段可正常解码")
    print("- 远端数据与本地数据一致")


if __name__ == "__main__":
    main()