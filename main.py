"""
RetinaFace 统一入口脚本
支持训练、预测、评估、FPS测试等所有功能

使用示例：
  # 预测
  python main.py predict --input examples/ --output outputs/

  # 训练
        python main.py train --backbone mobilenetv2_050 --batch_size 32

  # 评估
  python main.py evaluate

  # FPS测试
  python main.py fps --test_interval 100

  # 视频检测
  python main.py video --input 0  # 摄像头
  python main.py video --input test.mp4 --output result.mp4
"""
import argparse
import glob
import os
import sys

from retinaface.config import get_backbone_cfg, get_default_weight_path
from retinaface.evaluate import infer_backbone_from_weights, run_batch_evaluate
from retinaface.inference import run_fps, run_predict, run_video
from retinaface.trainer import export_onnx, run_train


DEFAULT_TRAIN_BATCH_SIZE = 32
DEFAULT_TRAIN_LR = 4e-3
DEFAULT_TRAIN_WORKERS = 8
DEFAULT_ONNX_OPSET = 12
DEFAULT_TRAIN_REPO_ID = 'zhouxzh/retinaface_widerface'
DEFAULT_DOWNLOAD_ROOT = 'data'
DEFAULT_TRAIN_SEED = 3407
DEFAULT_PREFETCH_FACTOR = 4
DEFAULT_EVAL_BATCH_SIZE = 16
DEFAULT_EVAL_WORKERS = 8


def run_export_onnx(args):
    cfg = get_backbone_cfg(args.backbone)

    if args.weights:
        if not os.path.isfile(args.weights):
            raise FileNotFoundError(f'权重文件不存在: {args.weights}')

        onnx_path = args.output or f"{os.path.splitext(args.weights)[0]}.onnx"
        export_onnx(args.weights, onnx_path, cfg, opset_version=args.opset)
        print(f'ONNX exported to: {onnx_path}')
        return

    if args.output:
        raise ValueError('批量转换时不支持 --output，请改为指定 --weights 后单独导出。')

    pattern = os.path.join(args.weights_dir, '*.pth')
    weights_paths = sorted(glob.glob(pattern))
    if not weights_paths:
        raise FileNotFoundError(f'在目录 {args.weights_dir} 下没有找到 .pth 文件。')

    export_failures = []
    for weights_path in weights_paths:
        onnx_path = f"{os.path.splitext(weights_path)[0]}.onnx"
        try:
            export_onnx(weights_path, onnx_path, cfg, opset_version=args.opset)
            print(f'ONNX exported to: {onnx_path}')
        except Exception as error:
            export_failures.append((weights_path, str(error)))

    if export_failures:
        failure_messages = '\n'.join(f'- {path}: {message}' for path, message in export_failures)
        raise RuntimeError(f'以下权重导出失败，请确认这些 pth 是否与 backbone={args.backbone} 匹配：\n{failure_messages}')


def resolve_evaluate_args(args):
    backbone = (args.backbone or '').strip()
    weights = (args.weights or '').strip()

    if not backbone and not weights:
        return '', ''

    if weights and not backbone:
        backbone = infer_backbone_from_weights(weights)

    if backbone and not weights:
        weights = get_default_weight_path(backbone)

    return backbone, weights


def main():
    parser = argparse.ArgumentParser(description='RetinaFace 人脸检测工具')
    subparsers = parser.add_subparsers(dest='command', help='选择功能模式')

    # ========== 预测模式 ==========
    predict_parser = subparsers.add_parser('predict', help='批量图片检测')
    predict_parser.add_argument('--input', type=str, default='examples/', help='输入图片文件夹')
    predict_parser.add_argument('--output', type=str, default='outputs/', help='输出结果文件夹')
    predict_parser.add_argument('--backbone', type=str, default='mobilenetv2_050', help='主干网络名称，支持 timm 的 resnet/mobilenet 家族')
    predict_parser.add_argument('--weights', type=str, default='', help='权重文件路径，为空时按 backbone 推导默认值')

    # ========== 视频模式 ==========
    video_parser = subparsers.add_parser('video', help='视频/摄像头检测')
    video_parser.add_argument('--input', type=str, default='0', help='视频路径或0表示摄像头')
    video_parser.add_argument('--output', type=str, default='', help='保存视频路径（可选）')
    video_parser.add_argument('--fps', type=float, default=25.0, help='输出视频帧率')
    video_parser.add_argument('--backbone', type=str, default='mobilenetv2_050', help='主干网络名称，支持 timm 的 resnet/mobilenet 家族')
    video_parser.add_argument('--weights', type=str, default='', help='权重文件路径，为空时按 backbone 推导默认值')

    # ========== FPS测试 ==========
    fps_parser = subparsers.add_parser('fps', help='测试模型推理速度')
    fps_parser.add_argument('--test_interval', type=int, default=100, help='测试次数')
    fps_parser.add_argument('--image', type=str, default='examples/street.jpg', help='测试图片')
    fps_parser.add_argument('--backbone', type=str, default='mobilenetv2_050', help='主干网络名称，支持 timm 的 resnet/mobilenet 家族')
    fps_parser.add_argument('--weights', type=str, default='', help='权重文件路径，为空时按 backbone 推导默认值')

    # ========== 训练模式 ==========
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--backbone', type=str, default='mobilenetv2_050', help='主干网络名称，支持 timm 的 resnet/mobilenet 家族')
    train_parser.add_argument('--batch_size', type=int, default=DEFAULT_TRAIN_BATCH_SIZE, help='批次大小，默认按 32G 显存单卡调到 32')
    train_parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    train_parser.add_argument('--lr', type=float, default=DEFAULT_TRAIN_LR, help='学习率，默认随更大的 batch size 调到 4e-3')
    train_parser.add_argument('--num_workers', type=int, default=DEFAULT_TRAIN_WORKERS, help='DataLoader worker 数，默认 8')
    train_parser.add_argument('--repo_id', type=str, default=DEFAULT_TRAIN_REPO_ID, help='Hugging Face 数据集仓库名，例如 zhouxzh/retinaface_widerface')
    train_parser.add_argument('--download_root', type=str, default=DEFAULT_DOWNLOAD_ROOT, help='远程数据集下载到本地的根目录，默认 data')
    train_parser.add_argument('--seed', type=int, default=DEFAULT_TRAIN_SEED, help='随机种子，默认 3407')
    train_parser.add_argument('--prefetch_factor', type=int, default=DEFAULT_PREFETCH_FACTOR, help='DataLoader 预取批次数，默认 4')
    train_parser.add_argument('--worker_start_method', type=str, default='auto', choices=['auto', 'default', 'fork', 'forkserver', 'spawn'], help='DataLoader 多进程启动方式，默认 auto 优先选择更稳的方式')
    train_parser.add_argument('--cache_dir', type=str, default='', help='datasets cache 目录；默认使用 Hugging Face 默认缓存位置')
    train_parser.add_argument('--deterministic', action='store_true', help='启用更强确定性；会牺牲部分训练吞吐')
    train_parser.add_argument('--pretrained', action='store_true', help='使用预训练权重')
    train_parser.add_argument('--restart', action='store_true', help='自动从 checkpoints/主干名 中最新的 epoch checkpoint 恢复训练')
    train_parser.add_argument('--onnx_opset', type=int, default=DEFAULT_ONNX_OPSET, help='训练结束后导出 ONNX 时使用的 opset，默认 12')

    # ========== ONNX导出模式 ==========
    onnx_parser = subparsers.add_parser('export_onnx', help='将 weights 中的 pth 权重转换为 ONNX')
    onnx_parser.add_argument('--backbone', type=str, default='mobilenetv2_050', help='主干网络名称，必须与 pth 对应')
    onnx_parser.add_argument('--weights', type=str, default='', help='指定单个 pth 文件路径；为空时扫描 weights_dir 下所有 pth')
    onnx_parser.add_argument('--weights_dir', type=str, default='weights', help='批量转换时扫描的权重目录')
    onnx_parser.add_argument('--output', type=str, default='', help='单个权重导出时的 ONNX 输出路径')
    onnx_parser.add_argument('--opset', type=int, default=DEFAULT_ONNX_OPSET, help='导出 ONNX 时使用的 opset，默认 12')

    # ========== 评估模式 ==========
    evaluate_parser = subparsers.add_parser('evaluate', help='评估模型')
    evaluate_parser.add_argument('--backbone', type=str, default='', help='主干网络名称；与 --weights 同时为空时扫描 weights 下全部 pth')
    evaluate_parser.add_argument('--weights', type=str, default='', help='权重文件路径；与 --backbone 同时为空时扫描 weights 下全部 pth')
    evaluate_parser.add_argument('--repo_id', type=str, default=DEFAULT_TRAIN_REPO_ID, help='Hugging Face 数据集仓库名，例如 zhouxzh/retinaface_widerface')
    evaluate_parser.add_argument('--download_root', type=str, default=DEFAULT_DOWNLOAD_ROOT, help='远程数据集下载到本地的根目录，默认 data')
    evaluate_parser.add_argument('--cache_dir', type=str, default='', help='datasets cache 目录；默认使用 Hugging Face 默认缓存位置')
    evaluate_parser.add_argument('--batch_size', type=int, default=DEFAULT_EVAL_BATCH_SIZE, help='评估批次大小，默认 16')
    evaluate_parser.add_argument('--num_workers', type=int, default=DEFAULT_EVAL_WORKERS, help='评估 DataLoader worker 数，默认 8')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        if hasattr(args, 'backbone') and not (args.command == 'evaluate' and not (args.backbone or '').strip()):
            get_backbone_cfg(args.backbone)
    except ValueError as error:
        parser.error(str(error))

    if args.command in {'predict', 'video', 'fps'} and hasattr(args, 'weights') and not args.weights:
        args.weights = get_default_weight_path(args.backbone)

    # 根据命令执行对应功能
    if args.command == 'predict':
        run_predict(args.input, args.output, args.backbone, args.weights)

    elif args.command == 'video':
        video_path = 0 if args.input == '0' else args.input
        run_video(video_path, args.output, args.fps, args.backbone, args.weights)

    elif args.command == 'fps':
        run_fps(args.image, args.test_interval, args.backbone, args.weights)

    elif args.command == 'train':
        cfg = get_backbone_cfg(args.backbone)
        run_train(args, cfg)

    elif args.command == 'export_onnx':
        run_export_onnx(args)

    elif args.command == 'evaluate':
        eval_backbone, eval_weights = resolve_evaluate_args(args)
        run_batch_evaluate(
            backbone=eval_backbone,
            weights=eval_weights,
            repo_id=args.repo_id,
            download_root=args.download_root,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()


