import os
import random
import re

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from retinaface.dataset import create_train_dataloader
from retinaface.losses import MultiBoxLoss, get_lr_scheduler, set_optimizer_lr, weights_init
from retinaface.model import RetinaFace
from retinaface.postprocess import Anchors


class LossHistory:
    def __init__(self, tensorboard_dir, model, input_shape):
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(tensorboard_dir)
        try:
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except Exception:
            pass

    def append_epoch_loss(self, epoch, total_loss, conf_loss, regression_loss, landmark_loss):
        self.writer.add_scalar('loss/total', total_loss, epoch)
        self.writer.add_scalar('loss/conf', conf_loss, epoch)
        self.writer.add_scalar('loss/regression', regression_loss, epoch)
        self.writer.add_scalar('loss/landmark', landmark_loss, epoch)

    def append_step_loss(self, global_step, total_loss, conf_loss, regression_loss, landmark_loss, lr):
        self.writer.add_scalar('step_loss/total', total_loss, global_step)
        self.writer.add_scalar('step_loss/conf', conf_loss, global_step)
        self.writer.add_scalar('step_loss/regression', regression_loss, global_step)
        self.writer.add_scalar('step_loss/landmark', landmark_loss, global_step)
        self.writer.add_scalar('step/lr', lr, global_step)


def create_run_dirs(logs_dir, backbone_name, epochs):
    backbone_tag = backbone_name.replace('/', '__').replace(' ', '_')
    run_name = backbone_tag
    tensorboard_dir = os.path.join(logs_dir, run_name)
    checkpoints_dir = os.path.join('checkpoints', run_name)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    return tensorboard_dir, checkpoints_dir, backbone_tag


def configure_runtime(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = not deterministic
        torch.backends.cudnn.allow_tf32 = not deterministic

    if hasattr(torch, 'set_float32_matmul_precision') and not deterministic:
        torch.set_float32_matmul_precision('high')

    cudnn.deterministic = deterministic
    cudnn.benchmark = not deterministic and torch.cuda.is_available()
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def run_epoch(model_train, loss_history, optimizer, criterion, epoch, epoch_step, dataloader, anchors, cfg, cuda, phase):
    total_r_loss = 0.0
    total_c_loss = 0.0
    total_landmark_loss = 0.0
    processed_steps = 0
    is_train = phase == 'train'

    if is_train:
        model_train.train()
    else:
        model_train.eval()

    for iteration, batch in enumerate(dataloader):
        if iteration >= epoch_step:
            break

        images, targets = batch
        if images.numel() == 0:
            continue

        if cuda:
            images = images.cuda(non_blocking=True)
            targets = [annotation.cuda(non_blocking=True) for annotation in targets]

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            out = model_train(images)
            r_loss, c_loss, landm_loss = criterion(out, anchors, targets)
            loss = cfg['loc_weight'] * r_loss + c_loss + landm_loss
            if is_train:
                loss.backward()
                optimizer.step()

        processed_steps += 1
        current_conf_loss = c_loss.item()
        current_regression_loss = cfg['loc_weight'] * r_loss.item()
        current_landmark_loss = landm_loss.item()
        current_total_loss = current_conf_loss + current_regression_loss + current_landmark_loss
        total_c_loss += current_conf_loss
        total_r_loss += current_regression_loss
        total_landmark_loss += current_landmark_loss

        if is_train:
            current_lr = optimizer.param_groups[0]['lr']
            global_step = epoch * epoch_step + processed_steps
            loss_history.append_step_loss(
                global_step,
                current_total_loss,
                current_conf_loss,
                current_regression_loss,
                current_landmark_loss,
                current_lr,
            )

    if processed_steps == 0:
        raise ValueError(f'{phase} 阶段没有产生有效 batch，请检查 parquet 数据集标注。')

    avg_conf_loss = total_c_loss / processed_steps
    avg_regression_loss = total_r_loss / processed_steps
    avg_landmark_loss = total_landmark_loss / processed_steps
    total_loss = avg_conf_loss + avg_regression_loss + avg_landmark_loss
    loss_history.append_epoch_loss(epoch + 1, total_loss, avg_conf_loss, avg_regression_loss, avg_landmark_loss)
    return {
        'total': total_loss,
        'conf': avg_conf_loss,
        'regression': avg_regression_loss,
        'landmark': avg_landmark_loss,
        'steps': processed_steps,
    }


def export_onnx(weights_path, onnx_path, cfg, opset_version=17):
    export_model = RetinaFace(cfg=cfg, pretrained=False, mode='eval')
    checkpoint = torch.load(weights_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    export_model.load_state_dict(state_dict)
    export_model.eval()

    input_size = cfg['train_image_size']
    dummy_input = torch.randn(1, 3, input_size, input_size)
    torch.onnx.export(
        export_model,
        dummy_input,
        onnx_path,
        input_names=['images'],
        output_names=['boxes', 'scores', 'landmarks'],
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )


def find_latest_epoch_checkpoint(checkpoints_dir, backbone_tag):
    if not os.path.isdir(checkpoints_dir):
        return ''

    latest_epoch = -1
    latest_checkpoint_path = ''
    pattern = re.compile(rf'^{re.escape(backbone_tag)}_epoch_(\d+)_loss_[0-9.]+\.pth$')
    for checkpoint_name in os.listdir(checkpoints_dir):
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
        if not os.path.isfile(checkpoint_path):
            continue
        matched = pattern.match(checkpoint_name)
        if matched is None:
            continue
        checkpoint_epoch = int(matched.group(1))
        if checkpoint_epoch > latest_epoch:
            latest_epoch = checkpoint_epoch
            latest_checkpoint_path = checkpoint_path
    return latest_checkpoint_path


def load_training_checkpoint(checkpoint_path, model, cuda):
    device = torch.device('cuda' if cuda else 'cpu')
    model_dict = model.state_dict()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        checkpoint_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint.get('optimizer_state_dict')
        completed_epoch = int(checkpoint.get('epoch', 0))
        best_loss = float(checkpoint.get('best_loss', float('inf')))
        best_state_dict = checkpoint.get('best_model_state_dict')
    else:
        checkpoint_state_dict = checkpoint
        optimizer_state_dict = None
        completed_epoch = 0
        best_loss = float('inf')
        best_state_dict = None
        matched = re.search(r'_epoch_(\d+)_loss_', os.path.basename(checkpoint_path))
        if matched is not None:
            completed_epoch = int(matched.group(1))

    checkpoint_state_dict = {
        key: value for key, value in checkpoint_state_dict.items()
        if key in model_dict and np.shape(model_dict[key]) == np.shape(value)
    }
    model_dict.update(checkpoint_state_dict)
    model.load_state_dict(model_dict)
    if best_state_dict is not None:
        best_state_dict = {
            key: value.detach().cpu().clone() for key, value in best_state_dict.items()
            if key in model_dict and np.shape(model_dict[key]) == np.shape(value)
        }
    return completed_epoch, optimizer_state_dict, best_loss, best_state_dict


def run_train(args, cfg):
    cuda = torch.cuda.is_available()
    dataset_repo_id = args.repo_id
    logs_dir = 'logs'
    weights_dir = 'weights'

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    tensorboard_dir, checkpoints_dir, backbone_tag = create_run_dirs(logs_dir, args.backbone, args.epochs)
    restart_checkpoint_path = ''
    if args.restart:
        restart_checkpoint_path = find_latest_epoch_checkpoint(checkpoints_dir, backbone_tag)
        if not restart_checkpoint_path:
            raise FileNotFoundError(f'未找到可重启的 checkpoint，请检查 {checkpoints_dir} 是否存在按 epoch 保存的权重。')

    configure_runtime(args.seed, args.deterministic)
    model = RetinaFace(cfg=cfg, pretrained=args.pretrained)
    if not args.pretrained and not restart_checkpoint_path:
        weights_init(model)

    start_epoch = 0
    optimizer_state_dict = None
    best_loss = float('inf')
    best_state_dict = None
    if restart_checkpoint_path:
        print(f'重启训练，加载 checkpoint: {restart_checkpoint_path}')
        start_epoch, optimizer_state_dict, best_loss, best_state_dict = load_training_checkpoint(restart_checkpoint_path, model, cuda)

    print(f'TensorBoard logs: {tensorboard_dir}')
    print(f'Checkpoints: {checkpoints_dir}')
    loss_history = LossHistory(tensorboard_dir, model, input_shape=(cfg['train_image_size'], cfg['train_image_size']))
    criterion = MultiBoxLoss(2, 0.35, 7, cfg['variance'], cuda)
    if cuda:
        model = model.cuda()
    model_train = model

    anchors = Anchors(cfg, image_size=(cfg['train_image_size'], cfg['train_image_size'])).get_anchors()
    if cuda:
        anchors = anchors.cuda()

    batch_size = args.batch_size
    epochs = args.epochs
    num_workers = max(0, args.num_workers)
    if batch_size < 2:
        raise ValueError('batch_size 不能小于 2，当前实现依赖 BatchNorm。')

    train_dataset, gen = create_train_dataloader(
        repo_id=dataset_repo_id,
        img_size=cfg['train_image_size'],
        batch_size=batch_size,
        download_root=args.download_root,
        num_workers=num_workers,
        pin_memory=cuda,
        seed=args.seed,
        prefetch_factor=args.prefetch_factor,
        worker_start_method=args.worker_start_method,
        cache_dir=args.cache_dir,
    )
    epoch_step = len(gen)
    if epoch_step == 0:
        raise ValueError('数据量过小，无法组成一个完整 batch，请检查数据集或调小 batch_size。')
    optimizer = optim.Adam(model.parameters(), args.lr)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    lr_scheduler_func = get_lr_scheduler('cos', args.lr, 1e-6, epochs)

    if start_epoch >= epochs:
        raise ValueError(f'重启 checkpoint 已完成到第 {start_epoch} 个 epoch，当前 --epochs={epochs}，没有可继续训练的轮数。')

    if start_epoch > 0:
        print(f'从 epoch {start_epoch + 1}/{epochs} 继续训练。')

    for epoch in range(start_epoch, epochs):
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        train_metrics = run_epoch(
            model_train,
            loss_history,
            optimizer,
            criterion,
            epoch,
            epoch_step,
            gen,
            anchors,
            cfg,
            cuda,
            phase='train',
        )

        monitor_loss = train_metrics['total']
        is_best = monitor_loss < best_loss
        if is_best:
            best_loss = monitor_loss
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        epoch_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        checkpoint_payload = {
            'epoch': epoch + 1,
            'backbone': args.backbone,
            'model_state_dict': epoch_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'best_model_state_dict': best_state_dict,
        }
        epoch_weights_path = os.path.join(checkpoints_dir, f'{backbone_tag}_epoch_{epoch + 1:03d}_loss_{monitor_loss:.4f}.pth')
        torch.save(checkpoint_payload, epoch_weights_path)
        print(
            f'Epoch {epoch + 1}/{epochs} '
            f'train_loss={train_metrics["total"]:.4f} '
            f'saved checkpoint: {epoch_weights_path}'
        )

    if best_state_dict is not None:
        best_weights_path = os.path.join(weights_dir, f'retinaface_{backbone_tag}.pth')
        onnx_path = os.path.join(weights_dir, f'retinaface_{backbone_tag}.onnx')
        torch.save(best_state_dict, best_weights_path)
        print(f'Best weights saved to: {best_weights_path}')
        try:
            export_onnx(best_weights_path, onnx_path, cfg, opset_version=args.onnx_opset)
            print(f'ONNX exported to: {onnx_path} (opset={args.onnx_opset})')
        except Exception as error:
            print(f'ONNX 导出失败，请稍后执行 main.py export_onnx 手动转换。错误信息: {error}')

    loss_history.writer.close()
    print('训练完成！')
