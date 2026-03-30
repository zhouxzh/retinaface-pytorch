## Retinaface：人脸检测模型在Pytorch当中的实现
---

## 目录
1. [项目结构 Structure](#项目结构)
2. [仓库更新 Top News](#仓库更新)
3. [性能情况 Performance](#性能情况)
4. [所需环境 Environment](#所需环境)
5. [文件下载 Download](#文件下载)
6. [预测步骤 How2predict](#预测步骤)
7. [训练步骤 How2train](#训练步骤)
8. [评估步骤 Eval](#评估步骤)
9. [主干推荐 Backbone Recommendation](#主干推荐)
10. [主干清单 Backbone Catalog](#主干清单)
11. [输入尺寸与兼容性 Input Size & Compatibility](#输入尺寸与兼容性)
12. [参考资料 Reference](#Reference)
13. [常见问题汇总 FAQ](#常见问题汇总-FAQ)

## 项目结构
```
retinaface-pytorch/
├── retinaface/              # 教学版核心模块
│   ├── __init__.py          # 包入口
│   ├── config.py            # 配置文件
│   ├── blocks.py            # FPN、SSH、MobileNet 等基础模块
│   ├── model.py             # RetinaFace 主模型
│   ├── dataset.py           # 基于 Hugging Face datasets 的 parquet 数据读取与增强
│   ├── losses.py            # 匹配逻辑与损失函数
│   ├── trainer.py           # 训练循环与日志记录
│   ├── inference.py         # 推理与预测入口
│   ├── postprocess.py       # Anchor、decode、NMS、坐标映射
│   └── evaluate.py          # WiderFace 评估
├── weights/                 # 预训练权重
├── examples/                # 示例图片
├── outputs/                 # 输出结果
├── data/                    # 训练数据集
├── logs/                    # 训练日志
└── main.py                  # 统一入口脚本
```

这个项目现在采用教学优先的简化结构：

- 不再把模型、训练、工具拆成多级子目录。
- 把同一教学主题的代码收敛到少量直观文件中。
- 初学者进入 retinaface 目录后，可以直接按 模型 -> 数据 -> 损失 -> 训练 -> 推理 -> 评估 的顺序阅读。

## Top News
**`2022-03`**:**进行了大幅度的更新，支持step、cos学习率下降法、支持adam、sgd优化器选择、支持学习率根据batch_size自适应调整。**  
BiliBili视频中的原仓库地址为：https://github.com/bubbliiiing/retinaface-pytorch/tree/bilibili

**`2020-09`**:**仓库创建，支持模型训练，大量的注释，多个主干的选择，多个可调整参数。**   

## 性能情况
| 训练数据集 | 权值文件名称 | 测试数据集 | 输入图片大小 | Easy | Medium | Hard |
| :-----: | :-----: | :------: | :------: | :------: | :-----: | :-----: |
| Widerface-Train | Retinaface_mobilenet0.25.pth | Widerface-Val | 1280x1280 | 89.76% | 86.96% | 74.69% |
| Widerface-Train | Retinaface_resnet50.pth | Widerface-Val | 1280x1280 | 94.72% | 93.13% | 84.48% |

## 所需环境
当前项目已经在现有的 conda 虚拟环境 retinaface 中验证通过，不再要求使用旧版 PyTorch 1.2.0。

当前验证环境如下：

| 组件 | 版本 |
| :--- | :--- |
| Python | 3.13.12 |
| torch | 2.10.0 |
| torchvision | 0.25.0 |
| numpy | 2.4.3 |
| scipy | 1.17.1 |
| opencv-python | 4.13.0.92 |
| tqdm | 4.67.3 |
| Pillow | 12.1.1 |
| tensorboard | 2.20.0 |
| timm | 1.0.26 |
| onnx | 1.19.1 |

说明：
- requirements.txt 已同步为当前可运行环境的核心依赖版本。
- h5py 不属于当前项目运行所需依赖，已从 requirements.txt 移除。
- torchsummary 仅在 retinaface/tools/model_summary.py 中查看模型结构时需要，默认不作为核心依赖安装。
- 如果需要训练结束后自动导出 ONNX，请确保已安装 onnx 依赖。
- 下文的 timm 主干清单基于当前验证环境中的 timm 1.0.26 实测枚举；如果你切回别的 timm 版本，型号列表可能会有少量差异。

## 文件下载
训练所需的Retinaface_resnet50.pth等文件可以在百度云下载。    
链接: https://pan.baidu.com/s/1Jt9Bo2UVP03bmEMuUpk_9Q 提取码: qknw     

数据集可以在如下连接里下载。      
链接: https://pan.baidu.com/s/1bsgay9iMihPlAKE49aWNTA 提取码: bhee    

## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，并使用你现有的 conda 环境 retinaface 运行统一入口 main.py：

**批量图片检测（默认）：**
```bash
conda run -n retinaface python main.py predict --input examples/ --output outputs/
```

**视频检测：**
```bash
# 检测视频文件
conda run -n retinaface python main.py video --input test.mp4 --output result.mp4 --fps 25.0

# 检测摄像头
conda run -n retinaface python main.py video --input 0
```

**FPS测试：**
```bash
conda run -n retinaface python main.py fps --test_interval 100
```

### b、使用自己训练的权重
1. 按照训练步骤训练。
2. 在 retinaface/inference.py 文件里面，修改 model_path 和 backbone 使其对应训练好的文件。
```python
_defaults = {
    "model_path"        : 'weights/Retinaface_mobilenet0.25.pth',
    "backbone"          : 'mobilenet',
    "confidence"        : 0.5,
    "nms_iou"           : 0.45,
    "cuda"              : True,
    "input_shape"       : [1280, 1280, 3],
    "letterbox_image"   : True
}
```
3. 运行 main.py 进行检测。  

## 训练步骤
1. 本文使用 widerface 数据集进行训练。
2. 可通过上述百度网盘下载 widerface 数据集。
3. 覆盖根目录下的 data 文件夹。
4. 先准备已经转换好的 Hugging Face parquet 数据集，目录结构需要包含 train/ 和 val/（或 validation/）split。
5. 使用统一的 main.py 入口进行训练：

**从预训练权重开始训练（推荐）：**
```bash
conda run -n retinaface python main.py train --backbone mobilenet --batch_size 32 --epochs 100 --pretrained
```

**自动从上一次训练中断处继续训练：**
```bash
conda run -n retinaface python main.py train --backbone mobilenet --batch_size 32 --epochs 100 --restart
```

**指定 parquet 数据集目录训练：**
```bash
conda run -n retinaface python main.py train --repo_id zhouxzh/retinaface_widerface --download_root data --backbone mobilenet --batch_size 32 --pretrained
```

**批量训练 README 中推荐骨干：**
```bash
bash run.sh
```

**训练参数说明：**
- `--backbone`: 主干网络，可选自定义 mobilenet，或 timm 中以 mobilenet / resnet 开头的骨干
- `--batch_size`: 批次大小，默认 32，按 32G 显存单卡做了上调
- `--epochs`: 训练轮数
- `--lr`: 学习率，默认 4e-3，和更大的默认 batch size 一起调整
- `--num_workers`: DataLoader worker 数，默认 8
- `--repo_id`: Hugging Face 数据集仓库名，例如 zhouxzh/retinaface_widerface
- `--download_root`: 远程数据集下载到本地的根目录，默认 data
- `--seed`: 随机种子，默认 3407
- `--prefetch_factor`: DataLoader 预取批次数，默认 4
- `--worker_start_method`: 多进程启动方式，默认 auto，会优先选更稳的方式
- `--cache_dir`: datasets cache 目录；默认使用 Hugging Face 默认缓存位置
- `--deterministic`: 启用更强确定性；会牺牲部分吞吐
- `--pretrained`: 使用主干网络预训练权重
- `--restart`: 自动从 checkpoints/主干名 中最新的 epoch checkpoint 恢复训练，并从记录的下一个 epoch 继续

默认训练配置说明：
- 当前默认值更偏向 32G 显存单卡环境，例如 RTX 5090D 这类卡。
- 默认训练主干仍然是 mobilenet，输入尺寸仍然是 840x840，没有修改数据尺度。
- 默认数据入口已经切换为 Hugging Face 远程数据集下载 + parquet 训练流，不再兼容本地 parquet 目录探测。
- 训练阶段只使用 train split，并只记录 train loss；val split 留给 evaluate 计算 mAP。
- 训练开始前会先通过 Hugging Face 默认缓存下载数据，再把需要的 parquet 文件同步到 data/仓库名 目录。
- datasets 的缓存默认交给 Hugging Face 自己管理，不再额外写入 data/retinaface_widerface 或 data/huggingface。
- 默认会固定 Python、NumPy、PyTorch 和 DataLoader worker 的随机种子，同时对多进程加载做更稳的启动方式选择。
- 如果你切换到更重的 timm backbone，例如 resnet101、mobilenetv5_base 或更大的 resnetv2 变体，建议把 batch size 再按实际显存回调。

6. TensorBoard 日志会保存在 logs/主干名 目录。  
7. 每次训练都会在根目录的 checkpoints/主干名 目录下按 epoch 保存 checkpoint，不再额外写出 best.pth 和 last.pth。`--restart` 会自动读取其中最新的 epoch checkpoint。  
8. 训练结束后会将本次最优权重导出到 weights/retinaface_主干名.pth。  
9. 训练结束后会自动基于最优权重导出对应的 ONNX 文件 weights/retinaface_主干名.onnx。  
10. 根目录下的 run.sh 会按顺序训练 README 里推荐且适合直接起手的 backbone，默认使用 pretrained、batch_size 32、lr 4e-3、num_workers 8。  

## 评估步骤
1. 下载好百度网盘上上传的数据集，其中包括了验证集，解压在根目录下。
2. 使用统一的 main.py 入口进行评估：
```bash
conda run -n retinaface python main.py evaluate --backbone mobilenet --weights weights/Retinaface_mobilenet0.25.pth
```

**默认批量评估 weights 下全部权重：**
```bash
conda run -n retinaface python main.py evaluate
```

**评估参数说明：**
- `--backbone`: 主干网络，可选自定义 mobilenet，或 timm 中以 mobilenet / resnet 开头的骨干
- `--weights`: 权重文件路径
- `--repo_id`: Hugging Face 数据集仓库名，例如 zhouxzh/retinaface_widerface
- `--download_root`: 远程数据集下载到本地的根目录，默认 data
- `--cache_dir`: datasets cache 目录；默认使用 Hugging Face 默认缓存位置
- `--batch_size`: 评估批次大小，默认 16
- `--num_workers`: 评估 DataLoader worker 数，默认 8

评估输出说明：
- 预测结果会保存到 outputs/evaluate/主干名/权重名/时间戳/predictions 目录。
- 评估日志会保存到同级目录下的 evaluate.log 文件。
- 评估图片会直接来自 Hugging Face parquet 的 val/validation split，不再依赖 data/widerface/val/images 本地目录。
- 评估默认走批量推理流程，不再走逐张图片的单独评估函数。
- 日志中会记录 backbone、backbone_source、通道配置、输入配置、图片数量以及 Easy/Medium/Hard AP。

## 主干推荐
当前代码已经支持两类主干：

- 仓库内置的自定义 mobilenet0.25。
- timm 中所有以 mobilenet 或 resnet 开头，且可以通过 features_only 接出至少 3 层特征的模型。

结合当前 RetinaFace 结构、FPN 接法和常见显存预算，建议优先这样选：

| 场景 | 推荐型号 | 原因 |
| :--- | :--- | :--- |
| 入门 / 低显存训练 | mobilenet、mobilenetv2_035、mobilenetv2_050、mobilenetv3_small_050 | 参数量和通道数较小，适合先跑通训练流程 |
| 速度优先推理 | mobilenetv2_050、mobilenetv2_100、mobilenetv3_small_100、resnet18 | FPN 输入通道适中，推理开销更可控 |
| 精度与速度平衡 | mobilenetv2_100、mobilenetv3_large_100、mobilenetv4_conv_small、resnet34、resnet50 | 通道规模适中，通常比超轻量模型更稳定 |
| 精度优先 | resnet50、resnet101、resnetv2_50、mobilenetv5_base | 深层语义更强，但训练和推理成本更高 |
| 不建议直接起手 | resnet50x64_clip、resnetv2_152x4_bit、resnetv2_101x3_bit | FPN 输入通道非常大，显存和算力压力明显 |

补充建议：

- 如果你只是想复现实验结果，优先使用仓库内置 mobilenet 或 resnet50。
- 如果你只有 8GB 左右显存，优先从 mobilenetv2_050、mobilenetv2_100、mobilenetv3_small_100、resnet18 开始。
- 如果你要在 timm 主干上继续训练，最好重新训练整套 RetinaFace 权重，不要混用仓库原始 mobilenet0.25 权重。

## 主干清单
下面清单基于当前环境中的 timm 1.0.26 实测列出。格式为：

模型名 -> RetinaFace 当前接入 FPN 的最后三层通道

### 自定义主干
- mobilenet -> [64, 128, 256]，这是仓库内置的 MobileNetV1 0.25，不属于 timm。

### timm mobilenet 家族（37 个）
- mobilenet_edgetpu_100 -> [48, 96, 192]
- mobilenet_edgetpu_v2_l -> [96, 240, 384]
- mobilenet_edgetpu_v2_m -> [80, 192, 320]
- mobilenet_edgetpu_v2_s -> [64, 160, 256]
- mobilenet_edgetpu_v2_xs -> [48, 144, 192]
- mobilenetv1_100 -> [256, 512, 1024]
- mobilenetv1_100h -> [256, 512, 1024]
- mobilenetv1_125 -> [320, 640, 1280]
- mobilenetv2_035 -> [16, 32, 112]
- mobilenetv2_050 -> [16, 48, 160]
- mobilenetv2_075 -> [24, 72, 240]
- mobilenetv2_100 -> [32, 96, 320]
- mobilenetv2_110d -> [32, 104, 352]
- mobilenetv2_120d -> [40, 112, 384]
- mobilenetv2_140 -> [48, 136, 448]
- mobilenetv3_large_075 -> [32, 88, 720]
- mobilenetv3_large_100 -> [40, 112, 960]
- mobilenetv3_large_150d -> [64, 168, 1440]
- mobilenetv3_rw -> [40, 112, 960]
- mobilenetv3_small_050 -> [16, 24, 288]
- mobilenetv3_small_075 -> [24, 40, 432]
- mobilenetv3_small_100 -> [24, 48, 576]
- mobilenetv4_conv_aa_large -> [96, 192, 960]
- mobilenetv4_conv_aa_medium -> [80, 160, 960]
- mobilenetv4_conv_blur_medium -> [80, 160, 960]
- mobilenetv4_conv_large -> [96, 192, 960]
- mobilenetv4_conv_medium -> [80, 160, 960]
- mobilenetv4_conv_small -> [64, 96, 960]
- mobilenetv4_conv_small_035 -> [24, 32, 336]
- mobilenetv4_conv_small_050 -> [32, 48, 480]
- mobilenetv4_hybrid_large -> [96, 192, 960]
- mobilenetv4_hybrid_large_075 -> [72, 144, 720]
- mobilenetv4_hybrid_medium -> [80, 160, 960]
- mobilenetv4_hybrid_medium_075 -> [64, 120, 720]
- mobilenetv5_300m -> [256, 640, 1280]
- mobilenetv5_300m_enc -> [256, 640, 1280]
- mobilenetv5_base -> [256, 512, 1024]

### timm resnet 家族（75 个）
- resnet101 -> [512, 1024, 2048]
- resnet101_clip -> [512, 1024, 2048]
- resnet101_clip_gap -> [512, 1024, 2048]
- resnet101c -> [512, 1024, 2048]
- resnet101d -> [512, 1024, 2048]
- resnet101s -> [512, 1024, 2048]
- resnet10t -> [128, 256, 512]
- resnet14t -> [512, 1024, 2048]
- resnet152 -> [512, 1024, 2048]
- resnet152c -> [512, 1024, 2048]
- resnet152d -> [512, 1024, 2048]
- resnet152s -> [512, 1024, 2048]
- resnet18 -> [128, 256, 512]
- resnet18d -> [128, 256, 512]
- resnet200 -> [512, 1024, 2048]
- resnet200d -> [512, 1024, 2048]
- resnet26 -> [512, 1024, 2048]
- resnet26d -> [512, 1024, 2048]
- resnet26t -> [512, 1024, 2048]
- resnet32ts -> [512, 1536, 1536]
- resnet33ts -> [512, 1536, 1280]
- resnet34 -> [128, 256, 512]
- resnet34d -> [128, 256, 512]
- resnet50 -> [512, 1024, 2048]
- resnet50_clip -> [512, 1024, 2048]
- resnet50_clip_gap -> [512, 1024, 2048]
- resnet50_gn -> [512, 1024, 2048]
- resnet50_mlp -> [512, 1024, 2048]
- resnet50c -> [512, 1024, 2048]
- resnet50d -> [512, 1024, 2048]
- resnet50s -> [512, 1024, 2048]
- resnet50t -> [512, 1024, 2048]
- resnet50x16_clip -> [768, 1536, 3072]
- resnet50x16_clip_gap -> [768, 1536, 3072]
- resnet50x4_clip -> [640, 1280, 2560]
- resnet50x4_clip_gap -> [640, 1280, 2560]
- resnet50x64_clip -> [1024, 2048, 4096]
- resnet50x64_clip_gap -> [1024, 2048, 4096]
- resnet51q -> [512, 1536, 2048]
- resnet61q -> [512, 1536, 2048]
- resnetaa101d -> [512, 1024, 2048]
- resnetaa34d -> [128, 256, 512]
- resnetaa50 -> [512, 1024, 2048]
- resnetaa50d -> [512, 1024, 2048]
- resnetblur101d -> [512, 1024, 2048]
- resnetblur18 -> [128, 256, 512]
- resnetblur50 -> [512, 1024, 2048]
- resnetblur50d -> [512, 1024, 2048]
- resnetrs101 -> [512, 1024, 2048]
- resnetrs152 -> [512, 1024, 2048]
- resnetrs200 -> [512, 1024, 2048]
- resnetrs270 -> [512, 1024, 2048]
- resnetrs350 -> [512, 1024, 2048]
- resnetrs420 -> [512, 1024, 2048]
- resnetrs50 -> [512, 1024, 2048]
- resnetv2_101 -> [512, 1024, 2048]
- resnetv2_101d -> [512, 1024, 2048]
- resnetv2_101x1_bit -> [512, 1024, 2048]
- resnetv2_101x3_bit -> [1536, 3072, 6144]
- resnetv2_152 -> [512, 1024, 2048]
- resnetv2_152d -> [512, 1024, 2048]
- resnetv2_152x2_bit -> [1024, 2048, 4096]
- resnetv2_152x4_bit -> [2048, 4096, 8192]
- resnetv2_18 -> [128, 256, 512]
- resnetv2_18d -> [128, 256, 512]
- resnetv2_34 -> [128, 256, 512]
- resnetv2_34d -> [128, 256, 512]
- resnetv2_50 -> [512, 1024, 2048]
- resnetv2_50d -> [512, 1024, 2048]
- resnetv2_50d_evos -> [512, 1024, 2048]
- resnetv2_50d_frn -> [512, 1024, 2048]
- resnetv2_50d_gn -> [512, 1024, 2048]
- resnetv2_50t -> [512, 1024, 2048]
- resnetv2_50x1_bit -> [512, 1024, 2048]
- resnetv2_50x3_bit -> [1536, 3072, 6144]

## 输入尺寸与兼容性
这份工程是 RetinaFace，不是标准 SSD；但它的检测头、anchor 和多尺度特征接法与 SSD 系列非常接近。

### 当前输入尺寸
- 训练阶段固定输入尺寸为 840x840。
- 训练数据增强会把图片整理到 840x840，再参与标注归一化和 anchor 匹配。
- 训练阶段 anchor 也是按 840x840 生成。
- 推理阶段默认输入尺寸为 1280x1280。
- 推理默认开启 letterbox，会先把原图按比例缩放并填充到 1280x1280。
- 如果关闭 letterbox，则会按原图尺寸动态生成 anchor，不强制固定成 1280x1280。

### 兼容性结论
- 结构兼容：当前 timm 中以 mobilenet 或 resnet 开头的 112 个模型，全部可以通过本项目的 features_only + FPN 接法。
- 名称兼容：当前代码只接受自定义 mobilenet，以及 timm 中以 mobilenet / resnet 开头的主干，其他家族不会被识别。
- 权重兼容：仓库自带的 mobilenet0.25 权重只能用于内置 mobilenet，不能直接加载到 timm 的 mobilenetv2、mobilenetv3、mobilenetv4、mobilenetv5 上。
- 权重文件兼容：如果你传入的是 timm 主干名称，但没有对应训练好的 RetinaFace 权重文件，加载阶段会失败。
- 训练 / 推理尺寸兼容：网络本身是全卷积的，所以 840x840 训练、1280x1280 推理在形状上是兼容的；但这属于分布不一致，速度、显存占用和精度表现可能会变化。
- 大通道主干兼容：像 resnet50x64_clip、resnetv2_152x4_bit 这类大通道骨干虽然结构上能接入，但显存占用、训练速度和收敛成本会明显更高。

### 实际选型建议
- 想先跑通项目：选 mobilenet 或 resnet50。
- 想兼顾速度和效果：优先试 mobilenetv2_100、mobilenetv3_large_100、resnet18、resnet34。
- 想做更强 backbone 的实验：优先试 resnet50、resnet101、resnetv2_50，不建议一开始就上极大通道模型。
- 如果准备写论文或做系统性对比，建议固定训练尺寸和推理尺寸各做一组对照，不要只换 backbone。

### Reference
https://github.com/bubbliiiing/retinaface-pytorch
https://github.com/biubug6/Pytorch_Retinaface

## 常见问题汇总 FAQ

### 1、下载与环境问题
**问：代码和权值文件在哪里下载？**
答：Github代码库地址已在对应页面提供。模型权值文件和Widerface数据集的网盘链接在README的“文件下载”部分。

**问：为什么提示 `No module named 'torch'` 或 `cv2`？**
答：确保在执行代码前已经激活了配置好各种依赖库的虚拟环境（如 `conda activate retinaface`）。部分额外提示可能是由于对应的包没有安装，请查阅 `requirements.txt`。

**问：如何判断是否在使用GPU进行训练？**
答：可以利用 `nvidia-smi` 命令行命令查看显存利用率，或者在任务管理器中查看GPU的Cuda选项卡（而非Copy）。

**问：我可以使用CPU进行训练与预测吗？**
答：在预测时，可以在 `retinaface/inference.py` 或者调用代码中将参数 `cuda` 项设置为 `False`。训练时若无可用GPU，框架会自动回落或需要手动指定device。

**问：为什么VSCODE提示一大堆错误？**
答：通常是VSCode环境或Pylance等插件未选择正确的Python解释器。不影响实际终端运行。

### 2、训练与预测问题
**问：为什么运行训练或预测时直接爆显存（OOM: CUDA out of memory）？**
答：说明当时显卡显存空间不足。可以尝试调小 `batch_size`，或者选用更轻量级的主干网络（如 `mobilenetv2_035`、`mobilenet`）。

**问：断点续练该如何操作？**
答：直接使用 `--restart`。程序会自动到 `checkpoints/主干名` 目录查找最近一次按 epoch 保存的 checkpoint，读取其中记录的 epoch 和 optimizer 状态，然后从下一个 epoch 继续训练。如果该目录下没有可用 checkpoint，会直接报错提示。

**问：不使用预训练权重可以吗？效果为什么很差？**
答：如果从零开始（不加 `--pretrained` 或不载入预先提取的主干权重），因为网络参数随机初始化，特征提取能力极差，很容易导致网络不收敛。强烈建议使用预训练权重。

**问：为什么修改了网络结构后预训练权重无法直接载入？**
答：主干结构如果发生变化，权重的Shape也会改变。PyTorch中可以编写脚本读取 `.state_dict()`，并过滤掉Shape不支持的对应层参数后再 `load_state_dict()`。如果只是增加网络后半部分的复杂度，主干部分仍可使用原有预训练权重。

**问：如何进行视频和摄像头的检测？**
答：现在已统一在 `main.py` 入口。请参考README中预测步骤的视频/摄像头指令，例如 `python main.py video --input 0` 开启本地摄像头。

**问：怎么对一个文件夹内的图片进行检测并保存？**
答：现在的 `main.py predict --input examples/ --output outputs/` 命令原生支持扫描 `input` 目录下的所有图片并将其检测结果保存在 `output` 文件夹中。

**问：检测速度（FPS）如何测算？**
答：运行 `python main.py fps` 即可进行FPS基准测试，具体FPS强依赖于硬件配置和当前选用主干网络的体量。

**问：遇到 `FileNotFoundError: No such file or directory`？**
答：请检查路径是否包含空格，相对路径和绝对目录是否正确配置。特别是在存放数据集时，检查数据的层级结构是否与代码读取逻辑匹配。

### 3、其他问题
**问：博主有没有交流群？**
答：没有没有，我没有时间管理QQ群……

**问：新手怎么学习深度学习和计算机视觉？**
答：建议先从基础的Python教学入门，学习框架（PyTorch）基础，然后从简单经典的分类、目标检测（如SSD、YOLO）等入手。花时间一行行阅读代码，理解整个代码的执行流程和特征层的形状变化。

