# Enhanced 3D Gaussian Splatting

基于 [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) 的增强实现，集成了多种改进方法，支持灵活组合使用。

## 集成的改进方法

### 1. AbsGS — 绝对梯度密度化 (Absolute Gradient Splatting)

基于 [AbsGS](https://github.com/YuxueYang1204/AbsGS) 实现，使用屏幕空间绝对梯度替代平均梯度进行高斯点云密度化，改善欠重建区域的加密效果。

**核心改进**：
- 使用 `absgrad`（绝对梯度）替代 `viewspace_points.grad`（平均梯度）指导 densification
- 支持渐进式分辨率训练（Progressive Resolution）
- 动态梯度阈值缩放（根据分辨率自动调整阈值）
- 动态点数控制：当点数超过目标值时自动提高阈值，防止 OOM

**使用方式**：
```shell
python train.py -s <dataset_path> --abs_gs
```

**相关参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--abs_gs` | False | 启用 AbsGS 绝对梯度密度化 |
| `--cap_max` | 0 | 目标点数上限（0=自动，默认2M） |
| `--progressive_resolution` | False | 启用渐进式分辨率训练 |
| `--resolution_schedule_1` | 3000 | 第一次分辨率切换的迭代 |
| `--resolution_scale_1` | 4.0 | 初始分辨率缩放（1/4） |
| `--resolution_schedule_2` | 6000 | 第二次分辨率切换的迭代 |
| `--resolution_scale_2` | 2.0 | 中间分辨率缩放（1/2） |
| `--densify_grad_threshold_scale` | False | 根据分辨率自动缩放梯度阈值 |

### 2. PixelGS — 深度感知梯度缩放

基于 [PixelGS](https://github.com/nyu-systems/PixelGS) 实现，根据高斯点到相机的深度对梯度进行缩放，使远处点的梯度不被过度抑制，改善远距离区域的重建质量。

**核心改进**：
- 梯度按 `(depth / depth_threshold)^2` 缩放，近处点梯度不变，远处点梯度被适当放大
- 深度阈值与场景范围（`cameras_extent`）关联

**使用方式**：
```shell
python train.py -s <dataset_path> --pixel_gs
```

**相关参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pixel_gs` | False | 启用 PixelGS 深度感知梯度缩放 |
| `--depth_threshold` | 0.37 | 深度阈值（乘以 cameras_extent） |

### 3. 深度先验正则化 (Depth Prior Regularization)

利用单目深度估计图作为先验，约束渲染深度与单目深度的一致性，改善几何重建质量。

**使用方式**：
```shell
python train.py -s <dataset_path> -d <depth_path>
```

**相关参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--depth_l1_weight_init` | 1.0 | 深度损失初始权重 |
| `--depth_l1_weight_final` | 0.01 | 深度损失最终权重 |

### 4. 深度一致性剪枝 (Depth Consistency Pruning)

在密度化时，根据深度一致性检查剪枝不可靠的高斯点，减少浮点伪影。

**使用方式**：
```shell
python train.py -s <dataset_path> --depth_prune
```

**相关参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--depth_prune` | False | 启用深度一致性剪枝 |
| `--depth_prune_threshold` | 0.3 | 深度一致性阈值 |
| `--depth_prune_min_views` | 2 | 最少可见视图数 |

### 5. 曝光补偿 (Exposure Compensation)

为每个相机学习一个仿射变换矩阵来补偿曝光差异，适用于不同光照/曝光条件的图像。

**使用方式**：
```shell
python train.py -s <dataset_path> --train_test_exp
```

**相关参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train_test_exp` | False | 启用曝光补偿 |
| `--exposure_lr_init` | 0.01 | 曝光参数初始学习率 |
| `--exposure_lr_final` | 0.001 | 曝光参数最终学习率 |
| `--exposure_lr_delay_steps` | 0 | 曝光学习率延迟步数 |
| `--exposure_lr_delay_mult` | 0.0 | 曝光学习率延迟乘子 |

### 6. 抗锯齿 (Anti-Aliasing)

基于 [EWA Splatting](https://github.com/graphdeco-inria/gaussian-splatting) 的抗锯齿渲染，减少高频伪影。

**使用方式**：
```shell
python train.py -s <dataset_path> --antialiasing
```

### 7. 空间正则化 (Spatial Regularization)

对高斯点施加空间正则化约束，防止点云过度聚集。

**使用方式**：
```shell
python train.py -s <dataset_path> --spatial_reg
```

**相关参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--spatial_reg` | False | 启用空间正则化 |
| `--spatial_reg_weight` | 0.1 | 正则化权重 |
| `--spatial_reg_percent` | 0.05 | 正则化采样比例 |

### 8. 梯度裁剪 (Gradient Clipping)

对优化器参数进行梯度范数裁剪，防止训练不稳定。

**使用方式**：
```shell
python train.py -s <dataset_path> --grad_clip_norm 1.0
```

### 9. 双渲染器支持

同时支持 `diff-gaussian-rasterization` 和 `gsplat` 两种渲染后端：
- **默认**：优先使用 `diff-gaussian-rasterization`
- **AbsGS**：优先使用 `diff-gaussian-rasterization`（梯度更稳定），不可用时自动降级到 `gsplat`
- 可通过 `--use_gsplat_renderer` 强制使用 gsplat 渲染器

## 组合使用

所有改进方法可以自由组合。例如，同时启用所有功能：

```shell
python train.py -s <dataset_path> \
  -d <depth_path> \
  --abs_gs \
  --pixel_gs \
  --antialiasing \
  --depth_prune \
  --train_test_exp \
  --densify_grad_threshold 0.0001 \
  --exposure_lr_init 0.001 \
  --exposure_lr_final 0.0001 \
  --exposure_lr_delay_steps 5000 \
  --exposure_lr_delay_mult 0.001 \
  --data_device cpu \
  --eval
```

仅使用 AbsGS + PixelGS + 抗锯齿：

```shell
python train.py -s <dataset_path> \
  --abs_gs \
  --pixel_gs \
  --antialiasing \
  --densify_grad_threshold 0.0002
```

## 安装

### 环境配置

```shell
conda env create --file environment.yml
conda activate gaussian_splatting
```

### 编译修改后的 diff-gaussian-rasterization

本项目对 `diff-gaussian-rasterization` 进行了修改（添加了 `absgrad` 和 `depth_threshold` 支持），需要重新编译安装：

```shell
cd submodules/diff-gaussian-rasterization
pip uninstall diff-gaussian-rasterization -y
pip install -e . --no-build-isolation
```

### （可选）安装 gsplat

如果需要使用 gsplat 渲染后端：

```shell
pip install gsplat
```

## 评估

```shell
python render.py -m <path to trained model> --train_test_exp  # 如果训练时使用了曝光补偿
python metrics.py -m <path to trained model>
```

## 项目结构

```
gaussian-splatting/
├── arguments/           # 命令行参数定义（含所有改进的参数）
├── gaussian_renderer/   # 渲染器（DGR + gsplat 双后端）
├── scene/               # 场景管理、高斯模型、数据加载
├── submodules/
│   └── diff-gaussian-rasterization/  # 修改版 DGR（支持 absgrad、depth_threshold）
├── utils/               # 工具函数
├── train.py             # 训练入口
├── render.py            # 渲染入口
└── metrics.py           # 评估指标
```

## 致谢

本项目基于以下工作：

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) — Kerbl et al., ACM TOG 2023
- [AbsGS](https://github.com/YuxueYang1204/AbsGS) — Absolute Gradient Splatting
- [PixelGS](https://github.com/nyu-systems/PixelGS) — Pixel-Aligned Gaussian Splatting
- [gsplat](https://github.com/nerfstudio-project/gsplat) — gsplat rendering backend
