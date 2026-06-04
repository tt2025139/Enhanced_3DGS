"""3D Gaussian Splatting 渲染质量评估脚本。

该脚本对训练好的 3D Gaussian Splatting 模型渲染结果进行定量评估，
计算三种广泛使用的图像质量指标：

- **SSIM** (Structural Similarity Index Measure): 结构相似性指标，
  衡量两幅图像在亮度、对比度和结构上的相似程度，取值范围 [0, 1]，
  越接近 1 表示越相似。
- **PSNR** (Peak Signal-to-Noise Ratio): 峰值信噪比，衡量渲染图像
  与真实图像之间的像素级误差，单位为 dB，值越大表示质量越好。
- **LPIPS** (Learned Perceptual Image Patch Similarity): 基于深度
  学习的感知相似度指标，使用预训练的 VGG 网络提取特征，更贴近
  人类视觉感知，值越小表示越相似。

工作原理：
    1. 遍历每个模型目录下的 train/test 子目录。
    2. 在每个子目录中寻找不同的渲染方法（method）文件夹。
    3. 对每个方法的 renders/ 和 gt/ 目录，逐图像对计算上述三种指标。
    4. 汇总结果写入 results1.json 和 per_view1.json。

依赖：
    - ``utils.loss_utils.ssim``: SSIM 的自定义 CUDA 实现
    - ``utils.image_utils.psnr``: PSNR 计算（基于 MSE）
    - ``lpipsPyTorch``: 封装了 LPIPS 官方 PyTorch 实现
"""

#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def iter_image_pairs(renders_dir, gt_dir):
    """生成渲染图像与对应真值图像的配对迭代器。

    该方法遍历渲染目录下的所有图像文件，按文件名排序，
    对每个文件名尝试在真值目录中找到同名文件。
    仅当两个文件都存在时才会 yield，确保配对完整。

    :param renders_dir: 渲染图像所在的目录（Path 对象）。
    :param gt_dir: 真值（ground truth）图像所在的目录（Path 对象）。
    :yield: (文件名, 渲染图像路径, 真值图像路径) 的三元组。
    """
    for fname in sorted(os.listdir(renders_dir)):
        render_path = renders_dir / fname
        gt_path = gt_dir / fname
        if not render_path.is_file() or not gt_path.is_file():
            continue
        yield fname, render_path, gt_path


def compute_metrics_from_dirs(renders_dir, gt_dir, desc_prefix="Metric evaluation progress"):
    """计算一个渲染/真值目录对的所有图像质量指标。

    对渲染目录和真值目录中的每对图像，依次计算 SSIM、PSNR 和 LPIPS。
    图像先通过 PIL 加载，再转换为 PyTorch 张量（值域 [0, 1]），
    并截取前 3 个通道（RGB）传输到 GPU 上。

    .. note::
        LPIPS 使用 VGG 网络作为特征提取骨干，计算量较大。
        所有指标计算均包裹在 ``torch.no_grad()`` 上下文中以节省显存。

    :param renders_dir: 渲染图像目录（Path 对象）。
    :param gt_dir: 真值图像目录（Path 对象）。
    :param desc_prefix: tqdm 进度条的前缀描述字符串。
    :return: 三元组 (summary, per_view, count)
        - summary: dict，包含三种指标的均值（"SSIM", "PSNR", "LPIPS"）。
        - per_view: dict，每个指标对应一个 {文件名: 指标值} 的子字典。
        - count: 成功计算的图像对数量。
    """
    # 每个视角/图像的独立指标值
    per_view = {
        "SSIM": {},
        "PSNR": {},
        "LPIPS": {},
    }
    # 累积和，用于最后计算平均值
    ssim_sum = 0.0
    psnr_sum = 0.0
    lpips_sum = 0.0
    count = 0

    # 将迭代器转为列表以获取总长度（用于 tqdm 进度条）
    image_pairs = list(iter_image_pairs(renders_dir, gt_dir))

    # torch.no_grad() 禁用梯度计算，节省显存并加速推理
    with torch.no_grad():
        for name, render_path, gt_path in tqdm(image_pairs, desc=desc_prefix):
            # 使用 PIL 加载图像，确保兼容各种图像格式
            render = Image.open(render_path)
            gt = Image.open(gt_path)

            # tf.to_tensor 将 PIL Image [0,255] 转为浮点张量 [0,1]，形状 (C, H, W)
            # unsqueeze(0) 添加 batch 维度 -> (1, C, H, W)
            # [:,:3,:,:] 截取前 3 个通道（RGB），丢弃可能的 alpha 通道
            # .cuda(non_blocking=True) 异步传输到 GPU，不阻塞 CPU 继续处理
            render_t = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda(non_blocking=True)
            gt_t = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda(non_blocking=True)

            # 计算三种质量指标
            # SSIM 和 PSNR 返回的是标量张量，.item() 取出 Python 浮点值
            ssim_v = float(ssim(render_t, gt_t).item())
            psnr_v = float(psnr(render_t, gt_t).item())
            # LPIPS 使用 VGG 网络作为感知特征提取器
            lpips_v = float(lpips(render_t, gt_t, net_type='vgg').item())

            # 记录每个视角的指标值
            per_view["SSIM"][name] = ssim_v
            per_view["PSNR"][name] = psnr_v
            per_view["LPIPS"][name] = lpips_v

            # 累加求和，用于计算全局均值
            ssim_sum += ssim_v
            psnr_sum += psnr_v
            lpips_sum += lpips_v
            count += 1

            # 显式释放 GPU 张量，避免显存累积（在 no_grad 上下文中不会记录计算图）
            del render_t, gt_t

    # 处理空目录的特殊情况
    if count == 0:
        empty_summary = {
            "SSIM": 0.0,
            "PSNR": 0.0,
            "LPIPS": 0.0,
        }
        return empty_summary, per_view, 0

    # 计算各指标的算术平均值
    summary = {
        "SSIM": ssim_sum / count,
        "PSNR": psnr_sum / count,
        "LPIPS": lpips_sum / count,
    }
    return summary, per_view, count

def evaluate(model_paths):
    """对一组模型目录进行完整的质量评估。

    该函数遍历每个模型目录，自动发现目录下的所有渲染方法（method），
    分别计算 train/test 两个子集的指标，并汇总输出。

    目录结构预期如下::

        <model_path>/
            train/
                <method_name>/
                    renders/   (渲染图像)
                    gt/        (真值图像)
            test/
                <method_name>/
                    renders/
                    gt/

    输出文件：
        - results1.json: 按方法和数据分割汇总的指标均值。
        - per_view1.json: 每个视角的独立指标值。

    :param model_paths: 模型目录路径的列表，每个路径包含 train/test 子目录。
    :type model_paths: list[str]
    """
    # full_dict: 全局汇总字典，结构为 {场景: {方法: {分割: {指标: 均值}}}}
    full_dict = {}
    # per_view_dict: 每个视角的详细指标，结构为 {场景: {方法: {分割: {指标: {文件名: 值}}}}}
    per_view_dict = {}

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}

            scene_path = Path(scene_dir)
            # 训练和测试两个数据分割
            split_dirs = {
                "train": scene_path / "train",
                "test": scene_path / "test",
            }

            # 自动发现所有存在的渲染方法名
            # 方法名 = train/ 或 test/ 下的子目录名
            methods = set()
            for split_dir in split_dirs.values():
                if split_dir.exists():
                    for method_name in os.listdir(split_dir):
                        if (split_dir / method_name).is_dir():
                            methods.add(method_name)

            # 按字母序排序方法名，保证输出顺序可复现
            for method in sorted(methods):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}

                # 用于跨 train/test 的全局汇总统计
                all_count = 0
                all_ssim_sum = 0.0
                all_psnr_sum = 0.0
                all_lpips_sum = 0.0
                all_per_view = {
                    "SSIM": {},
                    "PSNR": {},
                    "LPIPS": {},
                }

                # 分别处理 train 和 test 两个分割
                for split_name in ["train", "test"]:
                    method_dir = split_dirs[split_name] / method
                    gt_dir = method_dir / "gt"
                    renders_dir = method_dir / "renders"

                    # 如果 gt 或 renders 目录不存在，跳过该分割
                    if not gt_dir.exists() or not renders_dir.exists():
                        continue

                    # 调用核心指标计算函数
                    summary, per_view, split_count = compute_metrics_from_dirs(
                        renders_dir,
                        gt_dir,
                        desc_prefix=f"{split_name}"
                    )
                    if split_count == 0:
                        continue

                    # 打印当前分割的指标结果
                    print(f"  [{split_name}] SSIM : {summary['SSIM']:>12.7f}")
                    print(f"  [{split_name}] PSNR : {summary['PSNR']:>12.7f}")
                    print(f"  [{split_name}] LPIPS: {summary['LPIPS']:>12.7f}")

                    # 存储当前分割的指标
                    full_dict[scene_dir][method][split_name] = summary
                    per_view_dict[scene_dir][method][split_name] = per_view

                    # 累积跨分割的全局统计
                    # 注意：使用加权求和，每个分割按其图像数量贡献权重
                    all_count += split_count
                    all_ssim_sum += summary["SSIM"] * split_count
                    all_psnr_sum += summary["PSNR"] * split_count
                    all_lpips_sum += summary["LPIPS"] * split_count
                    # 合并每个视角的指标，为每个文件名添加 "train/" 或 "test/" 前缀以区分
                    for metric_name in ["SSIM", "PSNR", "LPIPS"]:
                        all_per_view[metric_name].update(
                            {f"{split_name}/{k}": v for k, v in per_view[metric_name].items()}
                        )

                # 如果至少有一个分割有数据，计算跨 train/test 的全局均值
                if all_count > 0:
                    all_summary = {
                        "SSIM": all_ssim_sum / all_count,
                        "PSNR": all_psnr_sum / all_count,
                        "LPIPS": all_lpips_sum / all_count,
                    }
                    print(f"  [all]   SSIM : {all_summary['SSIM']:>12.7f}")
                    print(f"  [all]   PSNR : {all_summary['PSNR']:>12.7f}")
                    print(f"  [all]   LPIPS: {all_summary['LPIPS']:>12.7f}")
                    print("")

                    full_dict[scene_dir][method]["all"] = all_summary
                    per_view_dict[scene_dir][method]["all"] = all_per_view

            # 将评估结果序列化为 JSON 文件，供后续分析和可视化使用
            with open(scene_dir + "/results1.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view1.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print("Unable to compute metrics for model", scene_dir, "because", e)

if __name__ == "__main__":
    # 设置默认 GPU 设备
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # 命令行参数解析
    # --model_paths / -m: 一个或多个模型输出目录路径（必需参数）
    # nargs="+" 允许接收一个或多个路径参数
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
