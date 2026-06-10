#!/usr/bin/env python3
"""
Camera trajectory rendering: interpolate between training camera positions and generate a video.
"""

import torch
import os
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np
import torchvision
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state
from utils.camera_utils import Camera
from argparse import ArgumentParser


def smooth_poses(poses, sigma):
    """
    对相机位姿序列做高斯平滑：平移用 gaussian_filter1d，旋转转四元数再平滑。
    sigma=0 表示不平滑。
    """
    if sigma <= 0:
        return poses
    mats = np.stack([p.numpy() for p in poses])  # (N,4,4)
    # 平移平滑
    trans = mats[:, :3, 3]                        # (N,3)
    trans_s = gaussian_filter1d(trans, sigma=sigma, axis=0)
    # 旋转平滑（四元数域）
    rots = mats[:, :3, :3]                        # (N,3,3)
    quats = Rotation.from_matrix(rots).as_quat()  # (N,4) xyzw
    # 翻转相邻四元数符号防止插值穿越球面
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i-1]) < 0:
            quats[i] = -quats[i]
    quats_s = gaussian_filter1d(quats, sigma=sigma, axis=0)
    quats_s /= np.linalg.norm(quats_s, axis=1, keepdims=True)
    rots_s = Rotation.from_quat(quats_s).as_matrix()
    # 重组
    out = mats.copy()
    out[:, :3, :3] = rots_s
    out[:, :3, 3]  = trans_s
    return [torch.from_numpy(m).float() for m in out]


def interpolate_poses(poses, n_frames=300):
    """
    Linear interpolation between poses.
    poses: list of 4x4 matrices
    Returns: list of interpolated 4x4 matrices
    """
    interpolated = []

    for i in range(len(poses)):
        start_pose = poses[i]
        end_pose = poses[(i + 1) % len(poses)]

        frames_per_segment = n_frames // len(poses)
        for j in range(frames_per_segment):
            alpha = j / frames_per_segment
            # Linear interpolation of position and rotation
            interp_pose = (1 - alpha) * start_pose + alpha * end_pose
            interpolated.append(interp_pose)

    return interpolated


def render_trajectory(dataset: ModelParams, pipeline: PipelineParams, iteration: int, n_frames: int = 300):
    """
    Render trajectory video from training camera poses.
    """
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Get training camera poses
        train_cameras = scene.getTrainCameras()
        if len(train_cameras) < 2:
            print("Need at least 2 training cameras for trajectory")
            return

        poses = [torch.from_numpy(np.linalg.inv(cam.world_view_transform.cpu().numpy())).float()
                 for cam in train_cameras]

        # 平滑相机位姿后插值
        print(f"Smoothing {len(train_cameras)} poses with sigma={args.smooth_sigma}...")
        poses = smooth_poses(poses, args.smooth_sigma)
        print(f"Interpolating {len(train_cameras)} training cameras into {n_frames} frames...")
        interp_poses = interpolate_poses(poses, n_frames)

        # Render trajectory
        render_path = Path(dataset.model_path) / f"trajectory_{iteration}" / "renders"
        render_path.mkdir(parents=True, exist_ok=True)

        frame_list = []
        print(f"Rendering {len(interp_poses)} frames...")

        for idx, interp_pose in enumerate(tqdm(interp_poses)):
            # Create virtual camera at interpolated pose
            # Use first camera as template (intrinsics)
            template_cam = train_cameras[0]

            # Set world view transform
            world_view_transform = torch.inverse(interp_pose).cuda()

            # Create camera object (simplified)
            class TrajCamera:
                def __init__(self, template, world_view):
                    self.image_width = template.image_width
                    self.image_height = template.image_height
                    self.FoVx = template.FoVx
                    self.FoVy = template.FoVy
                    self.world_view_transform = world_view
                    # [zzx 2026-06-04] 修复轨迹视频"拉丝": 原来 full_proj 固定用模板(第0个)相机的投影,
                    #   但 world_view 每帧都变 -> 投影矩阵与视图矩阵不匹配, 除第0帧外全是放射状拉丝.
                    #   改为每帧用本帧 world_view 重算 full_proj (投影只依赖内参/znear, 用模板的 projection_matrix).
                    # self.full_proj_transform = template.full_proj_transform
                    self.full_proj_transform = (world_view.unsqueeze(0).bmm(template.projection_matrix.unsqueeze(0))).squeeze(0)
                    self.projection_matrix = template.projection_matrix
                    # [zzx 2026-06-04] 原 -world_view[:3,3] 不符合本仓库 world_view 约定(行向量/转置),
                    #   会让视角相关颜色(SH)算错. 与 Camera 类一致用 inverse()[3,:3].
                    # self.camera_center = -world_view[:3, 3]
                    self.camera_center = world_view.inverse()[3, :3]
                    self.image_name = "trajectory"  # For exposure lookup

            traj_cam = TrajCamera(template_cam, world_view_transform)

            # Render (use template camera's exposure as fallback for trajectory)
            rendering = render(traj_cam, gaussians, pipeline, background,
                              use_trained_exp=False, separate_sh=False)["render"]

            # Save frame
            frame_path = render_path / f"{idx:05d}.png"
            torchvision.utils.save_image(rendering, frame_path)
            frame_list.append(rendering.cpu())

        print(f"\n✓ Rendered {len(frame_list)} frames to {render_path}")

        # Create video
        video_path = Path(dataset.model_path) / f"trajectory_{iteration}.mp4"
        print(f"Creating video: {video_path}")

        # 用系统ffmpeg避免conda环境ffmpeg缺libx264；pad确保偶数尺寸
        import shutil as _shutil
        _ffmpeg = "/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else (_shutil.which("ffmpeg") or "ffmpeg")
        cmd = f"{_ffmpeg} -framerate 30 -i {render_path}/%05d.png -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -c:v libx264 -preset fast -pix_fmt yuv420p -y {video_path}"
        os.system(cmd)

        print(f"✓ Video saved: {video_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Trajectory rendering script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--n_frames", default=300, type=int, help="Number of frames in trajectory")
    parser.add_argument("--smooth_sigma", default=3.0, type=float, help="Gaussian smoothing sigma for camera poses (0=off)")

    args = get_combined_args(parser)
    print(f"Rendering trajectory for {args.model_path}")
    print(f"Number of frames: {args.n_frames}, smooth_sigma: {args.smooth_sigma}")

    safe_state(False)
    render_trajectory(model.extract(args), pipeline.extract(args), args.iteration, args.n_frames)
