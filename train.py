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

import os
import sys

_conda_env = os.path.join(sys.prefix, 'lib')
_conda_env_targets = os.path.join(sys.prefix, 'targets', 'x86_64-linux', 'lib')
_current_lib = os.environ.get('LIBRARY_PATH', '')
_current_ld = os.environ.get('LD_LIBRARY_PATH', '')
if _conda_env not in _current_lib:
    os.environ['LIBRARY_PATH'] = f'{_conda_env}:{_conda_env_targets}:{_current_lib}'
if _conda_env not in _current_ld:
    os.environ['LD_LIBRARY_PATH'] = f'{_conda_env}:{_conda_env_targets}:{_current_ld}'
_tmpdir = os.environ.get('TMPDIR', '/tmp')
if _tmpdir == '/tmp':
    _alt_tmp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tmp')
    if not os.path.exists(_alt_tmp):
        try:
            os.makedirs(_alt_tmp, exist_ok=True)
        except OSError:
            _alt_tmp = '/tmp'
    _df = os.statvfs(_tmpdir)
    if _df.f_bavail * _df.f_bsize < 1024 * 1024 * 1024:
        os.environ['TMPDIR'] = _alt_tmp
        os.environ['TORCH_EXTENSIONS_DIR'] = os.path.join(_alt_tmp, 'torch_extensions')

import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

def get_resolution_scale(iteration, opt):
    if not opt.progressive_resolution:
        return 1.0
    if iteration < opt.resolution_schedule_1:
        return opt.resolution_scale_1
    elif iteration < opt.resolution_schedule_2:
        return opt.resolution_scale_2
    else:
        return opt.resolution_scale_3


def get_densify_grad_threshold(opt, current_scale, orig_resolution):
    if not opt.densify_grad_threshold_scale:
        return opt.densify_grad_threshold
    base_resolution = 800 * 600
    current_resolution_approx = orig_resolution / (current_scale ** 2)
    resolution_scale = current_resolution_approx / base_resolution
    return opt.densify_grad_threshold * (resolution_scale ** 0.5)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    use_progressive = opt.progressive_resolution and opt.abs_gs
    resolution_scales_to_load = [1.0]
    if use_progressive:
        resolution_scales_to_load = list(set([
            opt.resolution_scale_1,
            opt.resolution_scale_2,
            opt.resolution_scale_3,
            1.0
        ]))
    scene = Scene(dataset, gaussians, resolution_scales=resolution_scales_to_load)
    gaussians.training_setup(opt)

    current_resolution_scale = 1.0
    if use_progressive:
        train_cams = scene.getTrainCameras(opt.resolution_scale_1)
        orig_resolution = train_cams[0].image_width * train_cams[0].image_height * (opt.resolution_scale_1 ** 2)
        print(f"Original image resolution approx: {orig_resolution} (used for densify threshold scaling)")
        current_resolution_scale = get_resolution_scale(first_iter, opt)
    else:
        orig_resolution = 0

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # [zzx palette 2026-06-07] palette_init_iter<=0: 训练开始即从 SfM 初始颜色提取调色板
    if opt.use_palette and opt.palette_init_iter <= 0 and not gaussians.use_palette:
        gaussians.setup_palette(opt.palette_size, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    use_abs_gs = opt.abs_gs
    if use_abs_gs:
        absgrad_accum = torch.zeros(gaussians.get_xyz.shape[0], device="cuda")
        absgrad_denom = torch.zeros(gaussians.get_xyz.shape[0], device="cuda")

    viewpoint_stack = None
    if use_progressive:
        viewpoint_stack = scene.getTrainCameras(current_resolution_scale).copy()
    else:
        viewpoint_stack = scene.getTrainCameras().copy()

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, use_trained_exp=dataset.train_test_exp, abs_gs=use_abs_gs)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        if use_progressive:
            new_resolution_scale = get_resolution_scale(iteration, opt)
            if new_resolution_scale != current_resolution_scale:
                print(f"\n[ITER {iteration}] Switching resolution scale: {current_resolution_scale} -> {new_resolution_scale}")
                current_resolution_scale = new_resolution_scale
                viewpoint_stack = scene.getTrainCameras(current_resolution_scale).copy()

                if current_resolution_scale == opt.resolution_scale_2:
                    lr_scale = 0.8
                elif current_resolution_scale == opt.resolution_scale_3:
                    lr_scale = opt.lr_scale_factor
                else:
                    lr_scale = 1.0
                if lr_scale < 1.0:
                    for param_group in gaussians.optimizer.param_groups:
                        if param_group["name"] == "xyz":
                            param_group['lr'] = param_group['lr'] * lr_scale
                        elif param_group["name"] == "opacity":
                            param_group['lr'] = param_group['lr'] * max(lr_scale, 0.8)
                        elif param_group["name"] == "scaling":
                            param_group['lr'] = param_group['lr'] * max(lr_scale, 0.8)

        gaussians.update_learning_rate(iteration)

        # [zzx palette 2026-06-07] 延迟初始化: 先正常训练若干步, 再从学到的 DC 颜色提取调色板
        if opt.use_palette and not gaussians.use_palette and opt.palette_init_iter > 0 and iteration == opt.palette_init_iter:
            gaussians.setup_palette(opt.palette_size, opt)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            if use_progressive:
                viewpoint_stack = scene.getTrainCameras(current_resolution_scale).copy()
            else:
                viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        depth_threshold = opt.depth_threshold * scene.cameras_extent if opt.pixel_gs else None
        if use_abs_gs:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, depth_threshold=depth_threshold, abs_gs=True)
        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, depth_threshold=depth_threshold)
        pixels = render_pkg.get("pixels", None)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        meta = render_pkg.get("meta", {})

        if use_abs_gs:
            viewspace_point_tensor.retain_grad()

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        if dataset.masks:
            person_mask = viewpoint_cam.person_mask.cuda()
            Ll1 = l1_loss(image * person_mask, gt_image * person_mask)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image * person_mask, gt_image * person_mask))
        else:
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["invdepth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            if dataset.masks:
                depth_mask *= person_mask
            Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        if opt.spatial_reg and iteration < opt.densify_until_iter:
            Lspatial = opt.spatial_reg_weight * gaussians.spatial_regularization(scene.getTrainCameras(), opt.spatial_reg_percent)
            loss += Lspatial
        else:
            Lspatial = 0

        # [zzx palette 2026-06-07] 可选: 逐点权重熵正则, 鼓励每个高斯偏向单一调色板色 -> 重上色更干净
        if opt.use_palette and gaussians.use_palette and opt.palette_entropy_weight > 0:
            w = gaussians.get_palette_weights.clamp_min(1e-8)
            entropy = -(w * w.log()).sum(dim=1).mean()
            loss += opt.palette_entropy_weight * entropy

        loss.backward()

        if opt.grad_clip_norm > 0:
            params = [group['params'][0] for group in gaussians.optimizer.param_groups]
            torch.nn.utils.clip_grad_norm_(params, max_norm=opt.grad_clip_norm)

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            if iteration % 10 == 0:
                res_info = f" Res:{current_resolution_scale}" if use_progressive else ""
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth": f"{ema_Ll1depth_for_log:.{5}f}{res_info}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, use_abs_gs), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

                if use_abs_gs:
                    abs_grads = None
                    if meta and "means2d" in meta and hasattr(meta["means2d"], "absgrad") and meta["means2d"].absgrad is not None:
                        abs_grads = meta["means2d"].absgrad.clone()
                    elif meta and "_absgrad_holder" in meta and meta["_absgrad_holder"]["value"] is not None:
                        abs_grads = meta["_absgrad_holder"]["value"]
                    elif viewspace_point_tensor.grad is not None and hasattr(viewspace_point_tensor.grad, 'absgrad') and viewspace_point_tensor.grad.absgrad is not None:
                        abs_grads = viewspace_point_tensor.grad.absgrad.clone()

                    if abs_grads is not None:
                        n_gaussians = gaussians.get_xyz.shape[0]
                        if absgrad_accum.shape[0] != n_gaussians:
                            absgrad_accum = torch.zeros(n_gaussians, device="cuda")
                            absgrad_denom = torch.zeros(n_gaussians, device="cuda")

                        renderer_backend = meta.get("_renderer_backend", "diff_gaussian_rasterization") if meta else "diff_gaussian_rasterization"

                        if renderer_backend == "gsplat":
                            abs_grads_xy = abs_grads[..., :2].clone()
                            img_w = meta.get("width", viewpoint_cam.image_width) if meta else viewpoint_cam.image_width
                            img_h = meta.get("height", viewpoint_cam.image_height) if meta else viewpoint_cam.image_height
                            n_cams = meta.get("n_cameras", 1) if meta else 1
                            abs_grads_xy[..., 0] *= img_w / 2.0 * n_cams
                            abs_grads_xy[..., 1] *= img_h / 2.0 * n_cams

                            if meta and "_depth_threshold" in meta:
                                depth_threshold_val = meta["_depth_threshold"]
                                gaussian_ids_depth = meta.get("gaussian_ids", None) if meta else None
                                if gaussian_ids_depth is not None and gaussian_ids_depth.shape[0] > 0:
                                    visible_xyz = gaussians.get_xyz[gaussian_ids_depth]
                                else:
                                    visible_xyz = gaussians.get_xyz[visibility_filter]
                                points_view = viewpoint_cam.world_view_transform[:3, :3] @ visible_xyz.T + viewpoint_cam.world_view_transform[:3, 3:4]
                                depths = points_view[2:3].T
                                scaling_factor = torch.minimum(torch.ones_like(depths), (depths / depth_threshold_val) ** 2)
                                abs_grads_xy = abs_grads_xy * scaling_factor

                            grad_norms = abs_grads_xy.norm(dim=-1)
                            gaussian_ids = meta.get("gaussian_ids", None) if meta else None
                            if gaussian_ids is not None and gaussian_ids.shape[0] > 0:
                                absgrad_accum.index_add_(0, gaussian_ids, grad_norms)
                                absgrad_denom.index_add_(0, gaussian_ids, torch.ones_like(grad_norms))
                            else:
                                absgrad_accum[visibility_filter] += grad_norms[visibility_filter]
                                absgrad_denom[visibility_filter] += 1
                        else:
                            abs_grads_xy = abs_grads[..., :2].clone()

                            if meta and "_depth_threshold" in meta:
                                depth_threshold_val = meta["_depth_threshold"]
                                gaussian_ids_depth = meta.get("gaussian_ids", None) if meta else None
                                if gaussian_ids_depth is not None and gaussian_ids_depth.shape[0] > 0:
                                    visible_xyz = gaussians.get_xyz[gaussian_ids_depth]
                                else:
                                    visible_xyz = gaussians.get_xyz[visibility_filter]
                                points_view = viewpoint_cam.world_view_transform[:3, :3] @ visible_xyz.T + viewpoint_cam.world_view_transform[:3, 3:4]
                                depths = points_view[2:3].T
                                scaling_factor = torch.minimum(torch.ones_like(depths), (depths / depth_threshold_val) ** 2)
                                abs_grads_xy = abs_grads_xy * scaling_factor

                            grad_norms = abs_grads_xy.norm(dim=-1)
                            gaussian_ids = meta.get("gaussian_ids", None) if meta else None
                            if gaussian_ids is not None and gaussian_ids.shape[0] > 0:
                                absgrad_accum.index_add_(0, gaussian_ids, grad_norms)
                                absgrad_denom.index_add_(0, gaussian_ids, torch.ones_like(grad_norms))
                            else:
                                absgrad_accum[visibility_filter] += grad_norms[visibility_filter]
                                absgrad_denom[visibility_filter] += 1
                else:
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, pixels)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    depth_prune_cameras = scene.getTrainCameras() if opt.depth_prune else None
                    n_before = gaussians.get_xyz.shape[0]

                    if use_abs_gs:
                        if use_progressive and opt.densify_grad_threshold_scale:
                            current_threshold = get_densify_grad_threshold(opt, current_resolution_scale, orig_resolution)
                        else:
                            current_threshold = opt.densify_grad_threshold

                        target_point_count = opt.cap_max if opt.cap_max > 0 else 2000000
                        if n_before > target_point_count:
                            scale_factor = n_before / target_point_count
                            current_threshold = current_threshold * scale_factor

                        grads = absgrad_accum / absgrad_denom.clamp_min(1)
                        grads[grads.isnan()] = 0.0
                        grads = grads.unsqueeze(-1)

                        gaussians.tmp_radii = radii
                        gaussians.densify_and_clone(grads, current_threshold, scene.cameras_extent)
                        gaussians.densify_and_split(grads, current_threshold, scene.cameras_extent)

                        prune_mask = (gaussians.get_opacity < opt.opacity_prune_threshold).squeeze()
                        n_low_op = prune_mask.sum().item()
                        if size_threshold:
                            big_points_vs = gaussians.max_radii2D > size_threshold
                            big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * scene.cameras_extent
                            n_bvs = big_points_vs.sum().item()
                            n_bws = big_points_ws.sum().item()
                            n_both = (big_points_vs & big_points_ws).sum().item()
                            print(f"  [Iter {iteration}] Prune: low_op={n_low_op}, big_vs={n_bvs}, big_ws={n_bws}, both={n_both}, total_points={gaussians.get_xyz.shape[0]}")
                            print(f"  [Iter {iteration}] max_radii2D stats: min={gaussians.max_radii2D.min():.1f} max={gaussians.max_radii2D.max():.1f} mean={gaussians.max_radii2D.mean():.1f}")
                            print(f"  [Iter {iteration}] scaling stats: min={gaussians.get_scaling.max(dim=1).values.min():.6f} max={gaussians.get_scaling.max(dim=1).values.max():.6f}")
                            print(f"  [Iter {iteration}] cameras_extent={scene.cameras_extent:.4f}, 0.1*extent={0.1*scene.cameras_extent:.4f}")
                            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

                        if depth_prune_cameras is not None and opt.depth_prune_threshold > 0:
                            depth_prune_mask = gaussians.depth_consistency_prune(depth_prune_cameras, opt.depth_prune_threshold, opt.depth_prune_min_views)
                            prune_mask = torch.logical_or(prune_mask, depth_prune_mask)

                        n_before_prune = gaussians.get_xyz.shape[0]
                        gaussians.prune_points(prune_mask)
                        gaussians.tmp_radii = None
                        n_after_prune = gaussians.get_xyz.shape[0]
                        if iteration % 500 == 0:
                            print(f"  [Iter {iteration}] Points: {n_before}->{n_before_prune}->{n_after_prune}, Pruned: {n_before_prune - n_after_prune}")

                        n_gaussians = gaussians.get_xyz.shape[0]
                        absgrad_accum = torch.zeros(n_gaussians, device="cuda")
                        absgrad_denom = torch.zeros(n_gaussians, device="cuda")

                        torch.cuda.empty_cache()
                    else:
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_prune_threshold, scene.cameras_extent, size_threshold, radii, cameras=depth_prune_cameras, depth_prune_threshold=opt.depth_prune_threshold, depth_prune_min_views=opt.depth_prune_min_views)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    gaussians.max_radii2D = torch.zeros(gaussians.max_radii2D.shape, device="cuda")

            # Optimizer step
            if iteration < opt.iterations:
                if dataset.train_test_exp:
                    gaussians.exposure_optimizer.step()
                    gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if gaussians.use_palette:  # [zzx palette] 步进全局调色板色优化器
                    gaussians.step_palette()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    if len(renderArgs) >= 3:
                        pipe, bg, abs_gs = renderArgs[0], renderArgs[1], renderArgs[2]
                        image = torch.clamp(renderFunc(viewpoint, scene.gaussians, pipe, bg, use_trained_exp=train_test_exp, abs_gs=abs_gs)["render"], 0.0, 1.0)
                    else:
                        pipe, bg = renderArgs[0], renderArgs[1]
                        image = torch.clamp(renderFunc(viewpoint, scene.gaussians, pipe, bg, use_trained_exp=train_test_exp)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    eval_mask = None
                    if hasattr(viewpoint, 'person_mask') and viewpoint.person_mask is not None:
                        eval_mask = viewpoint.person_mask.to("cuda")
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                        if eval_mask is not None:
                            eval_mask = eval_mask[..., eval_mask.shape[-1] // 2:]
                    if eval_mask is not None:
                        image = image * eval_mask
                        gt_image = gt_image * eval_mask
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")
