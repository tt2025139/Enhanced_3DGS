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

import torch
import math
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

HAS_DIFF_GAUSSIAN_RASTERIZATION = False
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    HAS_DIFF_GAUSSIAN_RASTERIZATION = True
except ImportError:
    print("Module 'diff_gaussian_rasterization' not Found")

HAS_GSPLAT = False
try:
    from gsplat import rasterization
    HAS_GSPLAT = True
except ImportError:
    pass


def render_gsplat(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False, absgrad=False, depth_threshold=None):
    if not HAS_GSPLAT:
        raise ImportError("gsplat is not installed. Please install it with: pip install gsplat")

    means = pc.get_xyz
    quats = pc.get_rotation
    scales = pc.get_scaling * scaling_modifier
    opacities = pc.get_opacity.squeeze(-1)

    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            colors = colors_precomp
            sh_degree = None
        else:
            colors = pc.get_features
            sh_degree = pc.active_sh_degree
    else:
        colors = override_color
        sh_degree = None

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1)

    fx = viewpoint_camera.image_width / (2 * math.tan(viewpoint_camera.FoVx / 2))
    fy = viewpoint_camera.image_height / (2 * math.tan(viewpoint_camera.FoVy / 2))
    cx = viewpoint_camera.image_width / 2
    cy = viewpoint_camera.image_height / 2

    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], device="cuda", dtype=means.dtype)

    viewmats = viewmat.unsqueeze(0)
    Ks = K.unsqueeze(0)

    backgrounds = None

    rendered_image, render_alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        near_plane=0.01,
        far_plane=100.0,
        sh_degree=sh_degree,
        backgrounds=backgrounds,
        render_mode="RGB+D",
        packed=True,
        absgrad=absgrad,
        rasterize_mode="antialiased" if pipe.antialiasing else "classic"
    )

    rendered_image = rendered_image[0]
    render_alphas = render_alphas[0]

    if absgrad and "means2d" in meta and meta["means2d"].requires_grad:
        meta["means2d"].retain_grad()

    num_gaussians = means.shape[0]
    radii = torch.zeros(num_gaussians, device="cuda", dtype=torch.float32)

    visible_ids = meta['gaussian_ids']
    visible_radii = meta['radii']

    if visible_ids.shape[0] > 0:
        max_radii = visible_radii.max(dim=-1)[0].float()
        radii[visible_ids] = max_radii

    rendered_image = rendered_image.permute(2, 0, 1)
    depth_image = rendered_image[3:4, :, :]
    rendered_image = rendered_image[:3, :, :]

    if bg_color is not None:
        alpha = render_alphas.permute(2, 0, 1)
        rendered_image = rendered_image + (1.0 - alpha) * bg_color.view(3, 1, 1)

    points_view = viewpoint_camera.world_view_transform[:3, :3] @ means.T + viewpoint_camera.world_view_transform[:3, 3:4]
    points_proj = viewpoint_camera.projection_matrix @ torch.cat([points_view, torch.ones(1, points_view.shape[1], device=means.device)], dim=0)
    points_proj_normalized = points_proj[:2] / points_proj[3:4]
    screenspace_points = torch.cat([
        points_proj_normalized[0].unsqueeze(1),
        points_proj_normalized[1].unsqueeze(1),
        points_view[2:3].T
    ], dim=1)
    try:
        screenspace_points.retain_grad()
    except:
        pass

    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3, None, None]

    rendered_image = rendered_image.clamp(0, 1)

    meta["width"] = int(viewpoint_camera.image_width)
    meta["height"] = int(viewpoint_camera.image_height)
    meta["n_cameras"] = 1
    meta["_renderer_backend"] = "gsplat"

    if depth_threshold is not None:
        meta["_depth_threshold"] = depth_threshold

    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
        "invdepth": depth_image,
        "pixels": None,
        "meta": meta
    }

    return out


def render_diff_gaussian_rasterization(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False, depth_threshold = None, absgrad=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    if not HAS_DIFF_GAUSSIAN_RASTERIZATION:
        raise ImportError("diff_gaussian_rasterization is not installed.")

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        depth_threshold=depth_threshold,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing,
        absgrad=absgrad
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    if separate_sh:
        rendered_image, invdepth, radii, pixels = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, invdepth, radii, pixels = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    rendered_image = rendered_image.clamp(0, 1)

    meta = {}
    meta["width"] = int(viewpoint_camera.image_width)
    meta["height"] = int(viewpoint_camera.image_height)
    meta["n_cameras"] = 1
    meta["_renderer_backend"] = "diff_gaussian_rasterization"

    if absgrad and screenspace_points.requires_grad:
        absgrad_holder = {"value": None}
        def absgrad_hook(grad):
            if hasattr(grad, 'absgrad') and grad.absgrad is not None:
                absgrad_holder["value"] = grad.absgrad.clone()
            return grad
        screenspace_points.register_hook(absgrad_hook)
        meta["_absgrad_holder"] = absgrad_holder

    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
        "invdepth": invdepth,
        "pixels": pixels,
        "meta": meta
    }

    return out


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False, depth_threshold = None, abs_gs=False, use_gsplat_renderer=False):
    if abs_gs:
        if HAS_DIFF_GAUSSIAN_RASTERIZATION:
            return render_diff_gaussian_rasterization(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, separate_sh, override_color, use_trained_exp, depth_threshold, absgrad=True)
        elif HAS_GSPLAT:
            return render_gsplat(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, separate_sh, override_color, use_trained_exp, absgrad=True, depth_threshold=depth_threshold)
        else:
            raise ImportError("AbsGS requires gsplat or diff_gaussian_rasterization. Please install one of them.")
    else:
        if use_gsplat_renderer and HAS_GSPLAT:
            return render_gsplat(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, separate_sh, override_color, use_trained_exp, absgrad=False, depth_threshold=depth_threshold)
        elif HAS_DIFF_GAUSSIAN_RASTERIZATION:
            return render_diff_gaussian_rasterization(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, separate_sh, override_color, use_trained_exp, depth_threshold)
        elif HAS_GSPLAT:
            return render_gsplat(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, separate_sh, override_color, use_trained_exp, absgrad=False, depth_threshold=depth_threshold)
        else:
            raise ImportError("Neither gsplat nor diff_gaussian_rasterization is available")
