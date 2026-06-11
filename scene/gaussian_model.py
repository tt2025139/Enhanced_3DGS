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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB  # [zzx palette 2026-06-07] SH2RGB 用于调色板初始化
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        # [zzx palette 2026-06-07] 调色板编辑: DC 基础色重参数化为 K 个调色板色的凸组合
        # dc_rgb_i = softmax(W_i) @ sigmoid(Palette);  W: (N,K) 逐点权重, Palette: (K,3) 全局
        self.use_palette = False
        self.palette_size = 0
        self._palette = torch.empty(0)          # (K,3) 原始值, 经 sigmoid 得 RGB
        self._palette_weights = torch.empty(0)  # (N,K) 逐点权重 logits, 经 softmax 归一
        self.palette_optimizer = None
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            # [zzx palette 2026-06-07] 调色板状态
            self.use_palette,
            self.palette_size,
            self._palette,
            self._palette_weights,
        )

    def restore(self, model_args, training_args):
        # [zzx palette 2026-06-07] 兼容新增的调色板字段(放在末尾, 旧 checkpoint 无此字段)
        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        self.spatial_lr_scale,
        self.use_palette,
        self.palette_size,
        self._palette,
        self._palette_weights) = model_args
        self.training_setup(training_args)
        if self.use_palette:
            # 重建调色板优化器并把权重纳入主优化器
            self._setup_palette_optimizers(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    # [zzx palette 2026-06-07] 调色板色 (K,3) RGB, 经 sigmoid 约束到 (0,1)
    @property
    def get_palette(self):
        return torch.sigmoid(self._palette)

    # [zzx palette] 逐点归一权重 (N,K), softmax 保证非负且和为 1 (凸组合)
    @property
    def get_palette_weights(self):
        return torch.softmax(self._palette_weights, dim=1)

    # [zzx palette] 由调色板+权重算出 DC 项的 SH 系数 (N,1,3)
    def _palette_features_dc(self):
        rgb = self.get_palette_weights @ self.get_palette   # (N,3) 凸组合的 RGB
        return RGB2SH(rgb).unsqueeze(1)                      # (N,1,3)

    @property
    def get_features(self):
        # [zzx palette] 启用调色板时 DC 由调色板导出, 其余视角相关 SH 不变
        features_dc = self._palette_features_dc() if self.use_palette else self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._palette_features_dc() if self.use_palette else self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    # ====================== [zzx palette 2026-06-07] 调色板编辑相关 ======================
    @staticmethod
    def _kmeans(x, K, n_iter=30, seed=0):
        """简易 torch k-means (Lloyd). x:(N,3) -> centers:(K,3). 不依赖 sklearn。"""
        N = x.shape[0]
        g = torch.Generator(device=x.device).manual_seed(seed)
        # k-means++ 风格的简化初始化: 首点随机, 其余按距离平方概率采样
        centers = x[torch.randint(0, N, (1,), generator=g, device=x.device)]
        for _ in range(K - 1):
            d2 = torch.cdist(x, centers).min(dim=1).values ** 2
            probs = d2 / d2.sum().clamp_min(1e-12)
            idx = torch.multinomial(probs, 1, generator=g)
            centers = torch.cat([centers, x[idx]], dim=0)
        for _ in range(n_iter):
            assign = torch.cdist(x, centers).argmin(dim=1)   # (N,)
            new_centers = centers.clone()
            for k in range(K):
                m = assign == k
                if m.any():
                    new_centers[k] = x[m].mean(dim=0)
            if torch.allclose(new_centers, centers, atol=1e-5):
                centers = new_centers
                break
            centers = new_centers
        return centers

    def setup_palette(self, K, training_args, init_temp=0.05):
        """从当前 DC 颜色用 k-means 提取 K 个调色板色, 并初始化逐点凸组合权重。
        之后切换为调色板参数化 (use_palette=True), 渲染/densify 自动改用调色板。"""
        with torch.no_grad():
            rgb = SH2RGB(self._features_dc.squeeze(1)).clamp(0.0, 1.0)   # (N,3)
            centers = self._kmeans(rgb, K).clamp(1e-4, 1 - 1e-4)         # (K,3)
            # 权重 logits = -dist / temp, softmax 后近似 one-hot 到最近调色板色
            dist = torch.cdist(rgb, centers)                            # (N,K)
            logits = -dist / max(init_temp, 1e-6)
        self._palette = nn.Parameter(inverse_sigmoid(centers).contiguous().requires_grad_(True))
        self._palette_weights = nn.Parameter(logits.contiguous().requires_grad_(True))
        self.palette_size = K
        self.use_palette = True
        self._setup_palette_optimizers(training_args)
        print(f"[zzx palette] 调色板已初始化: K={K}, 点数={rgb.shape[0]}")

    def _setup_palette_optimizers(self, training_args):
        """把逐点权重加入主优化器(随 densify/prune 一起增删), 全局调色板色单独优化器。
        同时冻结原 f_dc (不再使用)。"""
        palette_lr = getattr(training_args, "palette_lr", 0.005)
        palette_weight_lr = getattr(training_args, "palette_weight_lr", 0.01)
        # 冻结 f_dc 学习率 (DC 改由调色板导出, _features_dc 不再训练)
        existing = {g["name"] for g in self.optimizer.param_groups}
        for g in self.optimizer.param_groups:
            if g["name"] == "f_dc":
                g["lr"] = 0.0
        # 逐点权重作为新参数组加入主优化器 (名字 palette_weights, 参与 densify/prune)
        if "palette_weights" not in existing:
            self.optimizer.add_param_group(
                {"params": [self._palette_weights], "lr": palette_weight_lr, "name": "palette_weights"})
        # 全局调色板色单独优化 (与 exposure 类似, 不参与逐点增删)
        self.palette_optimizer = torch.optim.Adam([self._palette], lr=palette_lr, eps=1e-15)

    def step_palette(self):
        if self.palette_optimizer is not None:
            self.palette_optimizer.step()
            self.palette_optimizer.zero_grad(set_to_none=True)
    # ==================================================================================

    def update_learning_rate(self, iteration):
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if self.use_palette:  # [zzx palette] 逐点调色板权重 logits, 供编辑时重建
            for k in range(self.palette_size):
                l.append('palette_w_{}'.format(k))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        # [zzx palette] 启用调色板时, 把调色板导出的 DC 烘焙进 f_dc, 使 ply 仍可被标准 render.py 渲染
        if self.use_palette:
            f_dc = self._palette_features_dc().detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        else:
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        arrays = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]
        if self.use_palette:  # [zzx palette] 追加逐点权重列 + 旁路写调色板 json
            arrays.append(self._palette_weights.detach().cpu().numpy())
            palette_rgb = self.get_palette.detach().cpu().numpy()
            with open(os.path.join(os.path.dirname(path), "palette.json"), "w") as f:
                json.dump({"palette_size": int(self.palette_size),
                           "palette_rgb": palette_rgb.tolist()}, f, indent=2)
        attributes = np.concatenate(arrays, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        # [zzx palette 2026-06-07] 若存在调色板旁路文件与逐点权重列, 重建调色板参数(供编辑)
        palette_json = os.path.join(os.path.dirname(path), "palette.json")
        pw_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("palette_w_")]
        if os.path.exists(palette_json) and len(pw_names) > 0:
            with open(palette_json, "r") as f:
                pinfo = json.load(f)
            K = int(pinfo["palette_size"])
            palette_rgb = np.asarray(pinfo["palette_rgb"], dtype=np.float32)          # (K,3) RGB
            pw_names = sorted(pw_names, key=lambda x: int(x.split('_')[-1]))
            weights = np.zeros((xyz.shape[0], len(pw_names)), dtype=np.float32)
            for idx, attr_name in enumerate(pw_names):
                weights[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # palette_rgb 是 sigmoid 后的 RGB, 反 sigmoid 还原为原始可优化值
            palette_rgb_t = torch.tensor(palette_rgb, dtype=torch.float, device="cuda").clamp(1e-4, 1 - 1e-4)
            self._palette = nn.Parameter(inverse_sigmoid(palette_rgb_t).requires_grad_(True))
            self._palette_weights = nn.Parameter(torch.tensor(weights, dtype=torch.float, device="cuda").requires_grad_(True))
            self.palette_size = K
            self.use_palette = True
            print(f"[zzx palette] 已从 ply 重建调色板: K={K}")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.use_palette:  # [zzx palette] 逐点权重随剪枝同步
            self._palette_weights = optimizable_tensors["palette_weights"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_palette_weights=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        if self.use_palette:  # [zzx palette] 新增点的逐点权重一并拼接
            d["palette_weights"] = new_palette_weights

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.use_palette:
            self._palette_weights = optimizable_tensors["palette_weights"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)
        # [zzx palette] split 出的新点继承父点权重
        new_palette_weights = self._palette_weights[selected_pts_mask].repeat(N, 1) if self.use_palette else None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii, new_palette_weights)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]
        # [zzx palette] clone 出的新点继承父点权重
        new_palette_weights = self._palette_weights[selected_pts_mask] if self.use_palette else None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_palette_weights)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii, cameras=None, depth_prune_threshold=0.3, depth_prune_min_views=2):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        if cameras is not None and depth_prune_threshold > 0:
            depth_prune_mask = self.depth_consistency_prune(cameras, depth_prune_threshold, depth_prune_min_views)
            prune_mask = torch.logical_or(prune_mask, depth_prune_mask)

        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, pixels=None):
        if pixels is not None:
            self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True) * pixels[update_filter]
            self.denom[update_filter] += pixels[update_filter]
        else:
            self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
            self.denom[update_filter] += 1

    def depth_consistency_prune(self, cameras, threshold, min_views):
        import random
        xyz = self.get_xyz
        N = xyz.shape[0]
        inconsistent_count = torch.zeros(N, device="cuda")

        reliable_cams = [cam for cam in cameras if cam.depth_reliable]
        sample_cams = random.sample(reliable_cams, min(len(reliable_cams), 64))

        for cam in sample_cams:

            world_view = cam.world_view_transform
            full_proj = cam.full_proj_transform

            ones = torch.ones(N, 1, device="cuda")
            homo_xyz = torch.cat([xyz, ones], dim=1)

            proj = homo_xyz @ full_proj.T
            w = proj[:, 3].clamp(min=1e-6)
            screen_x = (proj[:, 0] / w + 1) * 0.5 * cam.image_width
            screen_y = (proj[:, 1] / w + 1) * 0.5 * cam.image_height

            view_xyz = homo_xyz @ world_view.T
            gs_depth = -view_xyz[:, 2]

            valid = (screen_x >= 0) & (screen_x < cam.image_width) & \
                    (screen_y >= 0) & (screen_y < cam.image_height) & \
                    (gs_depth > 0)

            if not valid.any():
                continue

            sx = screen_x[valid].long().clamp(0, cam.image_width - 1)
            sy = screen_y[valid].long().clamp(0, cam.image_height - 1)

            mono_invdepth = cam.invdepthmap.cuda()
            if mono_invdepth.ndim == 3:
                mono_invdepth = mono_invdepth[0]

            sampled_mono_invdepth = mono_invdepth[sy, sx]

            valid_mono = sampled_mono_invdepth > 1e-6
            if not valid_mono.any():
                continue

            gs_invdepth = 1.0 / gs_depth[valid].clamp(min=1e-6)

            depth_ratio = gs_invdepth[valid_mono] / sampled_mono_invdepth[valid_mono]
            inconsistent = (depth_ratio < threshold) | (depth_ratio > (1.0 / threshold))

            idx_valid = torch.where(valid)[0]
            idx_inconsistent = idx_valid[valid_mono][inconsistent]
            inconsistent_count[idx_inconsistent] += 1

        prune_mask = inconsistent_count >= min_views
        return prune_mask

    def spatial_regularization(self, cameras, percent=0.05):
        import random
        xyz = self.get_xyz
        N = xyz.shape[0]

        sample_cams = random.sample(cameras, min(len(cameras), 32))
        cam_centers = torch.stack([c.camera_center for c in sample_cams])

        min_dist_sq = torch.full((N,), float('inf'), device="cuda")
        chunk_size = 8
        for i in range(0, len(sample_cams), chunk_size):
            chunk_centers = cam_centers[i:i+chunk_size]
            diff = xyz.unsqueeze(1) - chunk_centers.unsqueeze(0)
            dist_sq = (diff ** 2).sum(dim=2)
            chunk_min = dist_sq.min(dim=1).values
            min_dist_sq = torch.minimum(min_dist_sq, chunk_min)

        dist_threshold = torch.quantile(min_dist_sq.sqrt(), percent)
        far_mask = min_dist_sq.sqrt() > dist_threshold

        opacity = self.get_opacity.squeeze()
        reg_loss = far_mask.float() * opacity * min_dist_sq.sqrt() / dist_threshold.clamp(min=1e-6)

        return reg_loss.mean()
