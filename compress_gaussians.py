# [zzx] 3DGS 模型压缩(纯后处理,不需重训)
# 两种正交手段,可单独或组合使用:
#   1) --prune_opacity T  : 删除 sigmoid(opacity) < T 的高斯(减少点数)
#   2) --sh_degree D      : 截断球谐到 D 阶,丢弃 f_rest 高阶列(每点存储量减少)
#      D=3(45列,原始) D=2(24列) D=1(9列) D=0(0列,只保留DC分量/纯色)
#
# 输出一个完整的、可直接被现有 render.py 加载渲染的模型目录
# (复制 cameras.json/exposure.json,并把 cfg_args 的 sh_degree 改成截断后的阶数,
#  使 GaussianModel.load_ply 里的列数断言自动吻合)
#
# 用法:
#   python compress_gaussians.py -m <源模型目录> --iter 30000 -o <输出目录> \
#       [--prune_opacity 0.01] [--sh_degree 1]
import os, json, shutil, math
from argparse import ArgumentParser, Namespace
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement

parser = ArgumentParser()
parser.add_argument("-m", "--model", required=True)
parser.add_argument("--iter", default=30000, type=int)
parser.add_argument("-o", "--output", required=True)
parser.add_argument("--prune_opacity", default=0.0, type=float, help="删除 sigmoid(opacity) < T 的高斯, 0=不剪枝")
parser.add_argument("--sh_degree", default=3, type=int, choices=[0, 1, 2, 3], help="截断球谐到几阶, 3=不截断")
args = parser.parse_args()

src_ply = Path(args.model) / "point_cloud" / f"iteration_{args.iter}" / "point_cloud.ply"
plydata = PlyData.read(str(src_ply))
v = plydata["vertex"]
names = [p.name for p in v.properties]
data = {n: np.asarray(v[n]) for n in names}
n_in = len(data["x"])

# --- 1) 不透明度剪枝 ---
if args.prune_opacity > 0:
    logit_t = math.log(args.prune_opacity / (1 - args.prune_opacity))  # opacity 存的是 inverse_sigmoid 后的 logit
    keep = data["opacity"] >= logit_t
    for n in names:
        data[n] = data[n][keep]
n_kept = len(data["x"])

# --- 2) 球谐截断: 只保留到 sh_degree 阶, 丢弃更高阶的 f_rest 列 ---
n_rest_keep = 3 * (args.sh_degree + 1) ** 2 - 3  # 0/9/24/45
rest_names = sorted([n for n in names if n.startswith("f_rest_")], key=lambda x: int(x.split("_")[-1]))
drop_rest = set(rest_names[n_rest_keep:])

out_names = [n for n in names if n not in drop_rest]
dtype_full = [(n, "f4") for n in out_names]
elements = np.empty(n_kept, dtype=dtype_full)
for n in out_names:
    elements[n] = data[n]
el = PlyElement.describe(elements, "vertex")

# --- 写输出模型目录(与 render.py 期望的结构一致) ---
out_dir = Path(args.output)
pc_dir = out_dir / "point_cloud" / f"iteration_{args.iter}"
pc_dir.mkdir(parents=True, exist_ok=True)
out_ply = pc_dir / "point_cloud.ply"
PlyData([el]).write(str(out_ply))

for fn in ["cameras.json", "exposure.json"]:
    src = Path(args.model) / fn
    if src.exists():
        shutil.copy(src, out_dir / fn)

cfg_src = (Path(args.model) / "cfg_args").read_text()
cfg_ns = eval(cfg_src)
cfg_ns.sh_degree = args.sh_degree
cfg_ns.model_path = str(out_dir)
(out_dir / "cfg_args").write_text(repr(cfg_ns))

orig_size = src_ply.stat().st_size
comp_size = out_ply.stat().st_size
report = {
    "n_points_in": n_in,
    "n_points_out": n_kept,
    "prune_opacity": args.prune_opacity,
    "sh_degree": args.sh_degree,
    "orig_ply_bytes": orig_size,
    "compressed_ply_bytes": comp_size,
    "compression_ratio": round(orig_size / comp_size, 3),
}
print(json.dumps(report, indent=2))
(out_dir / "compress_report.json").write_text(json.dumps(report, indent=2))
print("saved:", out_ply)
