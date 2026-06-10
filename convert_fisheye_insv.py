import os
import struct
import logging
from argparse import ArgumentParser
import shutil

# [zzx修改] 基于 /data/group2/zzx/convert.py 改写，适配 INSV 鱼眼视频帧 (2944×2880)
# 主要改动：
#   1. [zzx修改] 相机参数 920,920,1472,1440 → 1840,1840,1472,1440  (适配2944×2880分辨率)
#   2. [zzx修改] exhaustive_matcher → sequential_matcher  (视频连续帧，更快更省资源)
#   3. [zzx添加] camera_params 改为命令行可配置
parser = ArgumentParser("Colmap converter (INSV fisheye)")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV_FISHEYE", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
# [zzx添加] 鱼眼相机初始内参，可命令行覆盖 (fx,fy,cx,cy,k1,k2,k3,k4)
parser.add_argument("--camera_params", default="1840,1840,1472,1440,0,0,0,0", type=str)
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0
os.environ["QT_QPA_PLATFORM"] = "offscreen"
if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction
    # [zzx修改] OPENCV_FISHEYE 鱼眼模型，相机参数适配 2944×2880
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model OPENCV_FISHEYE \
        --ImageReader.camera_params " + args.camera_params + " \
        --FeatureExtraction.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    # [zzx修改] exhaustive_matcher → sequential_matcher (视频连续帧，O(n)而非O(n^2)，更快更省显存)
    feat_matching_cmd = colmap_command + " sequential_matcher \
        --database_path " + args.source_path + "/distorted/database.db \
        --SequentialMatching.overlap 10 \
        --FeatureMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001 \
        --Mapper.ba_global_max_num_iterations=50 \
        --Mapper.ba_local_max_num_iterations=20 \
        --Mapper.init_min_num_inliers=100 \
        --Mapper.abs_pose_min_num_inliers=30 \
        --Mapper.filter_max_reproj_error=4 \
        --Mapper.max_reg_trials=3")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

def count_images_in_model(model_path):
    with open(os.path.join(model_path, "images.bin"), "rb") as f:
        num_images = struct.unpack('<Q', f.read(8))[0]
    return num_images

# 挑选包含最多图像的模型文件夹进行后续处理
sparse_dir = args.source_path + "/distorted/sparse"
model_dirs = [d for d in os.listdir(sparse_dir) if os.path.isdir(os.path.join(sparse_dir, d))]
best_model = max(model_dirs, key=lambda d: count_images_in_model(os.path.join(sparse_dir, d)))
best_model_dir = os.path.join(sparse_dir, best_model)
print(f"Best Model Path: {best_model_dir}")

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/input \
    --input_path " + best_model_dir + " \
    --output_path " + args.source_path + "\
    --output_type COLMAP")
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Undistortion failed with code {exit_code}. Exiting.")
    exit(exit_code)

files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

print("Done.")
