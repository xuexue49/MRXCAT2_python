import cProfile
import glob
import os
import pstats
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from mrxcat2.io.utils import load_bin_as_numpy, save_numpy_as_nifti, parse_xcat_log_from_path
from mrxcat2.phantom.tissue import TissueProcessor
from mrxcat2.simulation.coils import calculate_coil_sensitivities
from mrxcat2.simulation.mr_signal import generate_bssfp_signal, apply_low_pass_filter, apply_coil_sensitivities

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    start_time = time.time()

    # --- 1. 配置全局参数 ---
    numpy_dtype = '<f4'
    base_path = "/home/chenxinpeng/MRXCAT2_python/raw_data"
    case_name = "Patient1"

    input_dir = os.path.join(base_path, case_name, "xcat_bin")
    image_dir = os.path.join(base_path, case_name, "image")
    mask_dir = os.path.join(base_path, case_name, "mask")
    log_path = os.path.join(base_path, case_name, "log")

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"错误：输入目录不存在 -> {input_dir}")

    os.makedirs(image_dir, exist_ok=True)

    # --- 2. 设置GPU设备并读取元数据 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dims, voxel_size, rotation_deg = parse_xcat_log_from_path(log_path)
    if dims is None:
        raise ValueError("未能从日志文件成功解析出图像维度和体素大小。")

    # --- 3. 预先计算或加载静态线圈图谱 (缓存策略) ---
    coil_maps_path = "/tmp/coil_maps_cache.pt"
    if os.path.exists(coil_maps_path):
        print("从缓存加载线圈灵敏度图谱...")
        coil_maps = torch.load(coil_maps_path).to(device)
    else:
        print("计算新的线圈灵敏度图谱并缓存...")
        coil_maps = calculate_coil_sensitivities(dims, voxel_size, 12, rotation_deg=rotation_deg)
        torch.save(coil_maps, coil_maps_path)
        print(f"线圈图谱已保存至: {coil_maps_path}")
        coil_maps = coil_maps.to(device)

    tissue_process = TissueProcessor(property_set='ours', normalize=False)

    # --- 4. 扫描文件并开始处理循环 ---
    bin_files = sorted(glob.glob(os.path.join(input_dir, '*.bin')))
    if not bin_files:
        print(f"警告：在目录 '{input_dir}' 中没有找到任何 .bin 文件。")
    else:
        print(f"在 '{input_dir}' 中找到 {len(bin_files)} 个 .bin 文件，开始异步处理...")

        # 创建一个线程池专门用于I/O操作，worker数量可以根据磁盘性能调整
        with ThreadPoolExecutor(max_workers=4) as executor:
            for input_filename in bin_files:
                base_filename = os.path.basename(input_filename)
                print(f"\n--- 开始计算: {base_filename} ---")

                # --- CPU加载 ---
                image_matrix = load_bin_as_numpy(bin_path=input_filename, dims=dims, numpy_dtype=numpy_dtype)

                if image_matrix is None:
                    print(f"警告：未能为 {base_filename} 加载矩阵，跳过此文件。")
                    continue

                output_filename = os.path.join(mask_dir, f'{os.path.splitext(base_filename)[0]}.nii.gz')
                executor.submit(save_numpy_as_nifti, image_matrix, output_filename, voxel_size)

                # --- GPU 计算流水线 ---
                image_tensor = torch.from_numpy(image_matrix).int().to(device)
                simplified_tensor = tissue_process.simplify_xcat_labels(image_tensor)
                pd, t1, t2, _ = tissue_process.assign_tissue_properties(simplified_tensor)

                image = generate_bssfp_signal(pd, t1, t2)
                image = apply_low_pass_filter(image, filter_strength=1.5)
                image = apply_coil_sensitivities(image, coil_sens_maps=coil_maps)

                # GPU后处理并传回CPU
                # noinspection PyArgumentList
                final_image_numpy = torch.sqrt(torch.sum(torch.abs(image) ** 2, axis=-1)).cpu().numpy().astype(np.int8)

                print(f"--- 计算完成: {base_filename}，提交至后台保存... ---")

                # --- 异步提交保存任务 ---
                # 主循环不会在此等待，会立刻开始处理下一个文件
                output_filename = os.path.join(image_dir, f'{os.path.splitext(base_filename)[0]}.nii.gz')
                executor.submit(save_numpy_as_nifti, final_image_numpy, output_filename, voxel_size)

            # with语句结束时，会自动等待所有提交的任务完成
            print("\n所有计算任务已提交，正在等待后台I/O操作完成...")

    profiler.disable()

    end_time = time.time()

    # --- 格式化并打印分析报告 ---
    with open("profile_stats_async.txt", "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats('cumulative')
        stats.print_stats()

    print(f"\n所有文件处理完成！总耗时: {end_time - start_time:.2f} 秒。")
