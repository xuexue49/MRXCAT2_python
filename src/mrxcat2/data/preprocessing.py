import cProfile
import glob
import os
import pstats

import numpy as np

from mrxcat2.io.utils import load_bin_as_numpy, save_numpy_as_nifti, parse_xcat_log_from_path
from mrxcat2.phantom.tissue import TissueProcessor
from mrxcat2.simulation.coils import calculate_coil_sensitivities
from mrxcat2.simulation.mr_signal import generate_bssfp_signal, apply_low_pass_filter, apply_coil_sensitivities

if __name__ == "__main__":
    # --- 1. 配置图像和路径参数 ---
    # 图像元数据
    profiler = cProfile.Profile()
    profiler.enable()

    numpy_dtype = '<f4'  # 数据类型: 小端, 4字节浮点数

    # 路径参数
    base_path = "/home/chenxinpeng/MRXCAT2_python/raw_data"
    case_name = "Patient1"

    tissue_process = TissueProcessor(property_set='ours', normalize=False)

    # 根据新要求构建输入和输出目录
    input_dir = os.path.join(base_path, case_name, "xcat_bin")
    output_dir = os.path.join(base_path, case_name, "mask")
    log_path = os.path.join(base_path, case_name, "log")

    # 读取生成参数
    dims, voxel_size, rotation_deg = parse_xcat_log_from_path(log_path)

    # 生成静态参数
    coil_maps = calculate_coil_sensitivities(dims, voxel_size, 12, rotation_deg=rotation_deg)

    # --- 2. 自动扫描并处理文件 ---
    # 检查输入目录是否存在
    if not os.path.isdir(input_dir):
        raise f"错误：输入目录不存在 -> {input_dir}"
    else:
        # 使用 glob 查找所有 .bin 文件，返回一个列表
        bin_files = glob.glob(os.path.join(input_dir, '*.bin'))

    if not bin_files:
        raise f"警告：在目录 '{input_dir}' 中没有找到任何 .bin 文件。"
    else:
        print(f"在 '{input_dir}' 中找到 {len(bin_files)} 个 .bin 文件，开始处理...")
        # 排序文件以保证处理顺序一致
        bin_files.sort()

    for input_filename in bin_files:
        # 获取不带扩展名的基本文件名，例如 'act_1'
        base_filename = os.path.basename(input_filename)
        filename_without_ext = os.path.splitext(base_filename)[0]

        # 构建输出文件的完整路径
        output_filename = os.path.join(output_dir, f'{filename_without_ext}.nii.gz')

        print(f"\n--- 正在处理: {base_filename} ---")

        # 加载bin文件为Numpy矩阵
        image_matrix = load_bin_as_numpy(bin_path=input_filename, dims=dims, numpy_dtype=numpy_dtype)

        # 检查矩阵是否成功加载
        if image_matrix is None:
            raise f"未能成功加载矩阵"
        else:
            print(f"成功加载矩阵，Shape: {image_matrix.shape}, Dtype: {image_matrix.dtype}")

        # 图像生成逻辑
        image_matrix = tissue_process.simplify_xcat_labels(image_matrix)
        pd, t1, t2, t2s = tissue_process.assign_tissue_properties(image_matrix)

        image = generate_bssfp_signal(pd, t1, t2)
        image = apply_low_pass_filter(image, filter_strength=1.5)

        image = apply_coil_sensitivities(image, coil_sens_maps=coil_maps)
        image = (np.abs(image) ** 2).sum(axis=-1)
        image = np.sqrt(image)
        image = image.astype(np.int32)

        #for i in range(0, image.shape[3]):
        #    output_filename = os.path.join(output_dir, f'{filename_without_ext}_coil{i}.nii.gz')
        #    save_numpy_as_nifti(numpy_array=image[..., i], output_path=output_filename, voxel_size=voxel_size)
        save_numpy_as_nifti(numpy_array=image, output_path=output_filename, voxel_size=voxel_size)

        # # 保存矩阵
        # pd_filename = os.path.join(output_dir, f'pd/{filename_without_ext}.nii.gz')
        # save_numpy_as_nifti(numpy_array=pd, output_path=pd_filename, voxel_size=voxel_size)
        #
        # t1_filename = os.path.join(output_dir, f't1/{filename_without_ext}.nii.gz')
        # save_numpy_as_nifti(numpy_array=t1, output_path=t1_filename, voxel_size=voxel_size)
        #
        # t2_filename = os.path.join(output_dir, f't2/{filename_without_ext}.nii.gz')
        # save_numpy_as_nifti(numpy_array=t2, output_path=t2_filename, voxel_size=voxel_size)
        #
        # t2s_filename = os.path.join(output_dir, f't2s/{filename_without_ext}.nii.gz')
        # save_numpy_as_nifti(numpy_array=t2s, output_path=t2s_filename, voxel_size=voxel_size)

    profiler.disable()

    # --- 格式化并打印分析报告 ---
    with open("profile_stats.txt", "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats('cumulative')  # 按累计时间排序
        stats.print_stats()  # 打印所有统计信息
