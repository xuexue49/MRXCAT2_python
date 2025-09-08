import os
import re

import nibabel as nib
import numpy as np


def load_bin_as_numpy(bin_path, dims, numpy_dtype='<f4'):
    """
    从无头的原始二进制文件加载数据并返回一个Numpy矩阵。

    Args:
        bin_path (str): 输入的二进制文件路径。
        dims (tuple): 图像维度 (x, y, z)。
        numpy_dtype (str or np.dtype): 数据的Numpy类型 (例如 '<f4')。

    Returns:
        np.ndarray: 一个3D Numpy矩阵 (shape: z, y, x)，如果成功。
        None: 如果文件未找到或数据大小不匹配。
    """
    try:
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"错误: 输入文件未找到 '{bin_path}'")

        # --- 步骤 1: 读取原始二进制数据 ---
        # 使用 np.fromfile 读取文件。必须精确指定数据类型（dtype），
        # 包含了字节序和数据类型。
        raw_data_1d = np.fromfile(bin_path, dtype=numpy_dtype)

        # --- 步骤 2: 验证数据大小 ---
        expected_voxels = np.prod(dims)
        actual_voxels = len(raw_data_1d)
        if expected_voxels != actual_voxels:
            raise ValueError(
                f"数据大小不匹配！根据维度 {dims}，"
                f"应有 {expected_voxels} 个体素，但文件中实际读取到 {actual_voxels} 个。"
            )

        # --- 步骤 3: 将一维数组重塑为三维图像矩阵 (Z, Y, X) ---
        # NumPy 默认使用 C 语言风格的行主序 (row-major order)，
        # 这意味着最后一个维度索引变化最快。对于 (z, y, x) 的 shape，
        # x 会先变化，然后是 y，最后是 z。这通常符合医学图像的存储方式。
        image_shape = (dims[2], dims[1], dims[0])
        image_data_3d = raw_data_1d.reshape(image_shape)

        # 最后返回[x,y,z]的图像
        return image_data_3d.transpose(2, 1, 0)

    except (FileNotFoundError, ValueError) as e:
        print(f"处理文件 '{bin_path}' 时发生错误: {e}")
        return None


def save_numpy_as_nifti(numpy_array, output_path, voxel_size):
    """
    将一个Numpy矩阵保存为 .nii.gz 文件。

    Args:
        numpy_array (np.ndarray): 要保存的3D Numpy矩阵。
        output_path (str): 输出的 .nii.gz 文件路径。
        voxel_size (tuple): 体素大小 (vx, vy, vz)，单位mm。
    """
    try:
        # --- 创建仿射变换矩阵 (Affine Matrix) ---
        # 仿射矩阵定义了图像体素坐标到真实世界物理坐标的映射关系。
        vx, vy, vz = voxel_size
        affine_matrix = np.array([
            [vx, 0, 0, 0],
            [0, vy, 0, 0],
            [0, 0, vz, 0],
            [0, 0, 0, 1]
        ])

        # --- 创建并保存 NIfTI 图像对象 ---
        # 使用 nibabel 的 Nifti1Image 类，传入图像数据和仿射矩阵。
        # nibabel 会根据文件扩展名 .gz 自动进行 gzip 压缩。
        nifti_image = nib.Nifti1Image(numpy_array, affine_matrix)
        nib.save(nifti_image, output_path)
        print(f"文件成功保存至: {output_path}")

    except Exception as e:
        print(f"保存NIfTI文件时发生错误: {e}")


def parse_xcat_log_from_path(log_path: str):
    """
    从一个给定的完整路径解析XCAT日志文件，直接返回dims和voxel_size。

    Args:
        log_path (str): XCAT日志文件的完整路径。

    Returns:
        tuple[tuple, tuple] | tuple[None, None]:
            成功时返回 (dims, voxel_size)，例如 ((224, 224, 201), (2.0, 2.0, 2.0))。
            失败时返回 (None, None)。
    """
    print(f"正在尝试读取日志文件: {log_path}")
    if not os.path.exists(log_path):
        print(f"错误: 日志文件未找到 -> {log_path}")
        return None, None

    # 定义参数映射：{日志中的键: (内部存储名, 转换函数)}
    param_map = {
        "array_size": ("array_size", int),
        "pixel width": ("pixel_width_mm", lambda v: float(v) * 10),
        "slice width": ("slice_width_mm", lambda v: float(v) * 10),
        "starting slice number": ("start_slice", int),
        "ending slice number": ("end_slice", int),
        "rotation deg": ("rotation_deg", float),
    }

    parsed_params = {}
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 使用正则表达式稳健地提取 "key = value" 格式
                match = re.match(r"^\s*([\w\s]+?)\s*=\s*([\d.]+)", line.strip())
                if match:
                    key = match.group(1).strip()
                    value = match.group(2)

                    # 如果key在我们的映射中，就进行处理
                    if key in param_map:
                        internal_key, transform_func = param_map[key]
                        parsed_params[internal_key] = transform_func(value)

                angle_match = re.match(r"^\s*-*Angles \(x, y, z\)\s*=\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)", line.strip())
                if angle_match:
                    parsed_params["rotation_deg"] = tuple(map(float, angle_match.groups()))

                # 如果所有需要的参数都已找到，则提前退出
                if len(parsed_params) == len(param_map):
                    break
    except Exception as e:
        print(f"读取或解析日志文件时发生错误: {e}")
        return None, None

    # 验证并计算最终结果
    required_keys = ["array_size", "pixel_width_mm", "slice_width_mm", "start_slice", "end_slice"]
    if not all(key in parsed_params for key in required_keys):
        print(f"警告: 在 '{log_path}' 中未能找到所有必需的参数。")
        return None, None

    print("成功从日志中提取所有参数！")

    # 在函数内部完成计算，只返回最终需要的值
    dims = (
        parsed_params['array_size'],
        parsed_params['array_size'],
        parsed_params["end_slice"] - parsed_params['start_slice'] + 1
    )
    voxel_size = (
        parsed_params['pixel_width_mm'],
        parsed_params['pixel_width_mm'],
        parsed_params['slice_width_mm']
    )

    return dims, voxel_size, parsed_params['rotation_deg']
