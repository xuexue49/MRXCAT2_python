import os
import re
import glob
import pstats
import cProfile
import numpy as np
import nibabel as nib


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

        return image_data_3d

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

    return dims, voxel_size


def simplify_xcat_labels(xcat_array: np.ndarray) -> np.ndarray:
    """
    将XCAT体模中数十个详细的组织标签简化为7个主要类别。
    该函数逻辑完全基于 phantomModel.py 文件中的 Phantom 类。

    类别映射关系:
    - 0: Muscle (肌肉)
    - 1: Blood (血液)
    - 2: Air (空气)
    - 3: Liver (肝脏)
    - 4: Fat (脂肪)
    - 5: Bone (骨骼)
    - 6: Unknown (未知/其他)

    Args:
        xcat_array (np.ndarray): 包含原始XCAT详细标签的Numpy数组。

    Returns:
        np.ndarray: 包含7个简化类别标签的Numpy数组。
    """
    print("开始简化XCAT标签...")
    if not isinstance(xcat_array, np.ndarray):
        raise TypeError("输入必须是一个Numpy数组。")

    # [cite_start]根据 phantomModel.py 中的定义，为不同组织创建布尔掩码 [cite: 2]
    muscle_mask = np.isin(xcat_array, [1, 2, 3, 4, 10])
    blood_mask = np.isin(xcat_array, [5, 6, 7, 8])
    air_mask = np.isin(xcat_array, [0, 15, 16])
    liver_mask = np.isin(xcat_array, [13, 40, 41, 42, 43, 52])
    fat_mask = np.isin(xcat_array, [50, 99])
    bone_mask = np.isin(xcat_array, [31, 32, 33, 34, 35, 51])

    # [cite_start]任何不属于上述已知类别的体素都将被标记为“未知” [cite: 2]
    known_mask = air_mask | muscle_mask | blood_mask | bone_mask | fat_mask | liver_mask
    unknown_mask = ~known_mask

    # 创建一个新的数组来存储简化后的标签
    # [cite_start]默认值为0 (Muscle)，然后用其他类别的值覆盖 [cite: 2]
    simplified_array = np.zeros_like(xcat_array, dtype=np.int32)
    simplified_array[blood_mask] = 1
    simplified_array[air_mask] = 2
    simplified_array[liver_mask] = 3
    simplified_array[fat_mask] = 4
    simplified_array[bone_mask] = 5
    simplified_array[unknown_mask] = 6

    print("标签简化完成。")
    return simplified_array


def assign_tissue_properties(simplified_array: np.ndarray, property_set: str = 'ours') -> np.ndarray:
    """
    为简化后的体模类别分配初步的、平滑的组织物理属性值。
    该函数逻辑和物理参数值完全基于 phantomModel.py 文件。

    Args:
        simplified_array (np.ndarray): 由 simplify_xcat_labels 函数生成的、包含7个类别标签的数组。
        property_set (str, optional): 使用哪一套物理属性值，可选值为 'ours' 或 'xcat'。
                                      默认为 'ours'。

    Returns:
        np.ndarray: 一个形状为 (..., 4) 的数组，最后一维代表归一化后的 [PD, T1, T2, T2*]。
    """
    print(f"开始使用 '{property_set}' 属性集分配组织属性...")
    if not isinstance(simplified_array, np.ndarray):
        raise TypeError("输入必须是一个Numpy数组。")

    # [cite_start]从 phantomModel.py 中提取的物理属性值 [cite: 2]
    tissue_property_values = {}
    pd_max_ours, t1_max_ours, t2_max_ours, t2star_max_ours = 100., 1400., 285., 70.
    tissue_property_values['ours'] = [
        np.array([88. / pd_max_ours, 1000. / t1_max_ours, 43. / t2_max_ours, 28. / t2star_max_ours]),  # 0: Muscle
        np.array([79. / pd_max_ours, 1387. / t1_max_ours, 280. / t2_max_ours, 66. / t2star_max_ours]),  # 1: Blood
        np.array([34. / pd_max_ours, 1171. / t1_max_ours, 61. / t2_max_ours, 1. / t2star_max_ours]),  # 2: Air
        np.array([87. / pd_max_ours, 661. / t1_max_ours, 57. / t2_max_ours, 34. / t2star_max_ours]),  # 3: Liver
        np.array([60. / pd_max_ours, 250. / t1_max_ours, 70. / t2_max_ours, 39. / t2star_max_ours]),  # 4: Fat
        np.array([71. / pd_max_ours, 250. / t1_max_ours, 20. / t2_max_ours, 1. / t2star_max_ours]),  # 5: Bone
        np.array([65. / pd_max_ours, 750. / t1_max_ours, 60 / t2_max_ours, 30. / t2star_max_ours])  # 6: Unknown
    ]

    pd_max_xcat, t1_max_xcat, t2_max_xcat, t2star_max_xcat = 100., 1300., 105., 55.
    tissue_property_values['xcat'] = [
        np.array([80. / pd_max_xcat, 900. / t1_max_xcat, 50. / t2_max_xcat, 31. / t2star_max_xcat]),  # 0: Muscle
        np.array([95. / pd_max_xcat, 1200. / t1_max_xcat, 100. / t2_max_xcat, 50. / t2star_max_xcat]),  # 1: Blood
        np.array([34. / pd_max_xcat, 1171. / t1_max_xcat, 61. / t2_max_xcat, 1. / t2star_max_xcat]),  # 2: Air
        np.array([90. / pd_max_xcat, 800. / t1_max_xcat, 50. / t2_max_xcat, 31. / t2star_max_xcat]),  # 3: Liver
        np.array([70. / pd_max_xcat, 350. / t1_max_xcat, 30. / t2_max_xcat, 20. / t2star_max_xcat]),  # 4: Fat
        np.array([12. / pd_max_xcat, 250. / t1_max_xcat, 20. / t2_max_xcat, 1. / t2star_max_xcat]),  # 5: Bone
        np.array([70. / pd_max_xcat, 700. / t1_max_xcat, 50 / t2_max_xcat, 30. / t2star_max_xcat])  # 6: Unknown
    ]

    if property_set not in tissue_property_values:
        raise ValueError(f"属性集 '{property_set}' 无效, 请选择 'ours' 或 'xcat'。")

    properties = tissue_property_values[property_set]

    # 创建一个用于存储组织属性图 (Tissue Property Map, TPM) 的空数组
    # 形状为 (height, width, depth, 4)
    tpm_array = np.zeros(simplified_array.shape + (4,), dtype=float)

    # [cite_start]使用Numpy广播机制高效地为每个类别赋值 [cite: 2]
    for class_index in range(7):
        # 找到所有等于当前类别索引的像素，并乘以对应的物理属性向量
        tpm_array += (simplified_array[..., None] == class_index) * properties[class_index]

    print("组织属性分配完成。")
    return tpm_array


if __name__ == "__main__":
    # --- 1. 配置图像和路径参数 ---
    # 图像元数据
    profiler = cProfile.Profile()
    profiler.enable()

    numpy_dtype = '<f4'  # 数据类型: 小端, 4字节浮点数

    # 路径参数
    base_path = "/home/chenxinpeng/MRXCAT2_python/raw_data"
    case_name = "Patient1"

    # 根据新要求构建输入和输出目录
    input_dir = os.path.join(base_path, case_name, "xcat_bin")
    output_dir = os.path.join(base_path, case_name, "mask")
    log_path = os.path.join(base_path, case_name, "log")

    dims, voxel_size = parse_xcat_log_from_path(log_path)

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
        image_matrix = load_bin_as_numpy(
            bin_path=input_filename,
            dims=dims,
            numpy_dtype=numpy_dtype
        )

        # 检查矩阵是否成功加载，然后保存
        if image_matrix is not None:
            print(f"成功加载矩阵，Shape: {image_matrix.shape}, Dtype: {image_matrix.dtype}")

            image_matrix = simplify_xcat_labels(image_matrix)

            save_numpy_as_nifti(
                numpy_array=image_matrix,
                output_path=output_filename,
                voxel_size=voxel_size
            )

    profiler.disable()
