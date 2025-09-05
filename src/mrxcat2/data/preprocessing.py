import os
import re
import glob
import pstats
import cProfile
import numpy as np
import nibabel as nib
from scipy.ndimage import convolve


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


def create_tissue_property_lookup_table(property_set: str = 'xcat', normalize: bool = False) -> tuple:
    """
    创建组织属性的查找表 (Lookup Table)。
    此函数封装了所有物理参数的定义和计算，应在循环或多次调用前执行一次。

    Args:
        property_set (str, optional): 使用哪一套物理属性值，可选 'ours' 或 'xcat'。默认为 'ours'。
        normalize (bool, optional): 是否返回归一化的属性值。
                                    - True: 返回 [0, 1] 范围内的归一化值。
                                    - False: 返回物理单位的真实值 (PD in %, T1/T2 in ms)。
                                    默认为 True。

    Returns:
        tuple[np.ndarray, dict]:
            - 第一个元素是 (7, 4) 的Numpy数组，作为查找表。
            - 第二个元素是一个包含最大值的字典，用于可能的后续计算。
    """
    print(f"正在创建 '{property_set}' 属性集的查找表 (Normalized: {normalize})...")

    # 定义两套物理属性的原始值和最大值
    # [cite_start]从 phantomModel.py 中提取的物理属性值 [cite: 2]
    raw_values = {
        'ours': {
            'max_values': {'pd': 100., 't1': 1400., 't2': 285., 't2s': 70.},
            'properties': np.array([
                [88., 1000., 43., 28.],  # 0: Muscle
                [79., 1387., 280., 66.],  # 1: Blood
                [34., 1171., 61., 1.],  # 2: Air
                [87., 661., 57., 34.],  # 3: Liver
                [60., 250., 70., 39.],  # 4: Fat
                [71., 250., 20., 1.],  # 5: Bone
                [65., 750., 60., 30.]  # 6: Unknown
            ])
        },
        'xcat': {
            'max_values': {'pd': 100., 't1': 1300., 't2': 105., 't2s': 55.},
            'properties': np.array([
                [80., 900., 50., 31.],  # 0: Muscle
                [95., 1200., 100., 50.],  # 1: Blood
                [34., 1171., 61., 1.],  # 2: Air
                [90., 800., 50., 31.],  # 3: Liver
                [70., 350., 30., 20.],  # 4: Fat
                [12., 250., 20., 1.],  # 5: Bone
                [70., 700., 50., 30.]  # 6: Unknown
            ])
        }
    }
    # [cite_end]

    if property_set not in raw_values:
        raise ValueError(f"属性集 '{property_set}' 无效, 请选择 'ours' 或 'xcat'。")

    selected_set = raw_values[property_set]
    lookup_table = selected_set['properties']
    max_vals_dict = selected_set['max_values']

    if normalize:
        # 创建一个 (1, 4) 的数组用于广播除法
        max_vals_array = np.array([max_vals_dict['pd'], max_vals_dict['t1'], max_vals_dict['t2'], max_vals_dict['t2s']])
        lookup_table = lookup_table / max_vals_array

    return lookup_table, max_vals_dict


def assign_tissue_properties(simplified_array: np.ndarray, lookup_table: np.ndarray) -> tuple:
    """
    (高效版) 使用预先计算好的查找表为简化体模分配组织属性。
    此函数只执行核心的、高速的Numpy索引映射操作。

    Args:
        simplified_array (np.ndarray): 包含7个类别标签的数组 (值为 0-6)。
        lookup_table (np.ndarray): 由 create_tissue_property_lookup_table 函数生成的 (7, 4) 查找表。

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            返回一个包含四个Numpy数组的元组 (PD, T1, T2, T2*)。
    """
    if not isinstance(simplified_array, np.ndarray):
        raise TypeError("输入 simplified_array 必须是一个Numpy数组。")
    if not (isinstance(lookup_table, np.ndarray) and lookup_table.shape == (7, 4)):
        raise ValueError("lookup_table 必须是一个 (7, 4) 的Numpy数组。")

    # --- 核心操作: 使用高级索引进行高速映射 ---
    tpm_array = lookup_table[simplified_array]

    # 返回分离后的四个属性图
    return tpm_array[..., 0], tpm_array[..., 1], tpm_array[..., 2], tpm_array[..., 3]


def generate_bssfp_signal(pd: np.ndarray, t1: np.ndarray, t2: np.ndarray, tr: float = 3.0, te: float = 1.5,
                          flip_angle_deg: float = 60) -> np.ndarray:
    """
    根据给定的组织属性图(PD, T1, T2)和MR序列参数，使用bSSFP信号模型生成MR图像。

    此函数是一个纯粹的计算模块，不涉及任何体模或标签的转换。
    信号方程与 MRXCAT_CMR_CINE.m 中的模型一致。

    Args:
        pd (np.ndarray): 质子密度图 (Proton Density)，单位为百分比(%)。
        t1 (np.ndarray): T1弛豫时间图，单位为毫秒 (ms)。
        t2 (np.ndarray): T2弛豫时间图，单位为毫秒 (ms)。
        tr (float): 重复时间 (Repetition Time)，单位为毫秒 (ms)。
        te (float): 回波时间 (Echo Time)，单位为毫秒 (ms)。
        flip_angle_deg (float): 翻转角 (Flip Angle)，单位为度 (degrees)。

    Returns:
        np.ndarray: 模拟生成的、代表信号强度的灰度MR图像。
    """
    print(f"开始应用 bSSFP 信号模型 (TR={tr}ms, TE={te}ms, Flip Angle={flip_angle_deg}°)...")

    # --- 输入验证 ---
    if not (pd.shape == t1.shape == t2.shape):
        raise ValueError("输入的PD, T1, T2数组必须具有相同的形状。")

    # --- 步骤 1: 预处理输入，防止计算错误 ---
    # 创建输入的副本以避免修改原始数组
    t1_proc = t1.copy()
    t2_proc = t2.copy()

    # 防止T1/T2出现零值，避免后续计算中出现除零错误
    t1_proc[t1_proc == 0] = 1e-9
    t2_proc[t2_proc == 0] = 1e-9

    # --- 步骤 2: 应用平衡稳态自由进动 (bSSFP) 信号模型 ---
    # 将翻转角从度转换为弧度
    flip_angle_rad = np.deg2rad(flip_angle_deg)

    # 计算信号强度:
    # S = PD * sin(alpha) * exp(-TE/T2) / ( (T1/T2 + 1) - cos(alpha) * (T1/T2 - 1) )
    numerator = pd * np.sin(flip_angle_rad) * np.exp(-te / t2_proc)
    denominator = (t1_proc / t2_proc + 1) - np.cos(flip_angle_rad) * (t1_proc / t2_proc - 1)

    # 再次检查分母，防止在特定翻转角和T1/T2比值下出现零
    denominator[denominator == 0] = 1e-9

    signal_image = numerator / denominator

    print("信号模拟完成。")
    return signal_image


def apply_low_pass_filter(image_3d: np.ndarray, filter_strength: float = 1.5) -> np.ndarray:
    """
    (轻微模糊) 对3D图像的每个2D切片应用一个低通滤波器。

    此函数模仿 MRXCAT.m 中的 lowPassFilter 方法，使用一个圆盘平均核 (disk kernel)
    来对图像进行轻微的模糊处理，以模拟部分容积效应或轻微的运动模糊。

    Args:
        image_3d (np.ndarray): 输入的3D单通道图像 (height, width, depth)。
        filter_strength (float): 滤波器的强度，对应于圆盘核的半径。
                                 值越大，模糊效果越强。

    Returns:
        np.ndarray: 经过模糊处理后的3D图像。
    """
    print(f"开始应用低通滤波 (模糊)，强度: {filter_strength}...")
    if image_3d.ndim != 3:
        raise ValueError("输入图像必须是3D数组 (height, width, depth)。")

    # 创建一个圆盘形状的卷积核 (disk kernel)
    radius = int(filter_strength)
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    kernel = np.array(x ** 2 + y ** 2 <= radius ** 2, dtype=np.float32)
    kernel /= kernel.sum()  # 归一化，使其成为一个平均滤波器

    # 创建一个用于存储结果的空数组
    blurred_image = np.zeros_like(image_3d, dtype=image_3d.dtype)

    # 对每个2D切片独立应用卷积
    for i in range(image_3d.shape[2]):
        blurred_image[..., i] = convolve(image_3d[..., i], kernel, mode='reflect')

    print("低通滤波完成。")
    return blurred_image


def apply_coil_sensitivities(image_3d: np.ndarray, coil_sens_maps: np.ndarray) -> np.ndarray:
    """
    (线圈引入的亮度不均) 将多线圈灵敏度图谱应用到单通道图像上。

    此函数模仿 MRXCAT.m 中的 multiplyCoilMaps 方法。它将一个单通道的理想
    MR图像与每个接收线圈的灵敏度图进行逐元素相乘，生成一个多通道的图像，
    每个通道代表一个线圈接收到的信号。

    Args:
        image_3d (np.ndarray): 输入的3D单通道图像 (height, width, depth)。
        coil_sens_maps (np.ndarray): 4D的复数线圈灵敏度图谱，形状为
                                     (height, width, depth, num_coils)。

    Returns:
        np.ndarray: 经过线圈效应调制的4D多通道复数图像。
    """
    print(f"开始应用 {coil_sens_maps.shape[-1]} 个线圈的灵敏度图谱...")
    if image_3d.ndim != 3 or coil_sens_maps.ndim != 4:
        raise ValueError("输入图像必须是3D，线圈图谱必须是4D。")
    if image_3d.shape != coil_sens_maps.shape[:-1]:
        raise ValueError("图像和线圈图谱的空间维度必须匹配。")

    # 使用NumPy的广播机制 (broadcasting) 高效地完成操作
    # image_3d[..., np.newaxis] 将其形状变为 (h, w, d, 1)
    # 然后可以与 (h, w, d, num_coils) 的线圈图谱相乘
    multi_coil_image = image_3d[..., np.newaxis] * coil_sens_maps

    print("线圈灵敏度应用完成。")
    return multi_coil_image


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
    lookup_table, max_vals_dict = create_tissue_property_lookup_table()

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
        image_matrix = simplify_xcat_labels(image_matrix)
        pd, t1, t2, t2s = assign_tissue_properties(image_matrix, lookup_table)

        image = generate_bssfp_signal(pd, t1, t2)
        image = apply_low_pass_filter(image, filter_strength=1.5)

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
