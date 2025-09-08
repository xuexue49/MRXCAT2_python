import os
import torch
import glob
import pstats
import cProfile
from tqdm import tqdm
from scipy.ndimage import convolve

import numpy as np

from mrxcat2.io.utils import load_bin_as_numpy, save_numpy_as_nifti, parse_xcat_log_from_path
from mrxcat2.phantom.tissue import simplify_xcat_labels, create_tissue_property_lookup_table, assign_tissue_properties


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
        raise ValueError("输入图像必须是3D数组 (depth, height, width)。")

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
        print(image_3d.shape, coil_sens_maps.shape[:-1])
        raise ValueError("图像和线圈图谱的空间维度必须匹配。")

    # 使用NumPy的广播机制 (broadcasting) 高效地完成操作
    # image_3d[..., np.newaxis] 将其形状变为 (h, w, d, 1)
    # 然后可以与 (h, w, d, num_coils) 的线圈图谱相乘
    multi_coil_image = image_3d[..., np.newaxis] * coil_sens_maps

    print("线圈灵敏度应用完成。")
    return multi_coil_image


def _calculate_coil_centers(num_coils: int, coil_distance_mm: float, coils_per_row: int) -> tuple:
    """
    (辅助函数) 在圆柱体表面计算线圈中心的几何布局。
    此函数是 MATLAB 'coilCentres' 方法的直接Python翻译。

    Args:
        num_coils (int): 总线圈数量。
        coil_distance_mm (float): 线圈中心到扫描原点的距离 (身体半径)，单位mm。
        coils_per_row (int): 每圈（ring）最多放置的线圈数。

    Returns:
        tuple[np.ndarray, float]:
            - coil_centers: (num_coils, 3) 的数组，包含每个线圈的 (x, y, z) 坐标。
            - coil_radius: 单个线圈的半径，单位mm。
    """
    # 预设的特定角度分布 (单位: 弧度)
    angles = {
        2: np.deg2rad([150, 210]),
        3: np.deg2rad([130, 180, 230]),
        4: np.deg2rad([150, 210, 330, 30]),
        5: np.deg2rad([130, 180, 230, 330, 30]),
        6: np.deg2rad([130, 180, 230, 310, 0, 50]),
    }

    # 确定线圈环（ring）的数量和每环的线圈数
    if num_coils % coils_per_row == 0:
        num_rings = num_coils // coils_per_row
        coils_in_ring = [coils_per_row] * num_rings
    else:
        full_rings = max(num_coils // coils_per_row, 0)
        remaining_coils = num_coils - full_rings * coils_per_row
        if remaining_coils > coils_per_row:
            coils_in_ring = [remaining_coils // 2, remaining_coils - (remaining_coils // 2)]
            if full_rings > 0:
                coils_in_ring += [coils_per_row] * full_rings
        else:
            coils_in_ring = [remaining_coils] + [coils_per_row] * full_rings
        num_rings = len(coils_in_ring)

    # 根据布局计算线圈半径
    min_angle = np.pi * 2 / max(coils_in_ring) if max(coils_in_ring) > 6 else np.deg2rad(50)
    coil_radius = 0.5 * coil_distance_mm * min_angle

    # 计算每个环的z轴坐标
    z_coords = np.arange(-(num_rings - 1), num_rings, 2) * coil_radius

    coil_centers = np.zeros((num_coils, 3))
    coil_counter = 0
    for i, num_c in enumerate(coils_in_ring):
        if num_c in angles:
            theta = angles[num_c]
        else:
            theta = np.linspace(0, 2 * np.pi * (num_c - 1) / num_c, num_c)

        x = coil_distance_mm * np.cos(theta)
        y = coil_distance_mm * np.sin(theta)

        for j in range(num_c):
            coil_centers[coil_counter] = [y[j], x[j], z_coords[i]]  # 注意MATLAB中x,y与常规定义相反
            coil_counter += 1

    return coil_centers, coil_radius


def calculate_coil_sensitivities(
        image_shape: tuple, voxel_size_mm: tuple, num_coils: int, coil_distance_mm: float = 600.0,
        coils_per_row: int = 8, rotation_deg: tuple = (133, 38, 62), integration_angles: int = 60) -> np.ndarray:
    """
    根据Biot-Savart定律计算并生成3D多线圈灵敏度图谱。
    此函数是 MATLAB 'calculateCoilMaps' 方法的精确Python实现。

    Args:
        image_shape (tuple): 最终图像的维度 (height, width, depth)。
        voxel_size_mm (tuple): 体素的物理尺寸 (dy, dx, dz)，单位mm。
        num_coils (int): 要模拟的总线圈数量。
        coil_distance_mm (float): 线圈中心到扫描原点的距离。
        coils_per_row (int): 每圈最多放置的线圈数。
        rotation_deg (tuple): 图像坐标系相对于线圈的旋转角度 (rot_x, rot_y, rot_z)，单位度。
        integration_angles (int): 用于数值积分的角度步数。

    Returns:
        np.ndarray: 形状为 (height, width, depth, num_coils) 的4D复数线圈灵敏度图谱。
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This function requires a PyTorch installation with GPU support.")
    device = torch.device("cuda")
    print(f"开始在GPU ({torch.cuda.get_device_name(device)}) 上计算 {num_coils} 个线圈的灵敏度图谱...")

    if num_coils <= 1:
        # 调整输出以匹配 (z, y, x, coils)
        return np.ones((image_shape[2], image_shape[0], image_shape[1], 1), dtype=np.complex64)

    # --- CPU上的初始设置 ---
    coil_centers, coil_radius = _calculate_coil_centers(num_coils, coil_distance_mm, coils_per_row)

    h, w, d = image_shape
    dy, dx, dz = voxel_size_mm
    x_coords = (np.arange(w) - w / 2) * dx
    y_coords = (np.arange(h) - h / 2) * dy
    z_coords = (np.arange(d) - d / 2) * dz

    ax, ay, az = np.deg2rad(rotation_deg)
    rot_x = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    rot_y = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rot_z = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    inv_rotation_matrix = np.linalg.inv(rot_x @ rot_z @ rot_y)

    d_theta = 2 * np.pi / integration_angles
    theta = np.arange(-np.pi, np.pi, d_theta)

    # --- 数据迁移到GPU ---
    Y, X, Z = np.meshgrid(y_coords, x_coords, z_coords, indexing='ij')
    coords_vec = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    rotated_coords = inv_rotation_matrix @ coords_vec

    rotated_coords_gpu = torch.from_numpy(rotated_coords).float().to(device)
    cos_theta_gpu = torch.from_numpy(np.cos(theta)).float().to(device)
    sin_theta_gpu = torch.from_numpy(np.sin(theta)).float().to(device)

    sensitivity_maps = torch.zeros(image_shape + (num_coils,), dtype=torch.complex64, device=device)

    # --- 在GPU上循环计算 ---
    for i in tqdm(range(num_coils), desc="在GPU上计算线圈"):
        coil_center = torch.from_numpy(coil_centers[i]).float().to(device)

        Xc = rotated_coords_gpu[0, :].view(h, w, d) - coil_center[0]
        Yc = rotated_coords_gpu[1, :].view(h, w, d) - coil_center[1]
        Zc = rotated_coords_gpu[2, :].view(h, w, d) - coil_center[2]

        Xc = Xc.unsqueeze(-1)
        Yc = Yc.unsqueeze(-1)
        Zc = Zc.unsqueeze(-1)

        denominator = (coil_radius ** 2 + Xc ** 2 + Yc ** 2 + Zc ** 2 -
                       2 * coil_radius * (Yc * cos_theta_gpu + Zc * sin_theta_gpu))
        denominator = torch.pow(torch.abs(denominator), 1.5)
        torch.clamp_(denominator, min=1e7)

        coil_angle = torch.atan2(coil_center[1], coil_center[0])
        sin_a, cos_a = torch.sin(coil_angle), torch.cos(coil_angle)

        nom_x = coil_radius * (Yc * cos_theta_gpu + Zc * sin_theta_gpu * cos_a - coil_radius * cos_a)
        nom_y = coil_radius * (-Xc * cos_theta_gpu + Zc * sin_theta_gpu * sin_a - coil_radius * sin_a)
        nom_z = coil_radius * (-Yc * sin_theta_gpu * sin_a - Xc * sin_theta_gpu * cos_a)

        sx = d_theta * torch.sum(nom_x / denominator, dim=-1)
        sy = d_theta * torch.sum(nom_y / denominator, dim=-1)
        sz = d_theta * torch.sum(nom_z / denominator, dim=-1)

        ang_y = sin_a * (Xc.squeeze(-1) + coil_center[0]) - cos_a * (Yc.squeeze(-1) + coil_center[1])
        ang_z = Zc.squeeze(-1)
        yz_angle = torch.angle(ang_y + 1j * ang_z)

        real_part = cos_a * sx + sin_a * sy
        imag_part = (-sin_a * sx + cos_a * sy) * torch.cos(yz_angle) + sz * torch.sin(yz_angle)

        sensitivity_maps[..., i] = torch.complex(real_part, imag_part)

    # --- 归一化 ---
    mean_sens = torch.mean(sensitivity_maps[sensitivity_maps != 0])
    sensitivity_maps /= mean_sens

    print("GPU计算完成。")

    # --- 核心修改：传回CPU后，重排数组维度 ---
    # 原始PyTorch/Numpy维度: (0:height, 1:width, 2:depth, 3:coils)
    # 目标维度: (2:depth, 0:height, 1:width, 3:coils)
    numpy_maps = sensitivity_maps.cpu().numpy()
    return numpy_maps


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

    # 读取生成参数
    dims, voxel_size, rotation_deg = parse_xcat_log_from_path(log_path)

    # 生成静态参数
    lookup_table, max_vals_dict = create_tissue_property_lookup_table()
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
        image_matrix = simplify_xcat_labels(image_matrix)
        pd, t1, t2, t2s = assign_tissue_properties(image_matrix, lookup_table)

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
