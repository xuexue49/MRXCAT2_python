import numpy as np
import torch
from tqdm import tqdm


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

    # --- 核心修改：传回CPU后，重排数组维度 ---
    # 原始PyTorch/Numpy维度: (0:height, 1:width, 2:depth, 3:coils)
    # 目标维度: (2:depth, 0:height, 1:width, 3:coils)
    # numpy_maps = sensitivity_maps.cpu().numpy()
    return sensitivity_maps
