import numpy as np
import torch
from scipy.ndimage import convolve


def generate_bssfp_signal(pd: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, tr: float = 3.0, te: float = 1.5,
                          flip_angle_deg: float = 60) -> torch.Tensor:
    """
    根据给定的组织属性图(PD, T1, T2)和MR序列参数，使用bSSFP信号模型生成MR图像。

    此函数是一个纯粹的计算模块，不涉及任何体模或标签的转换。
    信号方程与 MRXCAT_CMR_CINE.m 中的模型一致。

    Args:
        pd (torch.Tensor): 质子密度图 (Proton Density)，单位为百分比(%)。
        t1 (torch.Tensor): T1弛豫时间图，单位为毫秒 (ms)。
        t2 (torch.Tensor): T2弛豫时间图，单位为毫秒 (ms)。
        tr (float): 重复时间 (Repetition Time)，单位为毫秒 (ms)。
        te (float): 回波时间 (Echo Time)，单位为毫秒 (ms)。
        flip_angle_deg (float): 翻转角 (Flip Angle)，单位为度 (degrees)。

    Returns:
        torch.Tensor: 模拟生成的、代表信号强度的灰度MR图像。
    """
    print(f"开始应用 bSSFP 信号模型 (TR={tr}ms, TE={te}ms, Flip Angle={flip_angle_deg}°)...")

    # --- 输入验证 ---
    if not (pd.shape == t1.shape == t2.shape):
        raise ValueError("输入的PD, T1, T2数组必须具有相同的形状。")

    # --- 步骤 1: 预处理输入，防止计算错误 ---
    # 确保在GPU上进行计算
    device = pd.device
    t1_proc = t1.clone()
    t2_proc = t2.clone()

    # 防止T1/T2出现零值，避免后续计算中出现除零错误
    t1_proc[t1_proc == 0] = 1e-9
    t2_proc[t2_proc == 0] = 1e-9

    # --- 步骤 2: 应用平衡稳态自由进动 (bSSFP) 信号模型 ---
    # 将翻转角从度转换为弧度
    flip_angle_rad = np.deg2rad(flip_angle_deg)

    # 计算信号强度:
    # S = PD * sin(alpha) * exp(-TE/T2) / ( (T1/T2 + 1) - cos(alpha) * (T1/T2 - 1) )
    numerator = pd * torch.sin(torch.tensor(flip_angle_rad, device=device)) * torch.exp(-te / t2_proc)
    denominator = (t1_proc / t2_proc + 1) - torch.cos(torch.tensor(flip_angle_rad, device=device)) * (
            t1_proc / t2_proc - 1)

    # 再次检查分母，防止在特定翻转角和T1/T2比值下出现零
    denominator[denominator == 0] = 1e-9

    signal_image = numerator / denominator

    print("信号模拟完成。")
    return signal_image


def apply_low_pass_filter(image_3d: torch.Tensor, filter_strength: float = 1.5) -> torch.Tensor:
    """
    (轻微模糊) 对3D图像的每个2D切片应用一个低通滤波器。

    此函数模仿 MRXCAT.m 中的 lowPassFilter 方法，使用一个圆盘平均核 (disk kernel)
    来对图像进行轻微的模糊处理，以模拟部分容积效应或轻微的运动模糊。

    Args:
        image_3d (torch.Tensor): 输入的3D单通道图像 (depth, height, width)。
        filter_strength (float): 滤波器的强度，对应于圆盘核的半径。
                                 值越大，模糊效果越强。

    Returns:
        torch.Tensor: 经过模糊处理后的3D图像。
    """
    print(f"开始应用低通滤波 (模糊)，强度: {filter_strength}...")
    if image_3d.ndim != 3:
        raise ValueError("输入图像必须是3D数组 (depth, height, width)。")

    # 创建一个圆盘形状的卷积核 (disk kernel)
    radius = int(filter_strength)
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    kernel = np.array(x ** 2 + y ** 2 <= radius ** 2, dtype=np.float32)
    kernel /= kernel.sum()

    # 将NumPy核转换为PyTorch张量并移动到与输入图像相同的设备
    device = image_3d.device
    kernel_tensor = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0).to(device)

    # PyTorch的conv2d需要 (N, C, H, W) 格式的输入
    # 我们将 (D, H, W) 的图像看作是 (D, 1, H, W) 的批次
    image_4d = image_3d.unsqueeze(1)

    # 使用padding='same'来模拟 mode='reflect' 的效果
    blurred_image_4d = torch.nn.functional.conv2d(image_4d, kernel_tensor, padding='same')

    print("低通滤波完成。")
    return blurred_image_4d.squeeze(1)


def apply_coil_sensitivities(image_3d: torch.Tensor, coil_sens_maps: torch.Tensor) -> torch.Tensor:
    """
    (线圈引入的亮度不均) 将多线圈灵敏度图谱应用到单通道图像上。

    此函数模仿 MRXCAT.m 中的 multiplyCoilMaps 方法。它将一个单通道的理想
    MR图像与每个接收线圈的灵敏度图进行逐元素相乘，生成一个多通道的图像，
    每个通道代表一个线圈接收到的信号。

    Args:
        image_3d (torch.Tensor): 输入的3D单通道图像 (height, width, depth)。
        coil_sens_maps (torch.Tensor): 4D的复数线圈灵敏度图谱，形状为
                                     (height, width, depth, num_coils)。

    Returns:
        torch.Tensor: 经过线圈效应调制的4D多通道复数图像。
    """
    print(f"开始应用 {coil_sens_maps.shape[-1]} 个线圈的灵敏度图谱...")
    if image_3d.ndim != 3 or coil_sens_maps.ndim != 4:
        raise ValueError("输入图像必须是3D，线圈图谱必须是4D。")
    if image_3d.shape != coil_sens_maps.shape[:-1]:
        print(image_3d.shape, coil_sens_maps.shape[:-1])
        raise ValueError("图像和线圈图谱的空间维度必须匹配。")

    # 使用PyTorch的广播机制 (broadcasting) 高效地完成操作
    # image_3d.unsqueeze(-1) 将其形状变为 (h, w, d, 1)
    # 然后可以与 (h, w, d, num_coils) 的线圈图谱相乘
    image_3d = image_3d.unsqueeze(-1)
    multi_coil_image = image_3d * coil_sens_maps

    print("线圈灵敏度应用完成。")
    return multi_coil_image
