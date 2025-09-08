import torch
import os
from .res_unet_model import ResUNet
from ..phantom.tissue import TissueProcessor


class TextureGenerator:
    """
    一个用于管理和应用深度学习纹理的类。

    该类在初始化时加载预训练的ResUNet模型权重，并准备好在GPU上
    进行推理。它提供了：
    1. `apply` 方法：为输入的组织属性图添加纹理。
    2. `normalize` 方法：对纹理化后的属性图进行校正，使其平均值符合生理标准。
    """

    def __init__(self, model_path: str, device: torch.device):
        """
        初始化TextureGenerator实例。

        Args:
            model_path (str): 预训练的ResUNet模型权重文件 (.pt) 的路径。
            device (torch.device): 用于加载模型和执行计算的设备 (例如, 'cuda:0' 或 'cpu')。
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"错误：找不到纹理模型文件 -> {model_path}")

        print("正在初始化TextureGenerator并加载纹理网络...")
        self.device = device

        # 定义网络结构并移动到指定设备
        self.net = ResUNet(n_channels=3, n_classes=3).to(self.device)

        # 加载预训练的权重
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))

        # 将网络设置为评估模式
        self.net.eval()

        # --- 新增：定义用于校正的标准组织属性参考表 ---
        # 这些值是各类组织的PD, T1, T2的公认平均值 (来自XCAT标准)
        # 顺序: 0:Muscle, 1:Blood, 2:Air, 3:Liver, 4:Fat, 5:Bone, 6:Unknown
        self.xcat_reference_properties = torch.tensor([
            # PD,   T1,    T2
            [80., 900., 50.],  # 0: Muscle (肌肉)
            [95., 1200., 100.],  # 1: Blood (血液)
            [34., 1171., 61.],  # 2: Air (空气)
            [90., 800., 50.],  # 3: Liver (肝脏)
            [70., 350., 30.],  # 4: Fat (脂肪)
            [12., 250., 20.],  # 5: Bone (骨骼)
            [70., 700., 50.]  # 6: Unknown (未知/平均)
        ], dtype=torch.float32, device=self.device)

        print("纹理网络加载并准备就绪。")

    @torch.no_grad()
    def apply(
            self,
            pd_map: torch.Tensor,
            t1_map: torch.Tensor,
            t2_map: torch.Tensor,
            t2s_map: torch.Tensor,
            tissue_processor: TissueProcessor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        使用预加载的网络为输入的组织属性图（TPMs）添加纹理。
        """
        print("开始应用深度学习纹理...")

        max_vals = tissue_processor.max_vals_dict
        pd_max, t1_max, t2_max = max_vals['pd'], max_vals['t1'], max_vals['t2']

        pd_norm = pd_map / pd_max
        t1_norm = t1_map / t1_max
        t2_norm = t2_map / t2_max

        tpm_norm_flat = torch.stack([pd_norm, t1_norm, t2_norm], dim=-1)
        net_input = tpm_norm_flat.permute(2, 3, 0, 1)

        textured_tpm_norm = self.net(net_input)
        textured_tpm_norm = textured_tpm_norm.permute(2, 3, 0, 1)

        pd_textured = textured_tpm_norm[..., 0] * pd_max
        t1_textured = textured_tpm_norm[..., 1] * t1_max
        t2_textured = textured_tpm_norm[..., 2] * t2_max

        print("纹理应用完成。")
        return pd_textured, t1_textured, t2_textured, t2s_map

    @torch.no_grad()
    def normalize_properties(
            self,
            pd_textured: torch.Tensor,
            t1_textured: torch.Tensor,
            t2_textured: torch.Tensor,
            simplified_labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对纹理化后的组织属性图进行校正，使其每个组织的平均值与XCAT标准匹配。

        Args:
            pd_textured (torch.Tensor): 经过纹理化的PD图。
            t1_textured (torch.Tensor): 经过纹理化的T1图。
            t2_textured (torch.Tensor): 经过纹理化的T2图。
            simplified_labels (torch.Tensor): 简化的7类组织标签图 (mask)。

        Returns:
            tuple[torch.Tensor, ...]: 返回校正后的 (PD, T1, T2) 图。
        """
        print("开始校正纹理化组织属性...")

        # 创建输出张量的副本以进行修改
        pd_corrected = pd_textured.clone()
        t1_corrected = t1_textured.clone()
        t2_corrected = t2_textured.clone()

        # 遍历7个简化的组织类别
        for class_idx in range(self.xcat_reference_properties.shape[0]):
            # 创建对应类别的布尔掩码
            mask = (simplified_labels == class_idx)

            # 如果图像中不存在这个类别的像素，则跳过
            if not torch.any(mask):
                continue

            # 获取该类别的目标（参考）属性值
            target_pd, target_t1, target_t2 = self.xcat_reference_properties[class_idx]

            # --- 校正PD ---
            mean_pd = pd_textured[mask].mean()
            if mean_pd > 1e-6:  # 避免除以零
                pd_corrected[mask] *= (target_pd / mean_pd)

            # --- 校正T1 ---
            mean_t1 = t1_textured[mask].mean()
            if mean_t1 > 1e-6:
                t1_corrected[mask] *= (target_t1 / mean_t1)

            # --- 校正T2 ---
            mean_t2 = t2_textured[mask].mean()
            if mean_t2 > 1e-6:
                t2_corrected[mask] *= (target_t2 / mean_t2)

        print("组织属性校正完成。")
        return pd_corrected, t1_corrected, t2_corrected
