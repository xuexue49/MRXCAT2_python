import numpy as np
from typing import Tuple, Dict, Literal
import torch


class TissueProcessor:
    """
    一个用于处理XCAT体模组织属性的类。

    该类封装了将详细的XCAT组织标签简化为主要类别，并根据预定义的
    物理属性集为这些类别分配组织属性（如PD, T1, T2, T2*）的逻辑。

    优化：
    在初始化(__init__)时，此类会预先创建并存储所有需要的查找表(LUTs)，
    包括标签简化LUT和组织属性LUT，以避免在处理函数中重复计算。
    """

    def __init__(self, property_set: Literal['ours', 'xcat'] = 'ours', normalize: bool = False):
        """
        初始化TissueProcessor实例。

        在实例化时，会根据所选的属性集创建并存储一个查找表。

        Args:
            property_set (str, optional): 使用哪一套物理属性值。
                                        可选 'ours' 或 'xcat'。默认为 'ours'。
            normalize (bool, optional): 是否归一化属性值。
                                        - True: 返回 [0, 1] 范围内的归一化值。
                                        - False: 返回物理单位的真实值 (PD in %, T1/T2 in ms)。
                                        默认为 False。
        """
        if property_set not in ['ours', 'xcat']:
            raise ValueError(f"属性集 '{property_set}' 无效, 请选择 'ours' 或 'xcat'。")

        self.property_set = property_set
        self.normalize = normalize
        print(f"正在初始化 TissueProcessor，使用 '{self.property_set}' 属性集 (Normalized: {self.normalize})...")

        # 1. 创建组织属性查找表 (Numpy)
        self.lookup_table, self.max_vals_dict = self._create_property_lut()
        # 将其转换为Torch张量以备后用
        self.lookup_tensor = torch.from_numpy(self.lookup_table).float()

        # 2. 创建标签简化查找表 (Numpy)
        self.simplification_lut = self._create_simplification_lut()
        # 将其转换为Torch张量以备后用
        self.simplification_lut_tensor = torch.from_numpy(self.simplification_lut).long()

    def _create_property_lut(self) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        (私有方法) 创建并返回组织物理属性的查找表 (Lookup Table)。
        """
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

        selected_set = raw_values[self.property_set]
        lookup_table = selected_set['properties']
        max_vals_dict = selected_set['max_values']

        if self.normalize:
            max_vals_array = np.array(
                [max_vals_dict['pd'], max_vals_dict['t1'], max_vals_dict['t2'], max_vals_dict['t2s']])
            lookup_table = lookup_table / max_vals_array

        return lookup_table, max_vals_dict

    @staticmethod
    def _create_simplification_lut() -> np.ndarray:
        """
        (私有方法) 创建并返回用于简化XCAT标签的查找表。
        XCAT标签的最大值是99，所以我们创建一个大小为100的LUT。
        """
        # 默认值为6 (Unknown)
        lut = np.full(100, 6, dtype=np.int32)

        # 根据映射关系填充LUT
        lut[[1, 2, 3, 4, 10]] = 0  # Muscle
        lut[[5, 6, 7, 8]] = 1  # Blood
        lut[[0, 15, 16]] = 2  # Air
        lut[[13, 40, 41, 42, 43, 52]] = 3  # Liver
        lut[[50, 99]] = 4  # Fat
        lut[[31, 32, 33, 34, 35, 51]] = 5  # Bone
        return lut

    def simplify_xcat_labels(self, xcat_tensor: torch.Tensor) -> torch.Tensor:
        """
        (实例方法) 使用预先计算好的查找表将XCAT标签简化为7个主要类别。
        此版本在PyTorch上运行，以利用GPU加速。

        Args:
            xcat_tensor (torch.Tensor): 包含原始XCAT详细标签的Torch张量。

        Returns:
            torch.Tensor: 包含7个简化类别标签的Torch张量。
        """
        print("开始简化XCAT标签 (on GPU)...")
        if not isinstance(xcat_tensor, torch.Tensor):
            raise TypeError("输入必须是一个Torch张量。")

        device = xcat_tensor.device

        # 将预先生成的LUT移动到目标设备
        lut_tensor = self.simplification_lut_tensor.to(device)

        # 检查输入Tensor中的最大标签值是否超出LUT的范围
        max_label = xcat_tensor.max().item()
        if max_label >= len(lut_tensor):
            # 如果超出范围，动态扩展LUT
            print(f"警告：输入张量中的最大标签值({max_label})超出现有LUT大小({len(lut_tensor)})。正在动态扩展LUT...")
            new_lut = torch.full((max_label + 1,), 6, dtype=torch.long, device=device)
            new_lut[:len(lut_tensor)] = lut_tensor
            lut_tensor = new_lut

        simplified_tensor = lut_tensor[xcat_tensor]

        print("标签简化完成。")
        return simplified_tensor

    def assign_tissue_properties(self, simplified_tensor: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        (实例方法) 使用预先计算好的查找表为简化体模分配组织属性。

        Args:
            simplified_tensor (torch.Tensor): 包含7个类别标签的Tensor (值为 0-6)。

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                返回一个包含四个Torch张量的元组 (PD, T1, T2, T2*)。
        """
        if not isinstance(simplified_tensor, torch.Tensor):
            raise TypeError("输入 simplified_tensor 必须是一个Torch张量。")

        device = simplified_tensor.device

        # 直接使用在__init__中创建并移动到目标设备的Torch查找表
        lookup_tensor_on_device = self.lookup_tensor.to(device)
        tpm_tensor = lookup_tensor_on_device[simplified_tensor]

        # 返回分离后的四个属性图
        return tpm_tensor[..., 0], tpm_tensor[..., 1], tpm_tensor[..., 2], tpm_tensor[..., 3]
