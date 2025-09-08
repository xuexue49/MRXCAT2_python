import numpy as np
from typing import Tuple, Dict, Literal


class TissueProcessor:
    """
    一个用于处理XCAT体模组织属性的类。

    该类封装了将详细的XCAT组织标签简化为主要类别，并根据预定义的
    物理属性集为这些类别分配组织属性（如PD, T1, T2, T2*）的逻辑。

    使用方法:
        # 1. 初始化处理器，选择一个属性集 ('ours' 或 'xcat')
        processor = TissueProcessor(property_set='ours', normalize=False)

        # 2. 简化XCAT体模标签
        simplified_phantom = processor.simplify_xcat_labels(raw_xcat_phantom)

        # 3. 分配组织属性
        pd, t1, t2, t2s = processor.assign_tissue_properties(simplified_phantom)
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
        self.lookup_table, self.max_vals_dict = self._create_lookup_table()

    def _create_lookup_table(self) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        (私有方法) 创建并返回组织属性的查找表 (Lookup Table)。
        此方法在类的构造函数中被调用。

        Returns:
            tuple[np.ndarray, dict]:
                - 第一个元素是 (7, 4) 的Numpy数组，作为查找表。
                - 第二个元素是一个包含最大值的字典，用于可能的后续计算。
        """
        # 从 phantomModel.py 中提取的物理属性值
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

        selected_set = raw_values[self.property_set]
        lookup_table = selected_set['properties']
        max_vals_dict = selected_set['max_values']

        if self.normalize:
            max_vals_array = np.array(
                [max_vals_dict['pd'], max_vals_dict['t1'], max_vals_dict['t2'], max_vals_dict['t2s']])
            lookup_table = lookup_table / max_vals_array

        return lookup_table, max_vals_dict

    @staticmethod
    def simplify_xcat_labels(xcat_array: np.ndarray) -> np.ndarray:
        """
        (静态方法) 将XCAT体模中数十个详细的组织标签简化为7个主要类别。
        该函数逻辑完全基于 phantomModel.py 文件中的 Phantom 类。

        类别映射关系:
        - 0: Muscle (肌肉), 1: Blood (血液), 2: Air (空气), 3: Liver (肝脏),
        - 4: Fat (脂肪), 5: Bone (骨骼), 6: Unknown (未知/其他)

        Args:
            xcat_array (np.ndarray): 包含原始XCAT详细标签的Numpy数组。

        Returns:
            np.ndarray: 包含7个简化类别标签的Numpy数组。
        """
        print("开始简化XCAT标签...")
        if not isinstance(xcat_array, np.ndarray):
            raise TypeError("输入必须是一个Numpy数组。")

        # 创建一个查找表 (LUT)
        # XCAT标签的最大值是99，所以我们创建一个大小为100的LUT
        xcat_array = xcat_array.astype(int)
        max_label = 100
        lut = np.full(max_label + 1, 6, dtype=np.int32)  # 默认值为6 (Unknown)

        # 根据映射关系填充LUT
        lut[[1, 2, 3, 4, 10]] = 0  # Muscle
        lut[[5, 6, 7, 8]] = 1  # Blood
        lut[[0, 15, 16]] = 2  # Air
        lut[[13, 40, 41, 42, 43, 52]] = 3  # Liver
        lut[[50, 99]] = 4  # Fat
        lut[[31, 32, 33, 34, 35, 51]] = 5  # Bone

        # 使用LUT进行映射
        simplified_array = lut[xcat_array]

        print("标签简化完成。")
        return simplified_array

    def assign_tissue_properties(self, simplified_array: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        (实例方法) 使用预先计算好的查找表为简化体模分配组织属性。
        此函数依赖于实例初始化时创建的 `self.lookup_table`。

        Args:
            simplified_array (np.ndarray): 包含7个类别标签的数组 (值为 0-6)。

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                返回一个包含四个Numpy数组的元组 (PD, T1, T2, T2*)。
        """
        if not isinstance(simplified_array, np.ndarray):
            raise TypeError("输入 simplified_array 必须是一个Numpy数组。")

        # --- 核心操作: 使用高级索引进行高速映射 ---
        # 利用实例中存储的查找表
        tpm_array = self.lookup_table[simplified_array]

        # 返回分离后的四个属性图
        return tpm_array[..., 0], tpm_array[..., 1], tpm_array[..., 2], tpm_array[..., 3]
