import numpy as np


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


def create_tissue_property_lookup_table(property_set: str = 'ours', normalize: bool = False) -> tuple:
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
