# 网络构建配置示例
# 复制这些配置到 network.py 中的 NetworkConfig 类来修改网络构建方法

"""
示例1: 密度控制方法（推荐用于比较分析）
保证所有条件具有相同的网络密度，便于比较网络拓扑差异
"""
EXAMPLE_1_DENSITY = {
    'THRESHOLDING_METHOD': 'density',
    'NETWORK_DENSITY': 0.1,          # 保留10%最强连接
    'CORRELATION_SIGN': 'absolute',   # 使用绝对值
    'BINARIZE_NETWORK': True         # 二值化网络
}

"""
示例2: 绝对阈值方法
使用固定的相关系数阈值，适合已知合适阈值的情况
"""
EXAMPLE_2_ABSOLUTE = {
    'THRESHOLDING_METHOD': 'absolute',
    'CORRELATION_THRESHOLD': 0.3,    # 相关系数阈值
    'CORRELATION_SIGN': 'absolute',  # 使用绝对值
    'BINARIZE_NETWORK': True        # 二值化网络
}

"""
示例3: 仅显著性检验
保留所有统计显著的连接，不额外设置相关性阈值
"""
EXAMPLE_3_SIGNIFICANCE = {
    'THRESHOLDING_METHOD': 'significance_only',
    'CORRELATION_SIGN': 'absolute',  # 使用绝对值
    'BINARIZE_NETWORK': False       # 保留权重信息
}

"""
示例4: 正负相关分离分析
分别分析正相关和负相关网络
"""
EXAMPLE_4_POSITIVE = {
    'THRESHOLDING_METHOD': 'density',
    'NETWORK_DENSITY': 0.05,        # 密度稍低
    'CORRELATION_SIGN': 'positive', # 只保留正相关
    'BINARIZE_NETWORK': True
}

EXAMPLE_4_NEGATIVE = {
    'THRESHOLDING_METHOD': 'density',
    'NETWORK_DENSITY': 0.05,
    'CORRELATION_SIGN': 'negative', # 只保留负相关
    'BINARIZE_NETWORK': True
}

"""
示例5: 加权网络分析
保留连接权重信息，进行加权网络分析
"""
EXAMPLE_5_WEIGHTED = {
    'THRESHOLDING_METHOD': 'density',
    'NETWORK_DENSITY': 0.15,        # 稍高密度
    'CORRELATION_SIGN': 'absolute',
    'BINARIZE_NETWORK': False       # 保留权重
}

"""
使用方法:
1. 在 network.py 中找到 NetworkConfig 类
2. 复制想要的配置参数
3. 替换对应的类属性

例如，使用示例1的配置:
class NetworkConfig:
    THRESHOLDING_METHOD = 'density'
    NETWORK_DENSITY = 0.1
    CORRELATION_SIGN = 'absolute'
    BINARIZE_NETWORK = True
    # ... 其他参数保持不变
"""

# 密度值建议
DENSITY_RECOMMENDATIONS = {
    'sparse': 0.05,      # 5% - 稀疏网络，关注最强连接
    'moderate': 0.1,     # 10% - 中等密度，平衡连接数和强度
    'dense': 0.2,        # 20% - 较密集网络，包含更多弱连接
    'very_dense': 0.3    # 30% - 高密度网络，用于探索分析
}

# 相关系数阈值建议
CORRELATION_THRESHOLDS = {
    'strong': 0.5,       # 强相关
    'moderate': 0.3,     # 中等相关
    'weak': 0.2,         # 弱相关
    'very_weak': 0.1     # 极弱相关
}