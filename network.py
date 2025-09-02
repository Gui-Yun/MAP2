# 用于对神经数据构建功能连接网络并分析
# guiy24@mails.tsinghua.edu.cn
# date: 25-08-29

# %% 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
from scipy.stats import pearsonr, spearmanr
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
from collections import Counter
import os

# 从loaddata.py导入函数
from loaddata import (
    load_data, segment_neuron_data, reclassify_labels, 
    fast_rr_selection, cfg
)

# %% 配置
class NetworkConfig:
    """网络分析配置"""
    
    # 相关性分析参数
    CORRELATION_METHOD = 'pearson'  # 'pearson' or 'spearman'
    SIGNIFICANCE_THRESHOLD = 0.05   # 显著性阈值
    
    # 网络构建方法选择
    THRESHOLDING_METHOD = 'density'  # 'absolute', 'density', 'significance_only'
    
    # 阈值参数（根据THRESHOLDING_METHOD选择使用）
    CORRELATION_THRESHOLD = 0.3     # 绝对阈值方法使用
    NETWORK_DENSITY = 0.1          # 密度方法使用（保留最强的10%连接）
    
    # 全局阈值统一比较配置
    USE_GLOBAL_THRESHOLD = True     # 是否使用全局统一阈值进行比较
    
    # 稳健性测试参数
    ROBUSTNESS_TEST_DENSITIES = [0.05, 0.1, 0.15, 0.2]  # 多密度稳健性测试
    ROBUSTNESS_TEST_THRESHOLDS = [0.2, 0.3, 0.4, 0.5]   # 多绝对阈值稳健性测试
    
    # 相关值处理方法
    CORRELATION_SIGN = 'positive'   # 'absolute', 'positive', 'negative', 'both'
    
    # 网络类型
    BINARIZE_NETWORK = True         # 是否二值化网络（False保留权重）
    
    # 可视化参数
    FIGURE_SIZE = (12, 8)
    FIGURE_SIZE_LARGE = (15, 10)
    FIGURE_SIZE_EXTRA_LARGE = (18, 12)
    COLORMAP = 'RdYlBu_r'
    
    # 科研绘图配置
    VISUALIZATION_DPI = 300         # 图像分辨率
    VISUALIZATION_STYLE = 'seaborn-v0_8-whitegrid'  # 科研绘图风格
    
    # 简洁专业网络配色方案
    NETWORK_COLORS = {
        'nodes': '#2E86AB',         # 节点颜色（专业蓝）
        'edges': '#6C757D',         # 边颜色（中性灰）
        'hub_nodes': '#E74C3C',     # Hub节点颜色（红色）
        'neutral': '#95A5A6',       # 中性色（灰色）
        'communities': ['#2E86AB', '#E74C3C', '#27AE60', '#F39C12', '#9B59B6', '#6C757D'],
        'category1': '#2E86AB',     # 类别1颜色（蓝色）
        'category2': '#E74C3C',     # 类别2颜色（红色）
        'resting': '#6C757D',       # 静息状态颜色（灰色）
        'background': '#FFFFFF',    # 纯白背景
        'positive': '#2E86AB',      # 正相关（蓝色）
        'negative': '#E74C3C',      # 负相关（红色）
        'significant': '#27AE60'    # 显著性（绿色）
    }
    
    # 连接矩阵专业配色
    CONNECTIVITY_COLORMAP = 'RdBu_r'    # 相关性矩阵（红蓝色）
    NETWORK_COLORMAP = 'viridis'        # 网络邻接矩阵（紫绿色）
    
    # 网络指标专业配色
    METRICS_COLORS = {
        'primary': '#2E86AB',       # 主要指标（专业蓝）
        'secondary': '#A23B72',     # 次要指标（紫红）
        'accent': '#F18F01',        # 强调色（橙色）
        'success': '#27AE60',       # 成功/正向（绿色）
        'warning': '#F39C12',       # 警告（橙色）
        'danger': '#E74C3C',        # 危险/负向（红色）
        'neutral': '#6C757D',       # 中性（灰色）
        'light': '#ECF0F1',         # 浅色背景
        'dark': '#2C3E50'          # 深色文本
    }
    
    # 分析的类别
    TARGET_CATEGORIES = [1, 2]      # 只分析类别1和2
    
# 全局配置实例
net_cfg = NetworkConfig()

def print_network_config():
    """打印当前网络构建配置"""
    print("=" * 50)
    print("当前网络构建配置:")
    print("=" * 50)
    print(f"相关性方法: {net_cfg.CORRELATION_METHOD}")
    print(f"阈值化方法: {net_cfg.THRESHOLDING_METHOD}")
    
    if net_cfg.THRESHOLDING_METHOD == 'absolute':
        print(f"  - 相关系数阈值: {net_cfg.CORRELATION_THRESHOLD}")
    elif net_cfg.THRESHOLDING_METHOD == 'density':
        print(f"  - 网络密度: {net_cfg.NETWORK_DENSITY} (保留{net_cfg.NETWORK_DENSITY*100:.1f}%最强连接)")
    elif net_cfg.THRESHOLDING_METHOD == 'significance_only':
        print(f"  - 仅使用显著性检验")
    
    print(f"显著性阈值: {net_cfg.SIGNIFICANCE_THRESHOLD}")
    print(f"相关值处理: {net_cfg.CORRELATION_SIGN}")
    
    sign_descriptions = {
        'absolute': '使用绝对值（忽略正负）',
        'positive': '仅保留正相关',
        'negative': '仅保留负相关',
        'both': '保留正负相关符号'
    }
    print(f"  - {sign_descriptions.get(net_cfg.CORRELATION_SIGN, '未知')}")
    
    print(f"网络类型: {'二值化' if net_cfg.BINARIZE_NETWORK else '加权网络'}")
    print("=" * 50)

def get_method_recommendations():
    """提供方法学建议"""
    print("\n" + "=" * 60)
    print("网络构建方法学建议:")
    print("=" * 60)
    
    print("\n阈值化方法选择:")
    print("1. 'absolute' - 固定阈值")
    print("   优点: 简单直观，跨条件可比")
    print("   缺点: 不同条件可能产生差异很大的网络密度")
    print("   推荐: 探索性分析，已知合适阈值")
    
    print("\n2. 'density' - 密度控制（推荐）")
    print("   优点: 确保所有条件具有相同网络密度，便于比较")
    print("   缺点: 阈值在条件间变化")
    print("   推荐: 比较分析，标准化网络密度")
    
    print("\n3. 'significance_only' - 仅显著性")
    print("   优点: 统计严格，保留所有显著连接")
    print("   缺点: 密度变化大，可能包含弱连接")
    print("   推荐: 保守分析，关注统计显著性")
    
    print("\n相关值符号处理:")
    print("1. 'absolute' - 绝对值（推荐）")
    print("   适用: 关注连接强度，忽略方向性")
    
    print("2. 'positive' - 正相关")
    print("   适用: 协同激活网络")
    
    print("3. 'negative' - 负相关")
    print("   适用: 竞争抑制网络")
    
    print("4. 'both' - 保留符号")
    print("   适用: 需要区分正负相关的分析")
    
    print("\n推荐配置组合:")
    print("- 标准分析: density + absolute + binarized")
    print("- 权重分析: density + absolute + weighted")
    print("- 方向性分析: density + both + weighted")
    print("=" * 60)

# %% 函数

def filter_data_by_category(segments, labels, target_categories=[1, 2]):
    """
    过滤出指定类别的数据
    
    参数:
    segments: 神经数据片段 (trials, neurons, timepoints)
    labels: 标签数组
    target_categories: 目标类别列表
    
    返回:
    filtered_segments: 过滤后的数据
    filtered_labels: 过滤后的标签
    """
    print(f"过滤类别 {target_categories} 的数据...")
    
    # 创建掩码
    mask = np.isin(labels, target_categories)
    
    filtered_segments = segments[mask]
    filtered_labels = labels[mask]
    
    print(f"原始数据: {len(labels)} 个试次")
    print(f"过滤后数据: {len(filtered_labels)} 个试次")
    print(f"标签分布: {Counter(filtered_labels)}")
    
    return filtered_segments, filtered_labels

def extract_neural_activity(segments, use_stimulus_period=True):
    """
    提取神经活动用于相关性分析
    
    参数:
    segments: 神经数据片段 (trials, neurons, timepoints)
    use_stimulus_period: 是否只使用刺激期数据
    
    返回:
    neural_activity: (trials, neurons) 每个试次每个神经元的平均活动
    """
    print("提取神经活动...")
    
    if use_stimulus_period:
        # 使用刺激期数据
        stimulus_start = cfg.PRE_FRAMES
        stimulus_end = cfg.PRE_FRAMES + cfg.STIMULUS_DURATION
        time_window = np.arange(stimulus_start, min(stimulus_end, segments.shape[2]))
        segments_subset = segments[:, :, time_window]
        print(f"使用刺激期数据，时间窗口: {stimulus_start}-{stimulus_end}")
    else:
        segments_subset = segments
        print("使用完整时间序列数据")
    
    # 计算每个试次每个神经元的平均活动
    neural_activity = np.mean(segments_subset, axis=2)  # (trials, neurons)
    
    print(f"神经活动矩阵形状: {neural_activity.shape}")
    return neural_activity

def compute_correlation_matrix(neural_activity, method='pearson'):
    """
    计算神经元间的相关系数矩阵
    
    参数:
    neural_activity: (trials, neurons) 神经活动矩阵
    method: 相关系数方法 ('pearson' 或 'spearman')
    
    返回:
    corr_matrix: (neurons, neurons) 相关系数矩阵
    p_matrix: (neurons, neurons) p值矩阵
    """
    print(f"计算 {method} 相关系数矩阵...")
    
    n_neurons = neural_activity.shape[1]
    corr_matrix = np.zeros((n_neurons, n_neurons))
    p_matrix = np.zeros((n_neurons, n_neurons))
    
    # 计算所有神经元对的相关系数
    for i in range(n_neurons):
        for j in range(i, n_neurons):
            if i == j:
                corr_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
            else:
                if method == 'pearson':
                    corr, p_val = pearsonr(neural_activity[:, i], neural_activity[:, j])
                elif method == 'spearman':
                    corr, p_val = spearmanr(neural_activity[:, i], neural_activity[:, j])
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
                p_matrix[i, j] = p_val
                p_matrix[j, i] = p_val
        
        if (i + 1) % 100 == 0:
            print(f"已完成 {i + 1}/{n_neurons} 个神经元")
    
    print(f"相关系数矩阵计算完成，形状: {corr_matrix.shape}")
    print(f"相关系数范围: {corr_matrix.min():.3f} ~ {corr_matrix.max():.3f}")
    
    return corr_matrix, p_matrix

def threshold_correlation_matrix(corr_matrix, p_matrix, 
                               method=net_cfg.THRESHOLDING_METHOD,
                               corr_threshold=net_cfg.CORRELATION_THRESHOLD,
                               network_density=net_cfg.NETWORK_DENSITY,
                               p_threshold=net_cfg.SIGNIFICANCE_THRESHOLD,
                               correlation_sign=net_cfg.CORRELATION_SIGN,
                               binarize=net_cfg.BINARIZE_NETWORK):
    """
    对相关系数矩阵进行阈值化处理
    
    参数:
    corr_matrix: 相关系数矩阵
    p_matrix: p值矩阵
    method: 阈值化方法 ('absolute', 'density', 'significance_only')
    corr_threshold: 相关系数绝对阈值
    network_density: 网络密度（0-1之间）
    p_threshold: 显著性阈值
    correlation_sign: 相关值符号处理 ('absolute', 'positive', 'negative', 'both')
    binarize: 是否二值化
    
    返回:
    adj_matrix: 邻接矩阵
    """
    print(f"应用阈值化处理...")
    print(f"方法: {method}, 显著性阈值: {p_threshold}, 符号处理: {correlation_sign}")
    
    # 复制矩阵
    adj_matrix = corr_matrix.copy()
    n_nodes = adj_matrix.shape[0]
    
    # 移除自连接（设为0）
    np.fill_diagonal(adj_matrix, 0)
    
    # 步骤1: 应用显著性阈值
    significant_mask = p_matrix < p_threshold
    adj_matrix[~significant_mask] = 0
    print(f"显著性过滤后保留 {np.sum(significant_mask)} 个连接")
    
    # 步骤2: 处理相关系数符号
    if correlation_sign == 'absolute':
        adj_matrix = np.abs(adj_matrix)
        print("使用相关系数绝对值")
    elif correlation_sign == 'positive':
        adj_matrix[adj_matrix < 0] = 0
        print("只保留正相关")
    elif correlation_sign == 'negative':
        adj_matrix[adj_matrix > 0] = 0
        adj_matrix = np.abs(adj_matrix)  # 转为正值表示负相关强度
        print("只保留负相关（转为正值）")
    elif correlation_sign == 'both':
        # 保持原始符号，但后续处理需要注意
        print("保留正负相关符号")
    
    # 步骤3: 应用不同的阈值化方法
    if method == 'absolute':
        print(f"使用绝对阈值: {corr_threshold}")
        if correlation_sign == 'both':
            # 对于保留符号的情况，使用绝对值比较但保留符号
            threshold_mask = np.abs(adj_matrix) > corr_threshold
        else:
            threshold_mask = adj_matrix > corr_threshold
        adj_matrix[~threshold_mask] = 0
        
    elif method == 'density':
        print(f"使用密度阈值: {network_density} (保留最强的{network_density*100:.1f}%连接)")
        
        # 获取上三角部分的非零值（避免重复计算对称连接）
        triu_indices = np.triu_indices_from(adj_matrix, k=1)
        triu_values = adj_matrix[triu_indices]
        
        # 过滤掉零值
        nonzero_values = triu_values[triu_values != 0]
        
        if len(nonzero_values) > 0:
            # 根据符号处理方法确定排序标准
            if correlation_sign == 'both':
                # 保留符号时，按绝对值排序
                sort_values = np.abs(nonzero_values)
            else:
                sort_values = nonzero_values
            
            # 计算要保留的连接数
            n_keep = int(len(nonzero_values) * network_density)
            n_keep = max(1, n_keep)  # 至少保留1个连接
            
            # 找到阈值
            threshold = np.partition(sort_values, -n_keep)[-n_keep]
            print(f"密度阈值计算: 总共{len(nonzero_values)}个显著连接，保留{n_keep}个，阈值={threshold:.3f}")
            
            # 应用密度阈值
            if correlation_sign == 'both':
                density_mask = np.abs(adj_matrix) >= threshold
            else:
                density_mask = adj_matrix >= threshold
            adj_matrix[~density_mask] = 0
        else:
            print("警告: 没有显著连接，无法应用密度阈值")
            
    elif method == 'significance_only':
        print("仅使用显著性阈值，不应用额外的相关性阈值")
        # adj_matrix已经在步骤1中应用了显著性阈值
        
    else:
        raise ValueError(f"不支持的阈值化方法: {method}")
    
    # 步骤4: 二值化处理
    if binarize:
        adj_matrix = (adj_matrix != 0).astype(int)
        print("网络已二值化")
    else:
        print("保留权重信息")
    
    # 再次移除自连接（确保对角线为0）
    np.fill_diagonal(adj_matrix, 0)
    
    # 计算最终网络统计
    if binarize:
        n_connections = np.sum(adj_matrix > 0) // 2
    else:
        n_connections = np.sum(adj_matrix != 0) // 2
    
    total_possible = (n_nodes * (n_nodes - 1)) // 2
    final_density = n_connections / total_possible
    
    print(f"最终网络: {n_connections}/{total_possible} 连接 (密度: {final_density:.3f})")
    
    return adj_matrix

def calculate_global_thresholds_for_all_conditions(network_segments, stimulus_data, rr_neurons=None):
    """
    计算所有条件的全局统一阈值（多阈值稳健性测试）
    
    参数:
    network_segments: 神经数据片段
    stimulus_data: 刺激数据
    rr_neurons: RR神经元索引（可选）
    
    返回:
    global_thresholds: 不同密度/阈值对应的全局阈值字典
    """
    print("\n=== 计算全局统一阈值（稳健性测试）===")
    
    # 收集所有条件的相关系数
    all_correlations = []
    all_pvalues = []
    
    stimulus_categories = stimulus_data[:, 0]  # 类别
    stimulus_intensities = stimulus_data[:, 1]  # 强度
    unique_intensities = sorted(np.unique(stimulus_intensities))
    
    print(f"分析类别 {net_cfg.TARGET_CATEGORIES}，强度 {unique_intensities}")
    
    for category in net_cfg.TARGET_CATEGORIES:
        for intensity in unique_intensities:
            category_mask = stimulus_categories == category
            intensity_mask = stimulus_intensities == intensity
            condition_mask = category_mask & intensity_mask
            
            if np.sum(condition_mask) == 0:
                continue
            
            print(f"处理条件：类别{category}, 强度{intensity} ({np.sum(condition_mask)}个试次)")
            
            condition_segments = network_segments[condition_mask]
            neural_activity = extract_neural_activity(condition_segments, use_stimulus_period=True)
            corr_matrix, p_matrix = compute_correlation_matrix(
                neural_activity, method=net_cfg.CORRELATION_METHOD)
            
            # 获取上三角部分（避免重复）
            triu_indices = np.triu_indices_from(corr_matrix, k=1)
            condition_corrs = corr_matrix[triu_indices]
            condition_pvals = p_matrix[triu_indices]
            
            all_correlations.extend(condition_corrs)
            all_pvalues.extend(condition_pvals)
    
    # 转换为numpy数组
    all_correlations = np.array(all_correlations)
    all_pvalues = np.array(all_pvalues)
    
    print(f"总共收集了 {len(all_correlations)} 个连接")
    
    # 应用显著性过滤
    significant_mask = all_pvalues < net_cfg.SIGNIFICANCE_THRESHOLD
    significant_corrs = all_correlations[significant_mask]
    
    print(f"显著连接: {len(significant_corrs)} 个")
    
    if len(significant_corrs) == 0:
        print("警告: 没有显著连接")
        return {}
    
    # 根据符号处理方法处理相关系数
    if net_cfg.CORRELATION_SIGN == 'absolute':
        sort_values = np.abs(significant_corrs)
    elif net_cfg.CORRELATION_SIGN == 'positive':
        positive_corrs = significant_corrs[significant_corrs > 0]
        sort_values = positive_corrs if len(positive_corrs) > 0 else significant_corrs
    elif net_cfg.CORRELATION_SIGN == 'negative':
        negative_corrs = significant_corrs[significant_corrs < 0]
        sort_values = np.abs(negative_corrs) if len(negative_corrs) > 0 else np.abs(significant_corrs)
    elif net_cfg.CORRELATION_SIGN == 'both':
        sort_values = np.abs(significant_corrs)
    else:
        sort_values = np.abs(significant_corrs)
    
    # 计算多个密度对应的全局阈值
    global_thresholds = {}
    
    if net_cfg.THRESHOLDING_METHOD == 'density':
        densities = net_cfg.ROBUSTNESS_TEST_DENSITIES
        for density in densities:
            if len(sort_values) > 0:
                n_keep = int(len(sort_values) * density)
                n_keep = max(1, n_keep)
                threshold = np.partition(sort_values, -n_keep)[-n_keep] if n_keep <= len(sort_values) else sort_values.min()
                global_thresholds[f'density_{density}'] = threshold
                print(f"密度 {density}: 阈值 = {threshold:.4f} (保留{n_keep}/{len(sort_values)}个连接)")
    
    elif net_cfg.THRESHOLDING_METHOD == 'absolute':
        thresholds = net_cfg.ROBUSTNESS_TEST_THRESHOLDS
        for threshold in thresholds:
            global_thresholds[f'absolute_{threshold}'] = threshold
            n_keep = np.sum(sort_values >= threshold)
            print(f"绝对阈值 {threshold}: 保留{n_keep}/{len(sort_values)}个连接")
    
    print(f"全局阈值计算完成，共 {len(global_thresholds)} 个阈值条件")
    return global_thresholds

def apply_consistent_threshold(corr_matrix, p_matrix, global_threshold):
    """
    应用统一的全局阈值，确保所有条件具有一致的网络密度
    
    参数:
    corr_matrix: 相关系数矩阵
    p_matrix: p值矩阵
    global_threshold: 全局阈值
    
    返回:
    adj_matrix: 邻接矩阵
    """
    print(f"应用全局统一阈值: {global_threshold:.4f}")
    
    # 复制矩阵
    adj_matrix = corr_matrix.copy()
    n_nodes = adj_matrix.shape[0]
    
    # 移除自连接
    np.fill_diagonal(adj_matrix, 0)
    
    # 步骤1: 应用显著性阈值
    significant_mask = p_matrix < net_cfg.SIGNIFICANCE_THRESHOLD
    adj_matrix[~significant_mask] = 0
    
    # 步骤2: 处理相关系数符号
    if net_cfg.CORRELATION_SIGN == 'absolute':
        adj_matrix = np.abs(adj_matrix)
    elif net_cfg.CORRELATION_SIGN == 'positive':
        adj_matrix[adj_matrix < 0] = 0
    elif net_cfg.CORRELATION_SIGN == 'negative':
        adj_matrix[adj_matrix > 0] = 0
        adj_matrix = np.abs(adj_matrix)
    elif net_cfg.CORRELATION_SIGN == 'both':
        # 保持原始符号
        pass
    
    # 步骤3: 应用全局阈值
    if net_cfg.CORRELATION_SIGN == 'both':
        threshold_mask = np.abs(adj_matrix) >= global_threshold
    else:
        threshold_mask = adj_matrix >= global_threshold
    adj_matrix[~threshold_mask] = 0
    
    # 步骤4: 二值化处理
    if net_cfg.BINARIZE_NETWORK:
        adj_matrix = (adj_matrix != 0).astype(int)
    
    # 确保对角线为0
    np.fill_diagonal(adj_matrix, 0)
    
    # 计算最终网络统计
    if net_cfg.BINARIZE_NETWORK:
        n_connections = np.sum(adj_matrix > 0) // 2
    else:
        n_connections = np.sum(adj_matrix != 0) // 2
    
    total_possible = (n_nodes * (n_nodes - 1)) // 2
    final_density = n_connections / total_possible
    
    print(f"条件网络: {n_connections}/{total_possible} 连接 (密度: {final_density:.3f})")
    
    return adj_matrix

def compute_network_metrics(adj_matrix):
    """
    计算网络拓扑指标
    
    参数:
    adj_matrix: 邻接矩阵
    
    返回:
    metrics: 网络指标字典
    """
    print("计算网络拓扑指标...")
    
    # 创建networkx图
    G = nx.from_numpy_array(adj_matrix)
    
    # 基础指标
    metrics = {}
    
    # 节点数和边数
    metrics['n_nodes'] = G.number_of_nodes()
    metrics['n_edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    
    # 度相关指标
    degrees = dict(G.degree())
    degree_values = list(degrees.values())
    metrics['avg_degree'] = np.mean(degree_values)
    metrics['max_degree'] = np.max(degree_values)
    metrics['degree_std'] = np.std(degree_values)
    
    # 连通性指标
    if nx.is_connected(G):
        metrics['is_connected'] = True
        metrics['avg_path_length'] = nx.average_shortest_path_length(G)
        metrics['diameter'] = nx.diameter(G)
    else:
        metrics['is_connected'] = False
        # 计算最大连通分量的指标
        largest_cc = max(nx.connected_components(G), key=len)
        G_cc = G.subgraph(largest_cc)
        metrics['largest_cc_size'] = len(largest_cc)
        metrics['largest_cc_ratio'] = len(largest_cc) / G.number_of_nodes()
        if len(largest_cc) > 1:
            metrics['avg_path_length'] = nx.average_shortest_path_length(G_cc)
            metrics['diameter'] = nx.diameter(G_cc)
        else:
            metrics['avg_path_length'] = 0
            metrics['diameter'] = 0
    
    # 聚类系数
    metrics['avg_clustering'] = nx.average_clustering(G)
    
    # 模块化
    try:
        communities = nx.community.greedy_modularity_communities(G)
        metrics['n_communities'] = len(communities)
        metrics['modularity'] = nx.community.modularity(G, communities)
    except:
        metrics['n_communities'] = 0
        metrics['modularity'] = 0
    
    # 小世界性
    try:
        # 计算随机网络的聚类系数和路径长度
        n_random_nets = 10
        random_clustering = []
        random_path_length = []
        
        for _ in range(n_random_nets):
            G_random = nx.erdos_renyi_graph(G.number_of_nodes(), metrics['density'])
            if nx.is_connected(G_random):
                random_clustering.append(nx.average_clustering(G_random))
                random_path_length.append(nx.average_shortest_path_length(G_random))
        
        if random_clustering and random_path_length:
            avg_random_clustering = np.mean(random_clustering)
            avg_random_path_length = np.mean(random_path_length)
            
            if avg_random_clustering > 0 and avg_random_path_length > 0:
                metrics['small_world_sigma'] = (metrics['avg_clustering'] / avg_random_clustering) / (metrics['avg_path_length'] / avg_random_path_length)
            else:
                metrics['small_world_sigma'] = 0
        else:
            metrics['small_world_sigma'] = 0
    except:
        metrics['small_world_sigma'] = 0
    
    print("网络指标计算完成")
    return metrics

def save_correlation_matrix_stats(corr_matrix, title="Correlation Matrix", save_dir="results"):
    """
    保存相关系数矩阵的统计信息
    
    参数:
    corr_matrix: 相关系数矩阵
    title: 标题前缀
    save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 计算统计信息
    n_neurons = corr_matrix.shape[0]
    triu_indices = np.triu_indices(n_neurons, k=1)  # 上三角，排除对角线
    upper_triangle_values = corr_matrix[triu_indices]
    
    stats = {
        'matrix': corr_matrix,
        'n_neurons': n_neurons,
        'mean_correlation': np.mean(upper_triangle_values),
        'std_correlation': np.std(upper_triangle_values),
        'min_correlation': np.min(upper_triangle_values),
        'max_correlation': np.max(upper_triangle_values),
        'median_correlation': np.median(upper_triangle_values),
        'positive_correlations': np.sum(upper_triangle_values > 0),
        'negative_correlations': np.sum(upper_triangle_values < 0),
        'strong_correlations': np.sum(np.abs(upper_triangle_values) > 0.5),
        'weak_correlations': np.sum(np.abs(upper_triangle_values) < 0.1)
    }
    
    filename = title.replace(' ', '_').replace(':', '').lower() + '_stats.npz'
    np.savez_compressed(
        os.path.join(save_dir, filename),
        **stats
    )
    print(f"相关矩阵统计信息已保存: {filename}")

def save_network_topology(adj_matrix, neuron_pos=None, title="Network", save_dir="results"):
    """
    保存网络拓扑信息
    
    参数:
    adj_matrix: 邻接矩阵
    neuron_pos: 神经元位置坐标 (2, n_neurons)  
    title: 标题前缀
    save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建networkx图
    G = nx.from_numpy_array(adj_matrix)
    
    # 计算节点度数
    degrees = dict(G.degree())
    degree_values = list(degrees.values())
    
    # 保存拓扑信息
    topology = {
        'adjacency_matrix': adj_matrix,
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'density': nx.density(G),
        'degrees': degree_values,
        'avg_degree': np.mean(degree_values),
        'max_degree': np.max(degree_values),
        'min_degree': np.min(degree_values),
        'degree_std': np.std(degree_values)
    }
    
    # 添加位置信息
    if neuron_pos is not None and neuron_pos.shape[0] >= 2:
        topology['neuron_positions'] = neuron_pos
    
    filename = title.replace(' ', '_').replace(':', '').lower() + '_topology.npz'
    np.savez_compressed(
        os.path.join(save_dir, filename),
        **topology
    )
    print(f"网络拓扑信息已保存: {filename}")

def analyze_networks_with_multiple_thresholds(network_segments, stimulus_data, global_thresholds):
    """
    使用多个全局阈值进行网络分析（稳健性测试）
    
    参数:
    network_segments: 神经数据片段
    stimulus_data: 刺激数据
    global_thresholds: 全局阈值字典
    
    返回:
    multi_threshold_results: 多阈值分析结果
    """
    print(f"\n=== 多阈值稳健性分析 ===")
    
    stimulus_categories = stimulus_data[:, 0]
    stimulus_intensities = stimulus_data[:, 1]
    unique_intensities = sorted(np.unique(stimulus_intensities))
    
    multi_threshold_results = {}
    
    # 对每个阈值条件进行分析
    for threshold_name, threshold_value in global_thresholds.items():
        print(f"\n--- 分析阈值条件: {threshold_name} (阈值={threshold_value:.4f}) ---")
        
        condition_networks = {}
        condition_metrics = {}
        
        # 分析每个实验条件
        for category in net_cfg.TARGET_CATEGORIES:
            for intensity in unique_intensities:
                condition_key = f"Cat{category}_Int{intensity}"
                
                category_mask = stimulus_categories == category
                intensity_mask = stimulus_intensities == intensity
                condition_mask = category_mask & intensity_mask
                
                if np.sum(condition_mask) == 0:
                    continue
                
                condition_segments = network_segments[condition_mask]
                neural_activity = extract_neural_activity(condition_segments, use_stimulus_period=True)
                corr_matrix, p_matrix = compute_correlation_matrix(
                    neural_activity, method=net_cfg.CORRELATION_METHOD)
                
                # 使用统一的全局阈值
                adj_matrix = apply_consistent_threshold(corr_matrix, p_matrix, threshold_value)
                
                # 计算网络指标
                metrics = compute_network_metrics(adj_matrix)
                
                condition_networks[condition_key] = {
                    'correlation_matrix': corr_matrix,
                    'p_matrix': p_matrix,
                    'adjacency_matrix': adj_matrix,
                    'neural_activity': neural_activity
                }
                condition_metrics[condition_key] = metrics
                
                print(f"  {condition_key}: {metrics['n_edges']}边, 密度={metrics['density']:.3f}")
        
        # 添加静息状态分析
        baseline_segments = network_segments[:, :, :cfg.PRE_FRAMES]
        baseline_activity = np.mean(baseline_segments, axis=2)
        resting_corr_matrix, resting_p_matrix = compute_correlation_matrix(
            baseline_activity, method=net_cfg.CORRELATION_METHOD)
        resting_adj_matrix = apply_consistent_threshold(resting_corr_matrix, resting_p_matrix, threshold_value)
        resting_metrics = compute_network_metrics(resting_adj_matrix)
        
        condition_networks['Resting'] = {
            'correlation_matrix': resting_corr_matrix,
            'p_matrix': resting_p_matrix,
            'adjacency_matrix': resting_adj_matrix,
            'neural_activity': baseline_activity
        }
        condition_metrics['Resting'] = resting_metrics
        
        print(f"  Resting: {resting_metrics['n_edges']}边, 密度={resting_metrics['density']:.3f}")
        
        multi_threshold_results[threshold_name] = {
            'threshold_value': threshold_value,
            'condition_networks': condition_networks,
            'condition_metrics': condition_metrics
        }
    
    print(f"\n多阈值分析完成，共分析了 {len(global_thresholds)} 个阈值条件")
    return multi_threshold_results

def save_degree_distribution(adj_matrix, title="Degree Distribution", save_dir="results"):
    """
    保存度分布统计信息
    
    参数:
    adj_matrix: 邻接矩阵
    title: 标题前缀
    save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建networkx图并计算度
    G = nx.from_numpy_array(adj_matrix)
    degrees = [d for n, d in G.degree()]
    
    # 计算度分布统计
    degree_counts = Counter(degrees)
    degrees_unique = sorted(degree_counts.keys())
    counts = [degree_counts[d] for d in degrees_unique]
    
    distribution_stats = {
        'degrees': degrees,
        'unique_degrees': degrees_unique,
        'degree_counts': counts,
        'mean_degree': np.mean(degrees),
        'std_degree': np.std(degrees),
        'max_degree': max(degrees),
        'min_degree': min(degrees),
        'degree_distribution': dict(degree_counts)
    }
    
    filename = title.replace(' ', '_').replace(':', '').lower() + '_degree_dist.npz'
    np.savez_compressed(
        os.path.join(save_dir, filename),
        **distribution_stats
    )
    print(f"度分布统计已保存: {filename}")

# %% 可视化函数

def setup_network_plot_style():
    """设置网络分析科研绘图风格"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        # 字体设置
        'font.size': 12,
        'font.family': 'Arial',
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        
        # 轴和边框设置
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.5,
        'axes.edgecolor': '#2C3E50',
        'axes.axisbelow': True,
        
        # 网格和背景
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        
        # 图例和标记
        'legend.framealpha': 0.9,
        'legend.fancybox': True,
        'legend.shadow': True,
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        
        # 保存设置
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none'
    })

def visualize_correlation_matrix(corr_matrix, title="Functional Connectivity Matrix", save_path=None):
    """专业可视化功能连接矩阵"""
    setup_network_plot_style()
    
    # 计算统计信息
    triu_indices = np.triu_indices_from(corr_matrix, k=1)
    upper_values = corr_matrix[triu_indices]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 完整相关性矩阵
    im1 = ax1.imshow(corr_matrix, cmap=net_cfg.CONNECTIVITY_COLORMAP, 
                     vmin=-1, vmax=1, aspect='auto', interpolation='nearest')
    
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Neuron Index', fontweight='bold')
    ax1.set_ylabel('Neuron Index', fontweight='bold')
    
    # 专业颜色条
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.8, aspect=30)
    cbar1.set_label('Correlation Coefficient', rotation=270, labelpad=20, fontweight='bold')
    cbar1.ax.tick_params(labelsize=10)
    
    # 下三角遮罩矩阵
    corr_masked = corr_matrix.copy()
    mask = np.triu(np.ones_like(corr_matrix), k=1)
    corr_masked[mask.astype(bool)] = np.nan
    
    im2 = ax2.imshow(corr_masked, cmap=net_cfg.CONNECTIVITY_COLORMAP,
                     vmin=-1, vmax=1, aspect='auto', interpolation='nearest')
    
    ax2.set_title('Unique Connections', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Neuron Index', fontweight='bold')
    ax2.set_ylabel('Neuron Index', fontweight='bold')
    
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8, aspect=30)
    cbar2.set_label('Correlation Coefficient', rotation=270, labelpad=20, fontweight='bold')
    cbar2.ax.tick_params(labelsize=10)
    
    # 添加颜色映射说明
    ax1.text(0.02, 0.98, f'n = {len(upper_values)} connections', 
             transform=ax1.transAxes, fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'),
             verticalalignment='top')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def visualize_adjacency_matrix(adj_matrix, title="Network Adjacency Matrix", save_path=None):
    """专业可视化网络邻接矩阵"""
    setup_network_plot_style()
    
    # 计算网络基本统计
    n_nodes = adj_matrix.shape[0]
    n_edges = np.sum(adj_matrix > 0) // 2
    density = n_edges / ((n_nodes * (n_nodes - 1)) / 2) if n_nodes > 1 else 0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 完整邻接矩阵
    im1 = ax1.imshow(adj_matrix, cmap=net_cfg.NETWORK_COLORMAP, 
                     aspect='auto', interpolation='nearest')
    
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Neuron Index', fontweight='bold')
    ax1.set_ylabel('Neuron Index', fontweight='bold')
    
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.8, aspect=30)
    cbar1.set_label('Connection Strength', rotation=270, labelpad=20, fontweight='bold')
    cbar1.ax.tick_params(labelsize=10)
    
    # 上三角矩阵（唯一连接）
    adj_masked = adj_matrix.copy()
    mask = np.tril(np.ones_like(adj_matrix), k=-1)
    
    import numpy.ma as ma
    adj_masked_array = ma.masked_array(adj_masked, mask=mask.astype(bool))
    
    im2 = ax2.imshow(adj_masked_array, cmap=net_cfg.NETWORK_COLORMAP,
                     aspect='auto', interpolation='nearest')
    
    ax2.set_title('Unique Connections', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Neuron Index', fontweight='bold')
    ax2.set_ylabel('Neuron Index', fontweight='bold')
    
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8, aspect=30)
    cbar2.set_label('Connection Strength', rotation=270, labelpad=20, fontweight='bold')
    cbar2.ax.tick_params(labelsize=10)
    
    # 添加网络信息标签
    network_type = 'Binary' if np.max(adj_matrix) <= 1 and np.min(adj_matrix) >= 0 else 'Weighted'
    info_text = f'{network_type} Network\n{n_nodes} nodes, {n_edges} edges\nDensity: {density:.3f}'
    
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'),
             verticalalignment='top')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def visualize_connectivity_distribution(corr_matrix, adj_matrix, title="Connectivity Distribution", save_path=None):
    """专业可视化连接分布"""
    setup_network_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 获取上三角数据
    triu_indices = np.triu_indices_from(corr_matrix, k=1)
    correlations = corr_matrix[triu_indices]
    adjacency_weights = adj_matrix[triu_indices]
    non_zero_weights = adjacency_weights[adjacency_weights != 0]
    connected_mask = adjacency_weights != 0
    
    # 1. 相关系数分布
    ax1 = axes[0, 0]
    n, bins, patches = ax1.hist(correlations, bins=40, alpha=0.8, 
                               color=net_cfg.METRICS_COLORS['primary'],
                               edgecolor='white', linewidth=1.2)
    
    # 添加均值线
    mean_corr = np.mean(correlations)
    ax1.axvline(mean_corr, color=net_cfg.METRICS_COLORS['accent'], 
               linestyle='--', linewidth=3, alpha=0.8)
    ax1.axvline(0, color=net_cfg.METRICS_COLORS['dark'], linestyle='-', alpha=0.4)
    
    ax1.set_xlabel('Correlation Coefficient', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.set_title('Correlation Distribution', fontsize=12, fontweight='bold', pad=15)
    
    # 2. 连接权重分布  
    ax2 = axes[0, 1]
    if len(non_zero_weights) > 0:
        ax2.hist(non_zero_weights, bins=30, alpha=0.8, 
                color=net_cfg.METRICS_COLORS['secondary'],
                edgecolor='white', linewidth=1.2)
        
        mean_weight = np.mean(non_zero_weights)
        ax2.axvline(mean_weight, color=net_cfg.METRICS_COLORS['accent'],
                   linestyle='--', linewidth=3, alpha=0.8)
        
        ax2.set_xlabel('Connection Weight', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title(f'Weight Distribution (n={len(non_zero_weights)})', 
                     fontsize=12, fontweight='bold', pad=15)
    else:
        ax2.text(0.5, 0.5, 'No Connections', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=16, fontweight='bold',
                color=net_cfg.METRICS_COLORS['neutral'])
        ax2.set_title('Weight Distribution', fontsize=12, fontweight='bold', pad=15)
    
    # 3. 散点图：相关性vs权重
    ax3 = axes[1, 0]
    if np.sum(connected_mask) > 0:
        # 未连接的点
        ax3.scatter(correlations[~connected_mask], 
                   np.zeros(np.sum(~connected_mask)), 
                   alpha=0.4, s=15, color=net_cfg.METRICS_COLORS['neutral'],
                   edgecolors='none', label='Removed')
        
        # 保留的连接
        ax3.scatter(correlations[connected_mask], 
                   adjacency_weights[connected_mask],
                   alpha=0.7, s=25, color=net_cfg.METRICS_COLORS['primary'],
                   edgecolors='white', linewidth=0.5, label='Kept')
        
        ax3.set_xlabel('Original Correlation', fontweight='bold')
        ax3.set_ylabel('Final Weight', fontweight='bold')
        ax3.set_title('Thresholding Effect', fontsize=12, fontweight='bold', pad=15)
        ax3.legend(frameon=True, fancybox=True, shadow=True)
    else:
        ax3.text(0.5, 0.5, 'No Connections', ha='center', va='center',
                transform=ax3.transAxes, fontsize=16, fontweight='bold',
                color=net_cfg.METRICS_COLORS['neutral'])
        ax3.set_title('Thresholding Effect', fontsize=12, fontweight='bold', pad=15)
    
    # 4. 连接密度可视化
    ax4 = axes[1, 1]
    total_possible = len(correlations)
    n_connections = np.sum(connected_mask)
    
    # 饼图显示连接比例
    sizes = [n_connections, total_possible - n_connections]
    labels = ['Connected', 'Removed']
    colors = [net_cfg.METRICS_COLORS['success'], net_cfg.METRICS_COLORS['neutral']]
    
    wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors, 
                                      autopct='%1.1f%%', startangle=90,
                                      textprops={'fontsize': 10, 'fontweight': 'bold'})
    
    ax4.set_title('Connection Density', fontsize=12, fontweight='bold', pad=15)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.96)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def visualize_network_topology(adj_matrix, neuron_pos=None, title="Network Topology", save_path=None):
    """简洁专业的网络拓扑结构可视化"""
    setup_network_plot_style()
    
    G = nx.from_numpy_array(adj_matrix)
    
    # 使用更简洁的布局
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 计算网络指标
    degrees = dict(G.degree())
    degree_values = list(degrees.values())
    
    # 设置节点布局
    if neuron_pos is not None and neuron_pos.shape[0] >= 2:
        pos = {i: (neuron_pos[0, i], neuron_pos[1, i]) for i in range(neuron_pos.shape[1])}
    else:
        pos = nx.spring_layout(G, k=1.0, iterations=50, seed=42)
    
    # 1. 简洁的网络图
    ax1 = axes[0]
    if degree_values:
        # 简化的节点大小和颜色
        node_sizes = [max(30, deg * 20 + 50) for deg in degree_values]
        
        nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=node_sizes, 
                              node_color=net_cfg.NETWORK_COLORS['nodes'],
                              alpha=0.7, edgecolors='white', linewidths=1)
        nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.3, width=0.8, 
                              edge_color=net_cfg.NETWORK_COLORS['edges'])
        
    ax1.set_title('Network Structure', fontsize=12, fontweight='bold', pad=15)
    ax1.axis('off')
    
    # 2. 度分布直方图
    ax2 = axes[1]
    if degree_values:
        n, bins, patches = ax2.hist(degree_values, bins=min(15, len(set(degree_values))), 
                                   alpha=0.7, color=net_cfg.NETWORK_COLORS['nodes'],
                                   edgecolor='white', linewidth=1)
        
        ax2.set_xlabel('Degree', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title('Degree Distribution', fontsize=12, fontweight='bold', pad=15)
        
        # 简洁的网络统计
        mean_deg = np.mean(degree_values)
        ax2.axvline(mean_deg, color=net_cfg.NETWORK_COLORS['hub_nodes'], 
                   linestyle='--', linewidth=2, alpha=0.8, label=f'Mean: {mean_deg:.1f}')
        ax2.legend()
        
    else:
        ax2.text(0.5, 0.5, 'No Data', ha='center', va='center',
                transform=ax2.transAxes, fontsize=14, fontweight='bold',
                color=net_cfg.NETWORK_COLORS['neutral'])
        ax2.set_title('Degree Distribution', fontsize=12, fontweight='bold', pad=15)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def visualize_network_metrics_comparison(category_metrics, title="Network Metrics Comparison", save_path=None):
    """简洁的网络指标对比可视化"""
    setup_network_plot_style()
    
    if len(category_metrics) < 2:
        print("需要至少2个条件进行比较")
        return
    
    metrics_to_plot = ['density', 'avg_clustering', 'avg_degree']
    categories = list(category_metrics.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 简洁的配色方案
    colors = [net_cfg.NETWORK_COLORS['category1'] if i % 2 == 0 
              else net_cfg.NETWORK_COLORS['category2'] for i in range(len(categories))]
    
    # 绘制各项指标对比
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        values = [category_metrics[cat][metric] for cat in categories]
        
        bars = ax.bar(range(len(categories)), values, 
                     color=colors, alpha=0.7, 
                     edgecolor='white', linewidth=1.5)
        
        ax.set_xlabel('Condition', fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold')
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha='right')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def visualize_degree_distribution(adj_matrix, title="Degree Distribution", save_path=None):
    """简洁的度分布分析可视化"""
    setup_network_plot_style()
    
    G = nx.from_numpy_array(adj_matrix)
    degrees = np.array([d for n, d in G.degree()])
    
    if len(degrees) == 0 or np.max(degrees) == 0:
        print("No network connections to analyze")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # 1. 度分布直方图
    n_bins = min(15, len(np.unique(degrees)))
    ax1.hist(degrees, bins=n_bins, alpha=0.7, 
            color=net_cfg.NETWORK_COLORS['nodes'],
            edgecolor='white', linewidth=1)
    
    # 统计线
    mean_degree = np.mean(degrees)
    ax1.axvline(mean_degree, color=net_cfg.NETWORK_COLORS['hub_nodes'], 
               linestyle='--', linewidth=2, alpha=0.8, 
               label=f'Mean: {mean_degree:.1f}')
    
    ax1.set_xlabel('Degree', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.set_title('Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    
    # 2. 基本统计
    ax2.axis('off')
    stats_text = (f'Nodes: {len(degrees)}\n'
                 f'Edges: {G.number_of_edges()}\n'
                 f'Density: {nx.density(G):.3f}\n'
                 f'Mean Degree: {mean_degree:.2f}\n'
                 f'Max Degree: {np.max(degrees)}\n'
                 f'Std: {np.std(degrees):.2f}')
    
    ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes, 
            fontsize=12, fontweight='bold', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    ax2.set_title('Network Statistics', fontsize=12, fontweight='bold')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def visualize_robustness_analysis(multi_threshold_results, save_path=None):
    """
    可视化多阈值稳健性分析结果
    """
    setup_network_plot_style()
    
    print("生成稳健性分析可视化...")
    
    # 提取所有阈值条件的关键指标
    threshold_names = list(multi_threshold_results.keys())
    metrics_to_plot = ['density', 'avg_clustering', 'modularity', 'avg_degree']
    
    # 收集所有条件的数据
    all_condition_keys = set()
    for threshold_data in multi_threshold_results.values():
        all_condition_keys.update(threshold_data['condition_metrics'].keys())
    all_condition_keys = sorted(list(all_condition_keys))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(threshold_names)))
    
    for metric_idx, metric in enumerate(metrics_to_plot):
        ax = axes[metric_idx]
        
        # 为每个阈值条件绘制条形图
        x_positions = np.arange(len(all_condition_keys))
        bar_width = 0.8 / len(threshold_names)
        
        for thresh_idx, threshold_name in enumerate(threshold_names):
            threshold_data = multi_threshold_results[threshold_name]
            condition_metrics = threshold_data['condition_metrics']
            
            values = []
            for condition_key in all_condition_keys:
                if condition_key in condition_metrics:
                    values.append(condition_metrics[condition_key][metric])
                else:
                    values.append(0)
            
            x_pos = x_positions + thresh_idx * bar_width - (len(threshold_names) - 1) * bar_width / 2
            bars = ax.bar(x_pos, values, bar_width, 
                         color=colors[thresh_idx], alpha=0.7,
                         label=threshold_name, edgecolor='white', linewidth=1)
        
        ax.set_xlabel('Experimental Conditions', fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ").title()} - Robustness Test', 
                    fontsize=12, fontweight='bold', pad=15)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(all_condition_keys, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Multi-Threshold Robustness Analysis', fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def generate_robustness_report(multi_threshold_results):
    """
    生成多阈值稳健性分析报告
    """
    report = []
    report.append("=" * 60)
    report.append("多阈值稳健性分析报告")
    report.append("=" * 60)
    report.append("")
    
    # 阈值条件总览
    threshold_names = list(multi_threshold_results.keys())
    report.append(f"分析的阈值条件数量: {len(threshold_names)}")
    report.append("")
    
    for threshold_name in threshold_names:
        threshold_data = multi_threshold_results[threshold_name]
        threshold_value = threshold_data['threshold_value']
        report.append(f"- {threshold_name}: 阈值 = {threshold_value:.4f}")
    
    report.append("")
    report.append("=" * 40)
    report.append("各阈值条件下的网络指标")
    report.append("=" * 40)
    
    # 关键指标比较
    key_metrics = ['density', 'avg_clustering', 'modularity', 'avg_degree']
    
    for metric in key_metrics:
        report.append(f"\n## {metric.replace('_', ' ').title()}")
        report.append("-" * 30)
        
        # 收集所有条件的该指标数据
        condition_keys = set()
        for threshold_data in multi_threshold_results.values():
            condition_keys.update(threshold_data['condition_metrics'].keys())
        condition_keys = sorted(list(condition_keys))
        
        # 表格表头
        header = f"{'Condition':<15}"
        for threshold_name in threshold_names:
            header += f"{threshold_name:<12}"
        header += "CV"  # 变异系数
        report.append(header)
        report.append("-" * len(header))
        
        # 每个条件的数据
        for condition_key in condition_keys:
            row = f"{condition_key:<15}"
            values = []
            
            for threshold_name in threshold_names:
                threshold_data = multi_threshold_results[threshold_name]
                if condition_key in threshold_data['condition_metrics']:
                    value = threshold_data['condition_metrics'][condition_key][metric]
                    row += f"{value:<12.3f}"
                    values.append(value)
                else:
                    row += f"{'N/A':<12}"
            
            # 计算变异系数（CV = std/mean）
            if len(values) > 1 and np.mean(values) != 0:
                cv = np.std(values) / np.mean(values)
                row += f"{cv:.3f}"
            else:
                row += "N/A"
            
            report.append(row)
        report.append("")
    
    # 稳健性总结
    report.append("=" * 40)
    report.append("稳健性总结")
    report.append("=" * 40)
    report.append("")
    
    # 计算平均变异系数
    total_cvs = []
    for metric in key_metrics:
        metric_cvs = []
        condition_keys = set()
        for threshold_data in multi_threshold_results.values():
            condition_keys.update(threshold_data['condition_metrics'].keys())
        
        for condition_key in condition_keys:
            values = []
            for threshold_data in multi_threshold_results.values():
                if condition_key in threshold_data['condition_metrics']:
                    values.append(threshold_data['condition_metrics'][condition_key][metric])
            
            if len(values) > 1 and np.mean(values) != 0:
                cv = np.std(values) / np.mean(values)
                metric_cvs.append(cv)
        
        if metric_cvs:
            avg_cv = np.mean(metric_cvs)
            report.append(f"{metric.replace('_', ' ').title()} 平均CV: {avg_cv:.3f}")
            total_cvs.extend(metric_cvs)
    
    if total_cvs:
        overall_cv = np.mean(total_cvs)
        report.append(f"\n整体平均CV: {overall_cv:.3f}")
        
        if overall_cv < 0.1:
            stability = "非常稳健"
        elif overall_cv < 0.2:
            stability = "稳健"
        elif overall_cv < 0.5:
            stability = "中等稳健"
        else:
            stability = "不稳健"
        
        report.append(f"稳健性评级: {stability}")
        report.append("")
        report.append("说明:")
        report.append("- CV < 0.1: 非常稳健，结果几乎不受阈值选择影响")
        report.append("- CV < 0.2: 稳健，结果在不同阈值下基本一致")
        report.append("- CV < 0.5: 中等稳健，存在一定阈值依赖性")
        report.append("- CV >= 0.5: 不稳健，结果强烈依赖于阈值选择")
    
    return "\n".join(report)

def visualize_intensity_effects(condition_metrics, target_category=2, title="Stimulus Intensity Effects", save_path=None):
    """简洁的刺激强度效应可视化"""
    setup_network_plot_style()
    
    # 筛选目标类别的强度条件
    cat_conditions = [k for k in condition_metrics.keys() if f'Cat{target_category}' in k and 'Int' in k]
    
    if len(cat_conditions) < 2:
        print(f"类别 {target_category} 的强度条件不足")
        return
    
    # 提取强度值和指标数据
    intensities = []
    metrics_data = {}
    metrics_to_plot = ['density', 'avg_clustering', 'avg_degree']
    
    for cond in cat_conditions:
        intensity_str = cond.split('_Int')[1]
        intensity = float(intensity_str)
        intensities.append(intensity)
        
        for metric in metrics_to_plot:
            if metric not in metrics_data:
                metrics_data[metric] = []
            metrics_data[metric].append(condition_metrics[cond][metric])
    
    # 按强度排序
    sorted_data = sorted(zip(intensities, *[metrics_data[m] for m in metrics_to_plot]))
    intensities = [d[0] for d in sorted_data]
    for i, metric in enumerate(metrics_to_plot):
        metrics_data[metric] = [d[i+1] for d in sorted_data]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 简洁配色
    colors = [net_cfg.NETWORK_COLORS['nodes'], net_cfg.NETWORK_COLORS['hub_nodes'], 
              net_cfg.NETWORK_COLORS['neutral']]
    
    # 各指标趋势图
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        ax.plot(intensities, metrics_data[metric], 'o-', 
               linewidth=2, markersize=5, color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Stimulus Intensity', fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold')
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# %% 主脚本
if __name__ == "__main__":
    print("开始构建网络")
    warnings.filterwarnings('ignore')
    
    # 显示当前配置
    print_network_config()
    get_method_recommendations()
    
    # %% 提取数据 + 预处理
    print("\n=== 数据加载与预处理 ===")
    
    # 加载数据
    neuron_data, neuron_pos, trigger_data, stimulus_data = load_data(cfg.DATA_PATH)
    
    # 数据分割
    segments, labels = segment_neuron_data(neuron_data, trigger_data, stimulus_data)
    
    # 重新分类标签
    reclassified_labels = reclassify_labels(stimulus_data)
    
    # RR神经元筛选
    print("\n筛选RR神经元...")
    rr_results = fast_rr_selection(segments, reclassified_labels)
    
    if len(rr_results['rr_neurons']) > 0:
        print(f"使用 {len(rr_results['rr_neurons'])} 个RR神经元")
        network_segments = segments[:, rr_results['rr_neurons'], :]
        network_neuron_pos = neuron_pos[:, rr_results['rr_neurons']] if neuron_pos.shape[1] > 0 else None
    else:
        print("未找到RR神经元，使用所有神经元")
        network_segments = segments
        network_neuron_pos = neuron_pos
    
    # 过滤类别1和2的数据
    filtered_segments, filtered_labels = filter_data_by_category(
        network_segments, stimulus_data[:, 0], net_cfg.TARGET_CATEGORIES)
    
    # %% 全局阈值计算和多阈值稳健性分析
    print("\n=== 全局阈值计算和多阈值稳健性分析 ===")
    
    if net_cfg.USE_GLOBAL_THRESHOLD:
        # 计算全局统一阈值
        global_thresholds = calculate_global_thresholds_for_all_conditions(
            network_segments, stimulus_data)
        
        if global_thresholds:
            # 进行多阈值稳健性分析
            multi_threshold_results = analyze_networks_with_multiple_thresholds(
                network_segments, stimulus_data, global_thresholds)
            
            # 选择主要阈值进行后续分析（通常是默认密度）
            if net_cfg.THRESHOLDING_METHOD == 'density':
                main_threshold_key = f'density_{net_cfg.NETWORK_DENSITY}'
            else:
                main_threshold_key = f'absolute_{net_cfg.CORRELATION_THRESHOLD}'
            
            if main_threshold_key in multi_threshold_results:
                main_results = multi_threshold_results[main_threshold_key]
                category_networks = {}
                category_metrics = {}
                
                # 提取类别级别的结果
                for category in net_cfg.TARGET_CATEGORIES:
                    # 合并该类别所有强度条件的结果作为代表
                    category_conditions = [k for k in main_results['condition_metrics'].keys() 
                                         if f'Cat{category}' in k]
                    if category_conditions:
                        # 使用第一个条件作为代表（或者可以选择平均）
                        representative_condition = category_conditions[0]
                        category_networks[category] = main_results['condition_networks'][representative_condition]
                        category_metrics[category] = main_results['condition_metrics'][representative_condition]
            else:
                print("警告: 主要阈值条件未找到，使用传统方法")
                category_networks = {}
                category_metrics = {}
        else:
            print("警告: 全局阈值计算失败，使用传统方法")
            net_cfg.USE_GLOBAL_THRESHOLD = False
            category_networks = {}
            category_metrics = {}
    else:
        multi_threshold_results = None
        category_networks = {}
        category_metrics = {}
    
    # 如果未使用全局阈值，则使用原始方法
    if not net_cfg.USE_GLOBAL_THRESHOLD or not category_networks:
        print("\n=== 使用传统方法构建功能连接矩阵 ===")
        for category in net_cfg.TARGET_CATEGORIES:
            print(f"\n--- 处理类别 {category} ---")
            
            # 提取该类别的数据
            category_mask = filtered_labels == category
            category_segments = filtered_segments[category_mask]
            
            if len(category_segments) == 0:
                print(f"类别 {category} 无数据，跳过")
                continue
            
            print(f"类别 {category} 数据: {len(category_segments)} 个试次")
            
            # 提取神经活动
            neural_activity = extract_neural_activity(category_segments, use_stimulus_period=True)
            
            # 计算相关系数矩阵
            corr_matrix, p_matrix = compute_correlation_matrix(
                neural_activity, method=net_cfg.CORRELATION_METHOD)
            
            # 阈值化处理
            adj_matrix = threshold_correlation_matrix(corr_matrix, p_matrix)
            
            # 计算网络指标
            metrics = compute_network_metrics(adj_matrix)
            
            # 保存结果
            category_networks[category] = {
                'correlation_matrix': corr_matrix,
                'p_matrix': p_matrix,
                'adjacency_matrix': adj_matrix,
                'neural_activity': neural_activity
            }
            category_metrics[category] = metrics
            
            # 打印关键指标
            print(f"类别 {category} 网络指标:")
            print(f"  节点数: {metrics['n_nodes']}")
            print(f"  连接数: {metrics['n_edges']}")
            print(f"  密度: {metrics['density']:.3f}")
            print(f"  平均度: {metrics['avg_degree']:.2f}")
            print(f"  平均聚类系数: {metrics['avg_clustering']:.3f}")
            print(f"  模块数: {metrics['n_communities']}")
            print(f"  模块化: {metrics['modularity']:.3f}")
            if metrics['is_connected']:
                print(f"  平均路径长度: {metrics['avg_path_length']:.3f}")
            print(f"  小世界系数: {metrics['small_world_sigma']:.3f}")
    
    # %% 创建保存目录
    results_dir = os.path.join(cfg.DATA_PATH, 'network_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # %% 多阈值稳健性分析可视化
    if net_cfg.USE_GLOBAL_THRESHOLD and 'multi_threshold_results' in locals() and multi_threshold_results:
        print("\n=== 多阈值稳健性分析可视化 ===")
        
        # 可视化稳健性分析
        visualize_robustness_analysis(
            multi_threshold_results,
            save_path=os.path.join(results_dir, 'multi_threshold_robustness_analysis.png')
        )
        
        # 保存稳健性分析数据
        robustness_save_path = os.path.join(results_dir, 'multi_threshold_robustness_data.npz')
        robustness_data = {}
        
        for threshold_name, threshold_data in multi_threshold_results.items():
            robustness_data[f'{threshold_name}_threshold_value'] = threshold_data['threshold_value']
            for condition_key, metrics in threshold_data['condition_metrics'].items():
                for metric_name, metric_value in metrics.items():
                    robustness_data[f'{threshold_name}_{condition_key}_{metric_name}'] = metric_value
        
        np.savez_compressed(robustness_save_path, **robustness_data)
        print(f"稳健性分析数据已保存: {robustness_save_path}")
        
        # 生成稳健性分析报告
        robustness_report = generate_robustness_report(multi_threshold_results)
        report_path = os.path.join(results_dir, 'robustness_analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(robustness_report)
        print(f"稳健性分析报告已保存: {report_path}")
    
    # %% 网络可视化
    print("\n=== 网络可视化 ===")
    
    for category in category_networks.keys():
        print(f"\n可视化类别 {category} 的网络...")
        
        # 可视化相关性矩阵
        visualize_correlation_matrix(
            category_networks[category]['correlation_matrix'],
            title=f"Category {category} Functional Connectivity Matrix",
            save_path=os.path.join(results_dir, f'category_{category}_correlation_matrix.png')
        )
        
        # 可视化邻接矩阵
        visualize_adjacency_matrix(
            category_networks[category]['adjacency_matrix'],
            title=f"Category {category} Network Adjacency Matrix",
            save_path=os.path.join(results_dir, f'category_{category}_adjacency_matrix.png')
        )
        
        # 可视化连接分布
        visualize_connectivity_distribution(
            category_networks[category]['correlation_matrix'],
            category_networks[category]['adjacency_matrix'],
            title=f"Category {category} Connectivity Distribution",
            save_path=os.path.join(results_dir, f'category_{category}_connectivity_distribution.png')
        )
        
        # 可视化网络拓扑
        visualize_network_topology(
            category_networks[category]['adjacency_matrix'],
            neuron_pos=network_neuron_pos,
            title=f"Category {category} Network Topology",
            save_path=os.path.join(results_dir, f'category_{category}_network_topology.png')
        )
        
        # 可视化度分布
        visualize_degree_distribution(
            category_networks[category]['adjacency_matrix'],
            title=f"Category {category} Degree Distribution Analysis",
            save_path=os.path.join(results_dir, f'category_{category}_degree_distribution.png')
        )
        
        # 保存相关矩阵统计信息
        save_correlation_matrix_stats(
            category_networks[category]['correlation_matrix'],
            title=f"Category {category} Correlation Matrix",
            save_dir=results_dir
        )
        
        # 保存网络拓扑信息
        save_network_topology(
            category_networks[category]['adjacency_matrix'],
            neuron_pos=network_neuron_pos,
            title=f"Category {category} Network",
            save_dir=results_dir
        )
        
        # 保存度分布统计
        save_degree_distribution(
            category_networks[category]['adjacency_matrix'],
            title=f"Category {category} Degree Distribution",
            save_dir=results_dir
        )
    
    # %% 比较不同类别的网络特征
    print("\n=== 网络特征比较 ===")
    
    if len(category_metrics) >= 2:
        # 可视化网络指标比较
        visualize_network_metrics_comparison(
            category_metrics,
            title="Network Metrics Comparison Across Categories",
            save_path=os.path.join(results_dir, 'network_metrics_comparison.png')
        )
        
        # 创建比较表格
        comparison_df = pd.DataFrame(category_metrics).T
        print("\n网络指标比较:")
        print(comparison_df.round(3))
        
        # 保存比较结果
        comparison_df.to_csv(os.path.join(results_dir, 'network_metrics_comparison.csv'))
        print(f"\n网络指标比较已保存到: {os.path.join(results_dir, 'network_metrics_comparison.csv')}")
        
        # 保存关键指标比较数据
        key_metrics = ['density', 'avg_clustering', 'modularity', 'avg_degree']
        categories = list(category_metrics.keys())
        
        comparison_data = {}
        for metric in key_metrics:
            comparison_data[f'{metric}_values'] = [category_metrics[cat][metric] for cat in categories]
            comparison_data[f'{metric}_categories'] = categories
        
        np.savez_compressed(
            os.path.join(results_dir, 'network_metrics_comparison.npz'),
            **comparison_data
        )
        print("网络指标比较数据已保存")
    
    # %% 按类别和强度的详细分析
    print("\n=== 按类别和强度的详细网络分析 ===")
    
    # 按类别和强度构建网络
    condition_networks = {}
    condition_metrics = {}
    
    # 获取完整的刺激数据用于强度分析
    stimulus_categories = stimulus_data[:, 0]  # 类别
    stimulus_intensities = stimulus_data[:, 1]  # 强度
    
    # 获取所有唯一的强度值
    unique_intensities = sorted(np.unique(stimulus_intensities))
    print(f"发现的强度值: {unique_intensities}")
    
    # 步骤1: 如果使用密度方法，先计算全局阈值以确保一致性
    global_threshold = None
    if net_cfg.THRESHOLDING_METHOD == 'density':
        print(f"\n计算全局密度阈值以确保所有条件具有一致密度...")
        
        # 收集所有条件的相关系数用于计算全局阈值
        all_correlations = []
        all_pvalues = []
        
        for category in [1, 2]:
            for intensity in unique_intensities:
                category_mask = stimulus_categories == category
                intensity_mask = stimulus_intensities == intensity
                condition_mask = category_mask & intensity_mask
                
                if np.sum(condition_mask) == 0:
                    continue
                
                condition_segments = network_segments[condition_mask]
                neural_activity = extract_neural_activity(condition_segments, use_stimulus_period=True)
                corr_matrix, p_matrix = compute_correlation_matrix(
                    neural_activity, method=net_cfg.CORRELATION_METHOD)
                
                # 获取上三角部分（避免重复）
                triu_indices = np.triu_indices_from(corr_matrix, k=1)
                condition_corrs = corr_matrix[triu_indices]
                condition_pvals = p_matrix[triu_indices]
                
                all_correlations.extend(condition_corrs)
                all_pvalues.extend(condition_pvals)
        
        # 转换为numpy数组
        all_correlations = np.array(all_correlations)
        all_pvalues = np.array(all_pvalues)
        
        # 应用显著性过滤
        significant_mask = all_pvalues < net_cfg.SIGNIFICANCE_THRESHOLD
        significant_corrs = all_correlations[significant_mask]
        
        if len(significant_corrs) > 0:
            # 根据符号处理方法处理相关系数
            if net_cfg.CORRELATION_SIGN == 'absolute':
                sort_values = np.abs(significant_corrs)
            elif net_cfg.CORRELATION_SIGN == 'positive':
                positive_corrs = significant_corrs[significant_corrs > 0]
                sort_values = positive_corrs
            elif net_cfg.CORRELATION_SIGN == 'negative':
                negative_corrs = significant_corrs[significant_corrs < 0]
                sort_values = np.abs(negative_corrs)
            elif net_cfg.CORRELATION_SIGN == 'both':
                sort_values = np.abs(significant_corrs)
            
            # 计算全局密度阈值
            if len(sort_values) > 0:
                n_keep = int(len(sort_values) * net_cfg.NETWORK_DENSITY)
                n_keep = max(1, n_keep)
                global_threshold = np.partition(sort_values, -n_keep)[-n_keep]
                print(f"全局密度阈值: {global_threshold:.4f} (基于{len(sort_values)}个显著连接)")
            else:
                print("警告: 没有符合条件的显著连接")
        else:
            print("警告: 没有显著连接")
    
    # 步骤2: 使用统一阈值分析所有条件
    for category in [1, 2]:
        for intensity in unique_intensities:
            condition_key = f"Cat{category}_Int{intensity}"
            print(f"\n--- 分析条件: 类别{category}, 强度{intensity} ---")
            
            # 创建条件掩码
            category_mask = stimulus_categories == category
            intensity_mask = stimulus_intensities == intensity
            condition_mask = category_mask & intensity_mask
            
            if np.sum(condition_mask) == 0:
                print(f"条件 {condition_key} 无数据，跳过")
                continue
            
            condition_segments = network_segments[condition_mask]
            print(f"条件 {condition_key} 数据: {len(condition_segments)} 个试次")
            
            # 提取神经活动（刺激期）
            neural_activity = extract_neural_activity(condition_segments, use_stimulus_period=True)
            
            # 计算相关系数矩阵
            corr_matrix, p_matrix = compute_correlation_matrix(
                neural_activity, method=net_cfg.CORRELATION_METHOD)
            
            # 使用统一阈值进行阈值化处理
            if net_cfg.THRESHOLDING_METHOD == 'density' and global_threshold is not None:
                # 使用全局密度阈值以确保一致性
                adj_matrix = apply_consistent_threshold(corr_matrix, p_matrix, global_threshold)
            else:
                # 使用默认方法
                adj_matrix = threshold_correlation_matrix(corr_matrix, p_matrix)
            
            # 计算网络指标
            metrics = compute_network_metrics(adj_matrix)
            
            # 保存结果
            condition_networks[condition_key] = {
                'correlation_matrix': corr_matrix,
                'p_matrix': p_matrix,
                'adjacency_matrix': adj_matrix,
                'neural_activity': neural_activity,
                'category': category,
                'intensity': intensity
            }
            condition_metrics[condition_key] = metrics
            
            # 打印关键指标
            print(f"条件 {condition_key} 网络指标:")
            print(f"  节点数: {metrics['n_nodes']}")
            print(f"  连接数: {metrics['n_edges']}")
            print(f"  密度: {metrics['density']:.3f}")
            print(f"  平均度: {metrics['avg_degree']:.2f}")
            print(f"  平均聚类系数: {metrics['avg_clustering']:.3f}")
            print(f"  模块化: {metrics['modularity']:.3f}")
    
    # %% 静息状态网络分析 (基线期)
    print("\n=== 静息状态网络分析 ===")
    
    print("分析基线期神经活动...")
    # 提取基线期活动（所有试次的基线期合并）
    baseline_activity = extract_neural_activity(network_segments, use_stimulus_period=False)
    
    # 只取基线期部分
    baseline_segments = network_segments[:, :, :cfg.PRE_FRAMES]  # 基线期数据
    baseline_mean_activity = np.mean(baseline_segments, axis=2)  # (trials, neurons)
    
    print(f"基线期数据: {len(baseline_mean_activity)} 个试次")
    
    # 计算静息状态相关矩阵
    resting_corr_matrix, resting_p_matrix = compute_correlation_matrix(
        baseline_mean_activity, method=net_cfg.CORRELATION_METHOD)
    
    # 阈值化处理
    resting_adj_matrix = threshold_correlation_matrix(resting_corr_matrix, resting_p_matrix)
    
    # 计算静息网络指标
    resting_metrics = compute_network_metrics(resting_adj_matrix)
    
    # 保存静息网络结果
    condition_networks['Resting'] = {
        'correlation_matrix': resting_corr_matrix,
        'p_matrix': resting_p_matrix,
        'adjacency_matrix': resting_adj_matrix,
        'neural_activity': baseline_mean_activity,
        'category': 'Resting',
        'intensity': 'Baseline'
    }
    condition_metrics['Resting'] = resting_metrics
    
    print("静息状态网络指标:")
    print(f"  节点数: {resting_metrics['n_nodes']}")
    print(f"  连接数: {resting_metrics['n_edges']}")
    print(f"  密度: {resting_metrics['density']:.3f}")
    print(f"  平均度: {resting_metrics['avg_degree']:.2f}")
    print(f"  平均聚类系数: {resting_metrics['avg_clustering']:.3f}")
    print(f"  模块化: {resting_metrics['modularity']:.3f}")
    
    # %% 多条件网络指标比较
    print("\n=== 多条件网络指标比较 ===")
    
    if len(condition_metrics) >= 2:
        # 创建详细比较表格
        detailed_comparison_df = pd.DataFrame(condition_metrics).T
        print("\n详细网络指标比较:")
        print(detailed_comparison_df.round(3))
        
        # 保存详细比较结果
        detailed_comparison_df.to_csv(os.path.join(results_dir, 'detailed_network_metrics_comparison.csv'))
        print(f"\n详细网络指标比较已保存")
        
        # 按类别比较强度效应（仅类别2）
        print("\n=== 类别2的强度效应分析 ===")
        cat2_conditions = [k for k in condition_metrics.keys() if k.startswith('Cat2')]
        
        if len(cat2_conditions) >= 2:
            # 可视化强度效应
            visualize_intensity_effects(
                condition_metrics,
                target_category=2,
                title="Category 2 Stimulus Intensity Effects Analysis",
                save_path=os.path.join(results_dir, 'category2_intensity_effects.png')
            )
            
            print("\n类别2不同强度的网络指标对比:")
            cat2_df = detailed_comparison_df.loc[cat2_conditions]
            print(cat2_df.round(3))
            
            # 按强度排序显示强度效应
            print("\n类别2各强度间的网络指标差异:")
            cat2_intensities = []
            for cond in cat2_conditions:
                intensity_str = cond.split('_Int')[1]
                cat2_intensities.append(float(intensity_str))
            
            # 按强度值排序
            intensity_order = sorted(zip(cat2_intensities, cat2_conditions))
            
            # 计算相对于最低强度的效应
            base_intensity = intensity_order[0][1]  # 最低强度条件
            print(f"\n以 {base_intensity} 为基准的强度效应:")
            
            for metric in ['density', 'avg_clustering', 'modularity', 'avg_degree']:
                print(f"\n{metric}:")
                base_value = condition_metrics[base_intensity][metric]
                for intensity_val, condition in intensity_order:
                    effect = condition_metrics[condition][metric] - base_value
                    print(f"  强度{intensity_val}: {condition_metrics[condition][metric]:.3f} (Δ{effect:+.3f})")
        
        # 保存多条件比较数据
        key_metrics = ['density', 'avg_clustering', 'modularity', 'avg_degree', 'small_world_sigma']
        conditions = list(condition_metrics.keys())
        
        detailed_comparison_data = {
            'conditions': conditions,
            'metrics': key_metrics
        }
        
        for metric in key_metrics:
            values = [condition_metrics[cond][metric] for cond in conditions]
            detailed_comparison_data[f'{metric}_values'] = values
        
        # 添加条件分类信息
        condition_types = []
        condition_colors = []
        for cond in conditions:
            if cond == 'Resting':
                condition_types.append('Resting')
                condition_colors.append('gray')
            elif 'Cat1' in cond:
                condition_types.append('Category1')
                if 'Int0.0' in cond:
                    condition_colors.append('navy')
                elif 'Int0.2' in cond:
                    condition_colors.append('blue')
                elif 'Int0.5' in cond:
                    condition_colors.append('lightblue')
                elif 'Int1.0' in cond:
                    condition_colors.append('skyblue')
                else:
                    condition_colors.append('blue')
            elif 'Cat2' in cond:
                condition_types.append('Category2')
                if 'Int0.0' in cond:
                    condition_colors.append('darkred')
                elif 'Int0.2' in cond:
                    condition_colors.append('red')
                elif 'Int0.5' in cond:
                    condition_colors.append('lightcoral')
                elif 'Int1.0' in cond:
                    condition_colors.append('pink')
                else:
                    condition_colors.append('red')
            else:
                condition_types.append('Other')
                condition_colors.append('green')
        
        detailed_comparison_data['condition_types'] = condition_types
        detailed_comparison_data['condition_colors'] = condition_colors
        
        np.savez_compressed(
            os.path.join(results_dir, 'detailed_network_metrics_comparison.npz'),
            **detailed_comparison_data
        )
        print("详细网络指标比较数据已保存")
        
        # 保存类别2的强度效应数据
        if len(cat2_conditions) >= 2:
            metrics_for_intensity = ['density', 'avg_clustering', 'modularity', 'avg_degree']
            
            # 获取类别2的所有强度条件，按强度值排序
            cat2_data = []
            for cond in cat2_conditions:
                intensity_str = cond.split('_Int')[1]
                intensity_val = float(intensity_str)
                cat2_data.append((intensity_val, cond))
            cat2_data = sorted(cat2_data)
            
            # 保存强度比较数据
            intensity_comparison = {
                'intensities': [data[0] for data in cat2_data],
                'conditions': [data[1] for data in cat2_data],
                'metrics': metrics_for_intensity
            }
            
            for metric in metrics_for_intensity:
                values = [condition_metrics[data[1]][metric] for data in cat2_data]
                intensity_comparison[f'{metric}_values'] = values
            
            np.savez_compressed(
                os.path.join(results_dir, 'category2_intensity_comparison.npz'),
                **intensity_comparison
            )
            print("类别2强度效应数据已保存")
            
            # 保存强度趋势分析
            trend_analysis = {}
            for metric in metrics_for_intensity:
                intensities = [data[0] for data in cat2_data]
                values = [condition_metrics[data[1]][metric] for data in cat2_data]
                
                # 计算趋势统计
                correlation_coef = np.corrcoef(intensities, values)[0, 1] if len(intensities) > 1 else 0
                slope = np.polyfit(intensities, values, 1)[0] if len(intensities) > 1 else 0
                
                trend_analysis[f'{metric}_trend'] = {
                    'intensities': intensities,
                    'values': values,
                    'correlation': correlation_coef,
                    'slope': slope,
                    'min_value': min(values),
                    'max_value': max(values),
                    'range': max(values) - min(values)
                }
            
            # 使用pickle保存嵌套字典结构
            import pickle
            with open(os.path.join(results_dir, 'category2_intensity_trends.pkl'), 'wb') as f:
                pickle.dump(trend_analysis, f)
            print("类别2强度趋势分析已保存")
        
        # 保存静息状态 vs 任务状态对比数据
        task_conditions = [k for k in condition_metrics.keys() if k != 'Resting']
        if task_conditions:
            # 选择关键指标进行比较
            comparison_metrics = ['density', 'avg_clustering', 'modularity', 'avg_degree']
            
            resting_vs_task_data = {
                'task_conditions': task_conditions,
                'comparison_metrics': comparison_metrics,
                'resting_state': condition_metrics['Resting']
            }
            
            # 保存每个指标的任务状态值
            for metric in comparison_metrics:
                resting_value = condition_metrics['Resting'][metric]
                task_values = [condition_metrics[cond][metric] for cond in task_conditions]
                
                resting_vs_task_data[f'{metric}_resting'] = resting_value
                resting_vs_task_data[f'{metric}_task_values'] = task_values
                resting_vs_task_data[f'{metric}_task_mean'] = np.mean(task_values)
                resting_vs_task_data[f'{metric}_task_std'] = np.std(task_values)
                resting_vs_task_data[f'{metric}_difference'] = np.mean(task_values) - resting_value
            
            np.savez_compressed(
                os.path.join(results_dir, 'resting_vs_task_comparison.npz'),
                **resting_vs_task_data
            )
            print("静息状态vs任务状态比较数据已保存")
    
    # %% 保存所有网络数据
    print("\n=== 保存网络分析结果 ===")
    
    # 保存所有网络矩阵
    networks_save_path = os.path.join(results_dir, 'all_network_matrices.npz')
    save_dict = {}
    
    for condition, network_data in condition_networks.items():
        save_dict[f'{condition}_correlation'] = network_data['correlation_matrix']
        save_dict[f'{condition}_adjacency'] = network_data['adjacency_matrix']
        save_dict[f'{condition}_pvalues'] = network_data['p_matrix']
    
    np.savez_compressed(networks_save_path, **save_dict)
    print(f"所有网络矩阵已保存: {networks_save_path}")
    
    # 保存详细的网络指标
    metrics_save_path = os.path.join(results_dir, 'detailed_network_metrics.json')
    import json
    
    # 转换numpy数据类型为Python原生类型
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    metrics_json = convert_numpy_types(condition_metrics)
    
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"详细网络指标已保存: {metrics_save_path}")
    
    print(f"\n=== 完整网络分析完成 ===")
    print(f"分析了以下条件:")
    for condition in condition_networks.keys():
        print(f"  - {condition}")
    print(f"所有结果保存在: {results_dir}") 