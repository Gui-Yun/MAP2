# 高级网络分析：富人俱乐部与信息分解
# guiy24@mails.tsinghua.edu.cn
# 2025-01-09
# 实现富人俱乐部组织分析和部分信息分解（PID）分析

# %% 导入必要的库
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置为非交互式后端，防止弹出窗口
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import mutual_info_score
import pandas as pd
from collections import defaultdict, Counter
import warnings
from itertools import combinations
import os
warnings.filterwarnings('ignore')

# 导入项目模块
from loaddata import (
    load_data, segment_neuron_data, reclassify_labels, 
    fast_rr_selection, cfg
)
from network import (
    compute_network_metrics, compute_correlation_matrix, threshold_correlation_matrix
)
from degree import (
    build_correlation_network, calculate_centrality_metrics
)

# %% 配置参数
class AdvancedAnalysisConfig:
    """高级网络分析配置"""
    
    # 富人俱乐部分析参数
    RICH_CLUB_K_RANGE = None          # 度值范围，None表示自动确定
    N_RANDOM_NETWORKS = 100           # 随机网络数量
    RICH_CLUB_THRESHOLD = 1.0         # 富人俱乐部系数阈值
    
    # PID分析参数
    PID_DISCRETIZATION_BINS = 10      # 离散化分箱数
    PID_HUB_PERCENTILE = 90          # 枢纽神经元百分位阈值
    PID_PERIPHERAL_PERCENTILE = 10    # 边缘神经元百分位阈值
    
    # 分析时间窗口
    STIMULUS_START = 10               # 刺激开始时间点
    STIMULUS_DURATION = 20            # 刺激持续时间
    
    # 可视化参数
    FIGURE_SIZE = (12, 8)
    FIGURE_SIZE_LARGE = (15, 10)
    DPI = 300
    VISUALIZATION_DPI = 300
    
    # 结果保存路径
    RESULTS_DIR = 'results/advanced_analysis'
    
    @classmethod
    def ensure_results_dir(cls):
        """确保结果目录存在"""
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)
    
    @classmethod
    def get_results_dir(cls):
        """获取版本感知的结果保存路径"""
        # 使用版本感知的路径管理
        base_dir = cfg.get_results_dir() if hasattr(cfg, 'get_results_dir') else 'results'
        advanced_dir = os.path.join(base_dir, 'advanced_analysis')
        os.makedirs(advanced_dir, exist_ok=True)
        return advanced_dir

# 实例化配置
acfg = AdvancedAnalysisConfig()

# %% 富人俱乐部分析函数

def calculate_rich_club_coefficient(G, k=None):
    """
    计算网络的富人俱乐部系数
    
    参数:
    G: NetworkX图对象
    k: 度值，如果为None则计算所有可能的度值
    
    返回:
    rich_club_coeffs: 富人俱乐部系数字典 {k: coefficient}
    """
    print("计算富人俱乐部系数...")
    
    # 获取度序列
    degrees = dict(G.degree())
    degree_sequence = list(degrees.values())
    
    # 确定k的范围
    if k is None:
        unique_degrees = sorted(set(degree_sequence))
        k_values = unique_degrees
    else:
        k_values = [k]
    
    rich_club_coeffs = {}
    
    for k_val in k_values:
        # 找到度值大于等于k的节点（富节点）
        rich_nodes = [node for node, deg in degrees.items() if deg >= k_val]
        
        if len(rich_nodes) <= 1:
            rich_club_coeffs[k_val] = 0.0
            continue
        
        # 计算富节点之间的连接数
        rich_subgraph = G.subgraph(rich_nodes)
        actual_edges = rich_subgraph.number_of_edges()
        
        # 计算最大可能连接数
        n_rich = len(rich_nodes)
        max_possible_edges = n_rich * (n_rich - 1) // 2
        
        # 富人俱乐部系数
        if max_possible_edges > 0:
            phi = actual_edges / max_possible_edges
        else:
            phi = 0.0
        
        rich_club_coeffs[k_val] = phi
        
        if k_val % 10 == 0 or len(k_values) <= 10:
            print(f"  k={k_val}: 富节点数={n_rich}, 连接数={actual_edges}/{max_possible_edges}, φ={phi:.4f}")
    
    print(f"富人俱乐部系数计算完成，k值范围: {min(k_values)}-{max(k_values)}")
    return rich_club_coeffs

def generate_random_networks(G, n_random=acfg.N_RANDOM_NETWORKS, method='degree_preserving'):
    """
    生成保持度分布的随机网络
    
    参数:
    G: 原始网络
    n_random: 随机网络数量
    method: 随机化方法
    
    返回:
    random_graphs: 随机图列表
    """
    print(f"生成 {n_random} 个随机网络（{method}）...")
    
    random_graphs = []
    degree_sequence = [d for n, d in G.degree()]
    
    if method == 'degree_preserving':
        # 生成保持度序列的随机图
        for i in range(n_random):
            try:
                # 使用configuration model生成随机图
                random_G = nx.configuration_model(degree_sequence)
                # 移除自环和重边
                random_G = nx.Graph(random_G)
                random_G.remove_edges_from(nx.selfloop_edges(random_G))
                random_graphs.append(random_G)
            except:
                # 如果configuration model失败，使用Erdős-Rényi图
                n_nodes = G.number_of_nodes()
                edge_prob = G.number_of_edges() / (n_nodes * (n_nodes - 1) / 2)
                random_G = nx.erdos_renyi_graph(n_nodes, edge_prob)
                random_graphs.append(random_G)
            
            if (i + 1) % 20 == 0:
                print(f"  已生成 {i + 1}/{n_random} 个随机网络")
    
    print(f"随机网络生成完成")
    return random_graphs

def calculate_normalized_rich_club_coefficient(G, random_graphs=None):
    """
    计算归一化的富人俱乐部系数
    
    参数:
    G: 原始网络
    random_graphs: 随机网络列表，如果为None则自动生成
    
    返回:
    results: 富人俱乐部分析结果
    """
    print("计算归一化富人俱乐部系数...")
    
    # 计算原始网络的富人俱乐部系数
    original_coeffs = calculate_rich_club_coefficient(G)
    
    # 生成随机网络（如果未提供）
    if random_graphs is None:
        random_graphs = generate_random_networks(G)
    
    # 计算随机网络的富人俱乐部系数
    print("计算随机网络的富人俱乐部系数...")
    k_values = list(original_coeffs.keys())
    random_coeffs = {k: [] for k in k_values}
    
    for i, random_G in enumerate(random_graphs):
        random_coeff = calculate_rich_club_coefficient(random_G, k=None)
        
        for k in k_values:
            if k in random_coeff:
                random_coeffs[k].append(random_coeff[k])
        
        if (i + 1) % 20 == 0:
            print(f"  已处理 {i + 1}/{len(random_graphs)} 个随机网络")
    
    # 计算归一化系数
    normalized_coeffs = {}
    random_means = {}
    random_stds = {}
    
    for k in k_values:
        if len(random_coeffs[k]) > 0:
            random_mean = np.mean(random_coeffs[k])
            random_std = np.std(random_coeffs[k])
            
            if random_mean > 0:
                normalized_coeffs[k] = original_coeffs[k] / random_mean
            else:
                normalized_coeffs[k] = 0.0
            
            random_means[k] = random_mean
            random_stds[k] = random_std
        else:
            normalized_coeffs[k] = 0.0
            random_means[k] = 0.0
            random_stds[k] = 0.0
    
    # 识别富人俱乐部区域
    rich_club_region = []
    for k in sorted(k_values):
        if normalized_coeffs[k] > acfg.RICH_CLUB_THRESHOLD:
            rich_club_region.append(k)
    
    results = {
        'original_coefficients': original_coeffs,
        'random_means': random_means,
        'random_stds': random_stds,
        'normalized_coefficients': normalized_coeffs,
        'rich_club_region': rich_club_region,
        'k_values': k_values,
        'n_random_networks': len(random_graphs)
    }
    
    print(f"富人俱乐部分析完成")
    if rich_club_region:
        print(f"检测到富人俱乐部结构，k值范围: {min(rich_club_region)}-{max(rich_club_region)}")
    else:
        print("未检测到显著的富人俱乐部结构")
    
    return results

# %% 信息分解（PID）分析函数

def discretize_data(data, n_bins=acfg.PID_DISCRETIZATION_BINS):
    """
    将连续数据离散化为分类变量（改进版本）
    
    参数:
    data: 连续数据数组
    n_bins: 分箱数量
    
    返回:
    discretized: 离散化后的数据
    """
    # 检查数据变异性
    if len(np.unique(data)) <= 1:
        print("警告: 数据缺乏变异性")
        return np.zeros_like(data, dtype=int)
    
    # 如果唯一值已经很少，直接使用
    if len(np.unique(data)) <= n_bins:
        unique_vals = sorted(np.unique(data))
        discrete = np.zeros_like(data, dtype=int)
        for i, val in enumerate(unique_vals):
            discrete[data == val] = i
        return discrete
    
    # 使用等频率分箱（改进版本）
    try:
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(data, quantiles)
        # 确保边界唯一
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) > 1:
            discretized = np.digitize(data, bin_edges[1:-1])
        else:
            discretized = np.zeros_like(data, dtype=int)
    except:
        # 备选方案：等宽度分箱
        min_val, max_val = data.min(), data.max()
        if max_val > min_val:
            bin_width = (max_val - min_val) / n_bins
            discretized = ((data - min_val) / bin_width).astype(int)
            discretized = np.clip(discretized, 0, n_bins - 1)
        else:
            discretized = np.zeros_like(data, dtype=int)
    
    return discretized.astype(int)

def calculate_mutual_information_discrete(X, Y):
    """
    计算两个离散变量的互信息（改进版本）
    
    参数:
    X, Y: 离散化的数据数组
    
    返回:
    mi: 互信息值
    """
    # 检查输入有效性
    if len(np.unique(X)) <= 1 or len(np.unique(Y)) <= 1:
        return 0.0
    
    # 使用改进的计算方法
    X_unique = sorted(np.unique(X))
    Y_unique = sorted(np.unique(Y))
    
    n_total = len(X)
    mi = 0.0
    
    for x in X_unique:
        for y in Y_unique:
            # 联合概率
            p_xy = np.sum((X == x) & (Y == y)) / n_total
            if p_xy == 0:
                continue
                
            # 边际概率
            p_x = np.sum(X == x) / n_total
            p_y = np.sum(Y == y) / n_total
            
            # 互信息贡献
            mi += p_xy * np.log2(p_xy / (p_x * p_y))
    
    return max(0, mi)

def calculate_conditional_mutual_information(X, Y, Z):
    """
    计算条件互信息 I(X;Y|Z)
    
    参数:
    X, Y, Z: 离散化的数据数组
    
    返回:
    cmi: 条件互信息值
    """
    # I(X;Y|Z) = I(X;Y,Z) - I(X;Z)
    
    # 将Y和Z组合
    YZ = Y * (np.max(Z) + 1) + Z  # 简单的组合方式
    
    mi_xyz = calculate_mutual_information_discrete(X, YZ)
    mi_xz = calculate_mutual_information_discrete(X, Z)
    
    cmi = mi_xyz - mi_xz
    return max(0, cmi)  # 条件互信息不能为负

def partial_information_decomposition(X1, X2, Y):
    """
    对三个变量进行部分信息分解（改进版PID）
    
    参数:
    X1, X2: 源变量（离散化）
    Y: 目标变量（离散化）
    
    返回:
    pid_results: PID分解结果
    """
    print("执行部分信息分解（PID）...")
    
    # 检查数据有效性
    if len(np.unique(X1)) <= 1 or len(np.unique(X2)) <= 1 or len(np.unique(Y)) <= 1:
        print("  警告: 某些变量缺乏变异性，返回零结果")
        return {
            'redundancy': 0.0,
            'unique_X1': 0.0,
            'unique_X2': 0.0,
            'synergy': 0.0,
            'total_info': 0.0,
            'total_reconstructed': 0.0,
            'I_X1_Y': 0.0,
            'I_X2_Y': 0.0,
            'I_X1X2_Y': 0.0
        }
    
    # 计算各种互信息
    I_X1_Y = calculate_mutual_information_discrete(X1, Y)
    I_X2_Y = calculate_mutual_information_discrete(X2, Y)
    
    # 创建X1X2的联合变量（改进方法）
    X1_max = np.max(X1) + 1
    X1X2_joint = X1 * X1_max + X2
    I_X1X2_Y = calculate_mutual_information_discrete(X1X2_joint, Y)
    
    # 简化的PID分解（更稳定的计算）
    # 协同信息: 联合信息减去单独信息之和
    synergy = max(0, I_X1X2_Y - I_X1_Y - I_X2_Y)
    
    # 冗余信息: 最小的单独信息（修正版本）
    redundancy = min(I_X1_Y, I_X2_Y) - max(0, (I_X1_Y + I_X2_Y - I_X1X2_Y) / 2)
    redundancy = max(0, redundancy)
    
    # 唯一信息
    unique_X1 = max(0, I_X1_Y - redundancy)
    unique_X2 = max(0, I_X2_Y - redundancy)
    
    total_reconstructed = redundancy + unique_X1 + unique_X2 + synergy
    
    pid_results = {
        'redundancy': redundancy,
        'unique_X1': unique_X1,
        'unique_X2': unique_X2,
        'synergy': synergy,
        'total_info': I_X1X2_Y,
        'total_reconstructed': total_reconstructed,
        'I_X1_Y': I_X1_Y,
        'I_X2_Y': I_X2_Y,
        'I_X1X2_Y': I_X1X2_Y
    }
    
    print(f"  冗余信息: {redundancy:.4f}")
    print(f"  X1唯一信息: {unique_X1:.4f}")
    print(f"  X2唯一信息: {unique_X2:.4f}")
    print(f"  协同信息: {synergy:.4f}")
    print(f"  总信息: {I_X1X2_Y:.4f}")
    print(f"  重构总和: {total_reconstructed:.4f}")
    
    return pid_results

def analyze_hub_peripheral_information_dynamics(segments, labels, rr_neurons, neuron_pos=None):
    """
    分析枢纽和边缘神经元的信息动力学
    
    参数:
    segments: 神经数据片段 (trials, neurons, timepoints)
    labels: 标签数组
    rr_neurons: RR神经元索引
    neuron_pos: 神经元位置信息（可选）
    
    返回:
    pid_analysis: 信息分解分析结果
    """
    print("分析枢纽-边缘神经元信息动力学...")
    
    # 过滤有效数据
    valid_mask = labels != 0
    valid_segments = segments[valid_mask][:, rr_neurons, :]
    valid_labels = labels[valid_mask]
    
    # 构建功能连接网络来识别枢纽和边缘神经元
    print("构建功能连接网络识别枢纽神经元...")
    
    # 提取刺激期数据
    stimulus_window = np.arange(acfg.STIMULUS_START, 
                               min(acfg.STIMULUS_START + acfg.STIMULUS_DURATION, 
                                   valid_segments.shape[2]))
    neural_activity = np.mean(valid_segments[:, :, stimulus_window], axis=2)
    
    # 计算相关性矩阵和构建网络
    corr_matrix, p_matrix = compute_correlation_matrix(neural_activity, method='pearson')
    adj_matrix = threshold_correlation_matrix(corr_matrix, p_matrix, 
                                            method='density', network_density=0.1)
    
    # 创建网络并计算度中心性
    G = nx.from_numpy_array(adj_matrix)
    degrees = dict(G.degree())
    degree_values = list(degrees.values())
    
    # 识别枢纽和边缘神经元
    hub_threshold = np.percentile(degree_values, acfg.PID_HUB_PERCENTILE)
    peripheral_threshold = np.percentile(degree_values, acfg.PID_PERIPHERAL_PERCENTILE)
    
    hub_indices = [i for i in range(len(degree_values)) if degree_values[i] >= hub_threshold]
    peripheral_indices = [i for i in range(len(degree_values)) if degree_values[i] <= peripheral_threshold]
    
    print(f"识别枢纽神经元: {len(hub_indices)} 个 (度 >= {hub_threshold})")
    print(f"识别边缘神经元: {len(peripheral_indices)} 个 (度 <= {peripheral_threshold})")
    
    if len(hub_indices) == 0 or len(peripheral_indices) == 0:
        print("警告: 枢纽或边缘神经元数量不足，无法进行PID分析")
        return None
    
    # 准备刺激标签用于信息分解
    unique_labels = np.unique(valid_labels)
    if len(unique_labels) > 1:
        # 如果有多个类别，直接使用标签
        stimulus_labels = valid_labels.astype(int)
    else:
        print("警告: 只有一个刺激类别，PID分析可能不可靠")
        stimulus_labels = valid_labels.astype(int)
    
    pid_results = {}
    
    # 分析不同类型的神经元组合
    analysis_pairs = [
        ('hub_hub', hub_indices, hub_indices),
        ('hub_peripheral', hub_indices, peripheral_indices),
        ('peripheral_peripheral', peripheral_indices, peripheral_indices)
    ]
    
    for pair_name, group1, group2 in analysis_pairs:
        print(f"\n分析 {pair_name} 信息动力学...")
        
        if len(group1) == 0 or len(group2) == 0:
            continue
        
        # 选择代表性神经元进行分析
        n_pairs_to_analyze = min(20, len(group1) * len(group2))  # 限制分析的神经元对数量
        
        pair_results = []
        
        for i, idx1 in enumerate(group1):
            for j, idx2 in enumerate(group2):
                if pair_name == 'hub_hub' and i >= j:  # 避免重复分析同一对
                    continue
                if pair_name == 'peripheral_peripheral' and i >= j:
                    continue
                
                if len(pair_results) >= n_pairs_to_analyze:
                    break
                
                # 提取神经元活动数据
                neuron1_activity = np.mean(valid_segments[:, idx1, stimulus_window], axis=1)
                neuron2_activity = np.mean(valid_segments[:, idx2, stimulus_window], axis=1)
                
                # 离散化
                neuron1_discrete = discretize_data(neuron1_activity)
                neuron2_discrete = discretize_data(neuron2_activity)
                
                # 执行PID分析
                try:
                    pid_result = partial_information_decomposition(
                        neuron1_discrete, neuron2_discrete, stimulus_labels)
                    pair_results.append(pid_result)
                except Exception as e:
                    print(f"  神经元对 ({idx1}, {idx2}) PID分析失败: {e}")
                    continue
            
            if len(pair_results) >= n_pairs_to_analyze:
                break
        
        if pair_results:
            # 汇总结果
            avg_results = {}
            for key in pair_results[0].keys():
                values = [result[key] for result in pair_results]
                avg_results[key + '_mean'] = np.mean(values)
                avg_results[key + '_std'] = np.std(values)
                avg_results[key + '_values'] = values
            
            avg_results['n_pairs_analyzed'] = len(pair_results)
            pid_results[pair_name] = avg_results
            
            print(f"  分析了 {len(pair_results)} 个神经元对")
            print(f"  平均冗余信息: {avg_results['redundancy_mean']:.4f} ± {avg_results['redundancy_std']:.4f}")
            print(f"  平均协同信息: {avg_results['synergy_mean']:.4f} ± {avg_results['synergy_std']:.4f}")
    
    # 总体分析结果
    analysis_summary = {
        'hub_indices': hub_indices,
        'peripheral_indices': peripheral_indices,
        'hub_threshold': hub_threshold,
        'peripheral_threshold': peripheral_threshold,
        'n_hubs': len(hub_indices),
        'n_peripheral': len(peripheral_indices),
        'network_metrics': {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'density': nx.density(G),
            'avg_degree': np.mean(degree_values)
        }
    }
    
    return {
        'pid_results': pid_results,
        'analysis_summary': analysis_summary,
        'network': G,
        'correlation_matrix': corr_matrix,
        'adjacency_matrix': adj_matrix
    }

# %% 数据保存和可视化函数

def visualize_rich_club_results(rich_club_results, condition_name="", save_path=None):
    """专业可视化富人俱乐部分析结果"""
    import matplotlib.pyplot as plt
    
    # 设置专业绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12, 'font.family': 'Arial', 'axes.titlesize': 14,
        'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 11, 'figure.titlesize': 16, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': 1.5, 'axes.edgecolor': '#2C3E50',
        'grid.alpha': 0.3, 'grid.linewidth': 0.8, 'figure.facecolor': 'white',
        'axes.facecolor': 'white', 'legend.framealpha': 0.9, 'lines.linewidth': 2.5,
        'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.facecolor': 'white'
    })
    
    # 专业配色方案
    colors = {
        'observed': '#2E86AB',
        'random': '#6C757D', 
        'significant': '#F18F01',
        'threshold': '#E74C3C',
        'fill': '#95A5A6'
    }
    
    k_values = rich_club_results['k_values']
    original_coeffs = [rich_club_results['original_coefficients'][k] for k in k_values]
    normalized_coeffs = [rich_club_results['normalized_coefficients'][k] for k in k_values]
    random_means = [rich_club_results['random_means'][k] for k in k_values]
    random_stds = [rich_club_results['random_stds'][k] for k in k_values]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. 原始系数 vs 随机网络对比
    ax1.plot(k_values, original_coeffs, 'o-', label='Observed Network', 
            linewidth=3, markersize=7, color=colors['observed'], alpha=0.9,
            markerfacecolor='white', markeredgewidth=2)
    
    # 随机网络置信区间
    ax1.fill_between(k_values, 
                     [max(0, m - s) for m, s in zip(random_means, random_stds)],
                     [m + s for m, s in zip(random_means, random_stds)],
                     alpha=0.3, color=colors['fill'], label='Random Networks ±1σ')
    
    ax1.plot(k_values, random_means, '--', label='Random Mean', 
            linewidth=2.5, alpha=0.8, color=colors['random'])
    
    ax1.set_xlabel('Degree k', fontweight='bold')
    ax1.set_ylabel('Rich Club Coefficient φ(k)', fontweight='bold')
    ax1.set_title('Rich Club Coefficient', fontsize=12, fontweight='bold', pad=15)
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.set_ylim(bottom=0)
    
    # 2. 归一化系数
    ax2.plot(k_values, normalized_coeffs, 'o-', color=colors['threshold'], 
            linewidth=3, markersize=7, alpha=0.9,
            markerfacecolor='white', markeredgewidth=2)
    
    # 显著性阈值线
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, 
               alpha=0.6, label='Significance Threshold')
    
    # 标记富人俱乐部区域
    rich_club_region = rich_club_results['rich_club_region']
    if rich_club_region:
        for k in rich_club_region:
            if k in rich_club_results['normalized_coefficients']:
                coeff = rich_club_results['normalized_coefficients'][k]
                ax2.plot(k, coeff, 'o', color=colors['significant'], 
                        markersize=12, alpha=0.8, markeredgecolor='white', 
                        markeredgewidth=2)
        
        # 添加富人俱乐部区域标注
        ax2.text(0.98, 0.95, f'Rich Club Region\nk ∈ [{min(rich_club_region)}, {max(rich_club_region)}]',
                transform=ax2.transAxes, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['significant'], alpha=0.8),
                color='white', verticalalignment='top', horizontalalignment='right')
    else:
        ax2.text(0.98, 0.95, 'No Rich Club\nStructure Detected',
                transform=ax2.transAxes, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['random'], alpha=0.8),
                color='white', verticalalignment='top', horizontalalignment='right')
    
    ax2.set_xlabel('Degree k', fontweight='bold')
    ax2.set_ylabel('Normalized Coefficient φnorm(k)', fontweight='bold')
    ax2.set_title('Normalized Rich Club Coefficient', fontsize=12, fontweight='bold', pad=15)
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.set_ylim(bottom=0)
    
    # 添加条件信息
    if condition_name:
        fig.suptitle(f'Rich Club Analysis - {condition_name}', 
                    fontsize=14, fontweight='bold', y=0.95)
    else:
        fig.suptitle('Rich Club Analysis', fontsize=14, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def visualize_pid_results(pid_results, title="Partial Information Decomposition", save_path=None):
    """专业可视化PID分析结果"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 设置专业绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12, 'font.family': 'Arial', 'axes.titlesize': 14,
        'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 11, 'figure.titlesize': 16, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': 1.5, 'axes.edgecolor': '#2C3E50',
        'grid.alpha': 0.3, 'grid.linewidth': 0.8, 'figure.facecolor': 'white',
        'axes.facecolor': 'white', 'legend.framealpha': 0.9, 'lines.linewidth': 2.5,
        'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.facecolor': 'white'
    })
    
    # 专业配色方案
    colors = {
        'redundancy': '#E74C3C', 'synergy': '#2E86AB', 
        'unique_X1': '#F39C12', 'unique_X2': '#27AE60',
        'primary': '#2E86AB', 'secondary': '#A23B72', 
        'accent': '#F18F01', 'neutral': '#6C757D'
    }
    
    if 'pid_results' not in pid_results:
        return
    
    pair_types = list(pid_results['pid_results'].keys())
    n_pairs = len(pair_types)
    
    # 主要分析图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # 四种信息成分
    info_components = ['redundancy_mean', 'synergy_mean', 'unique_X1_mean', 'unique_X2_mean']
    component_names = ['Redundancy', 'Synergy', 'Unique X1', 'Unique X2']
    component_colors = [colors['redundancy'], colors['synergy'], colors['unique_X1'], colors['unique_X2']]
    
    for i, (component, name, color) in enumerate(zip(info_components, component_names, component_colors)):
        ax = axes[i]
        
        values = []
        errors = []
        labels = []
        
        for pair_type in pair_types:
            if component in pid_results['pid_results'][pair_type]:
                values.append(pid_results['pid_results'][pair_type][component])
                std_key = component.replace('_mean', '_std')
                errors.append(pid_results['pid_results'][pair_type].get(std_key, 0))
                labels.append(pair_type.replace('_', '-'))
        
        if values:
            bars = ax.bar(range(len(values)), values, yerr=errors, 
                         color=color, alpha=0.8, capsize=5,
                         edgecolor='white', linewidth=2,
                         error_kw={'linewidth': 1.5, 'capthick': 1.5})
            
            ax.set_xlabel('Neuron Pair Type', fontweight='bold')
            ax.set_ylabel(f'{name}', fontweight='bold')
            ax.set_title(f'{name}', fontsize=12, fontweight='bold', pad=15)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontweight='bold')
            
            # 设置y轴范围
            max_val = max([v + e for v, e in zip(values, errors)])
            ax.set_ylim(0, max_val * 1.15)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace('.png', '_components.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # 信息分解饼图
    if len(pair_types) > 0:
        n_plots = min(3, len(pair_types))
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        
        for i, pair_type in enumerate(pair_types[:n_plots]):
            ax = axes[i] if n_plots > 1 else axes[0]
            
            results = pid_results['pid_results'][pair_type]
            values = [
                results.get('redundancy_mean', 0),
                results.get('unique_X1_mean', 0),
                results.get('unique_X2_mean', 0),
                results.get('synergy_mean', 0)
            ]
            
            labels = ['Redundancy', 'Unique X1', 'Unique X2', 'Synergy']
            pie_colors = [colors['redundancy'], colors['unique_X1'], colors['unique_X2'], colors['synergy']]
            
            # 只显示非零值
            non_zero_indices = [j for j, v in enumerate(values) if v > 0.001]
            if non_zero_indices:
                filtered_values = [values[j] for j in non_zero_indices]
                filtered_labels = [labels[j] for j in non_zero_indices]
                filtered_colors = [pie_colors[j] for j in non_zero_indices]
                
                wedges, texts, autotexts = ax.pie(filtered_values, labels=filtered_labels,
                                                 colors=filtered_colors, autopct='%1.3f',
                                                 startangle=90, textprops={'fontweight': 'bold'},
                                                 wedgeprops={'edgecolor': 'white', 'linewidth': 2})
                
                # 设置自动文本样式
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(10)
                
                ax.set_title(f'{pair_type.replace("_", "-")}', 
                           fontsize=12, fontweight='bold', pad=15)
        
        plt.suptitle(f'{title} - Information Breakdown', fontsize=14, fontweight='bold', y=0.95)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path.replace('.png', '_breakdown.png'), dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

def visualize_pid_conditions_comparison(pid_conditions_dict, save_path=None):
    """
    可视化多条件PID结果对比
    
    参数:
    pid_conditions_dict: 多条件PID结果字典
    save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 设置科研绘图风格
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'font.family': 'Arial',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300
    })
    
    # 提取所有条件和神经元对类型
    conditions = list(pid_conditions_dict.keys())
    if not conditions:
        return
    
    # 获取第一个条件的神经元对类型作为参考
    first_condition = conditions[0]
    if 'pid_results' not in pid_conditions_dict[first_condition]:
        return
        
    pair_types = list(pid_conditions_dict[first_condition]['pid_results'].keys())
    
    # PID成分
    components = ['redundancy_mean', 'unique_X1_mean', 'unique_X2_mean', 'synergy_mean']
    component_labels = ['Redundancy', 'Unique X1', 'Unique X2', 'Synergy']
    component_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 为每种神经元对类型创建一个子图
    for idx, pair_type in enumerate(pair_types):
        if idx >= 4:  # 最多显示4种类型
            break
            
        ax = axes[idx // 2, idx % 2]
        
        # 收集数据
        condition_names = []
        component_values = {comp: [] for comp in components}
        component_errors = {comp: [] for comp in components}
        
        for condition in conditions:
            if (condition in pid_conditions_dict and 
                'pid_results' in pid_conditions_dict[condition] and
                pair_type in pid_conditions_dict[condition]['pid_results']):
                
                condition_names.append(condition.replace('_', ' ').title())
                results = pid_conditions_dict[condition]['pid_results'][pair_type]
                
                for comp in components:
                    value = results.get(comp, 0)
                    component_values[comp].append(value)
                    
                    # 获取标准差
                    std_key = comp.replace('_mean', '_std')
                    error = results.get(std_key, 0)
                    component_errors[comp].append(error)
        
        if not condition_names:
            continue
            
        # 创建分组条形图
        x = np.arange(len(condition_names))
        width = 0.2
        
        for i, (comp, label, color) in enumerate(zip(components, component_labels, component_colors)):
            values = component_values[comp]
            errors = component_errors[comp]
            
            bars = ax.bar(x + i * width, values, width, label=label,
                         color=color, alpha=0.8, edgecolor='black', linewidth=0.8,
                         yerr=errors, capsize=3)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                if value > 0.01:  # 只对较大的值添加标签
                    ax.text(bar.get_x() + bar.get_width()/2, 
                           bar.get_height() + max(errors) * 0.1,
                           f'{value:.3f}', ha='center', va='bottom', 
                           fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Condition')
        ax.set_ylabel('Information (bits)')
        ax.set_title(f'PID Components - {pair_type.replace("_", "-").title()}')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(condition_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
    
    # 隐藏未使用的子图
    for idx in range(len(pair_types), 4):
        axes[idx // 2, idx % 2].set_visible(False)
    
    plt.suptitle('Partial Information Decomposition - Multi-Condition Comparison', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=acfg.VISUALIZATION_DPI, bbox_inches='tight')
        print(f"Multi-condition PID comparison saved to: {save_path}")
    plt.show()

def save_rich_club_analysis(rich_club_results, condition_name, save_dir=None):
    """
    保存富人俱乐部分析结果
    """
    if save_dir is None:
        save_dir = acfg.get_results_dir()
    
    filename = f"rich_club_analysis_{condition_name}.npz"
    
    save_data = {
        'k_values': rich_club_results['k_values'],
        'original_coefficients': [rich_club_results['original_coefficients'][k] 
                                for k in rich_club_results['k_values']],
        'normalized_coefficients': [rich_club_results['normalized_coefficients'][k] 
                                  for k in rich_club_results['k_values']],
        'random_means': [rich_club_results['random_means'][k] 
                        for k in rich_club_results['k_values']],
        'random_stds': [rich_club_results['random_stds'][k] 
                       for k in rich_club_results['k_values']],
        'rich_club_region': rich_club_results['rich_club_region'],
        'n_random_networks': rich_club_results['n_random_networks']
    }
    
    np.savez_compressed(
        os.path.join(save_dir, filename),
        **save_data
    )
    print(f"富人俱乐部分析结果已保存: {filename}")

def save_pid_analysis(pid_results, condition_name, save_dir=None):
    """
    保存信息分解分析结果
    """
    if save_dir is None:
        save_dir = acfg.get_results_dir()
    
    filename = f"pid_analysis_{condition_name}.npz"
    
    # 准备保存数据
    save_data = {}
    
    # 保存分析摘要
    if 'analysis_summary' in pid_results:
        summary = pid_results['analysis_summary']
        save_data.update({
            'hub_indices': summary['hub_indices'],
            'peripheral_indices': summary['peripheral_indices'],
            'hub_threshold': summary['hub_threshold'],
            'peripheral_threshold': summary['peripheral_threshold'],
            'n_hubs': summary['n_hubs'],
            'n_peripheral': summary['n_peripheral'],
        })
        
        # 保存网络指标
        for key, value in summary['network_metrics'].items():
            save_data[f'network_{key}'] = value
    
    # 保存PID结果
    if 'pid_results' in pid_results:
        for pair_type, results in pid_results['pid_results'].items():
            for metric, value in results.items():
                if isinstance(value, (int, float)):
                    save_data[f'{pair_type}_{metric}'] = value
                elif isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], (int, float)):
                        save_data[f'{pair_type}_{metric}'] = value
    
    np.savez_compressed(
        os.path.join(save_dir, filename),
        **save_data
    )
    print(f"信息分解分析结果已保存: {filename}")

# %% 表征稳定性分析函数

def analyze_representational_stability(segments, labels, rr_neurons, stimulus_window=None):
    """
    分析神经表征的稳定性 - 核心假说：结构化刺激应诱发比噪声更稳定的神经活动模式
    
    参数:
    segments: 神经数据片段 (trials, neurons, timepoints)
    labels: 标签数组
    rr_neurons: RR神经元索引
    stimulus_window: 刺激时间窗口，如果为None则使用配置的默认值
    
    返回:
    stability_results: 表征稳定性分析结果
    """
    print("=" * 50)
    print("表征稳定性分析")
    print("=" * 50)
    print("核心假说：结构化刺激应诱发比噪声更稳定的神经活动模式")
    print("-" * 50)
    
    # 过滤有效数据
    valid_mask = labels != 0
    valid_segments = segments[valid_mask][:, rr_neurons, :]
    valid_labels = labels[valid_mask]
    
    # 确定刺激时间窗口
    if stimulus_window is None:
        stimulus_window = np.arange(acfg.STIMULUS_START,
                                   min(acfg.STIMULUS_START + acfg.STIMULUS_DURATION,
                                       valid_segments.shape[2]))
    
    print(f"分析参数:")
    print(f"- 有效试次数: {len(valid_segments)}")
    print(f"- RR神经元数: {len(rr_neurons)}")
    print(f"- 刺激时间窗口: {stimulus_window[0]}-{stimulus_window[-1]} 时间点")
    
    # 提取刺激期神经活动
    neural_activity = valid_segments[:, :, stimulus_window]
    
    # 将神经活动数据展平为向量（trial x features）
    # features = neurons * timepoints
    n_trials, n_neurons, n_timepoints = neural_activity.shape
    neural_vectors = neural_activity.reshape(n_trials, n_neurons * n_timepoints)
    
    print(f"- 神经向量维度: {neural_vectors.shape} (trials x features)")
    
    # 第一步：计算各类刺激的"神经模板"
    print("\n第一步：计算神经模板...")
    unique_labels = np.unique(valid_labels)
    neural_templates = {}
    condition_trials = {}
    
    for condition in unique_labels:
        condition_mask = valid_labels == condition
        condition_data = neural_vectors[condition_mask]
        condition_trials[condition] = condition_data
        
        # 计算该条件的神经模板（跨试次平均）
        neural_templates[condition] = np.mean(condition_data, axis=0)
        
        print(f"  条件 {condition}: {np.sum(condition_mask)} 个试次")
        print(f"    模板向量统计: 均值={np.mean(neural_templates[condition]):.4f}, "
              f"标准差={np.std(neural_templates[condition]):.4f}")
    
    # 第二步：计算"试次-模板相似度"
    print("\n第二步：计算试次-模板相似度...")
    trial_template_similarities = {}
    
    for condition in unique_labels:
        print(f"  处理条件 {condition}...")
        
        condition_data = condition_trials[condition]
        template = neural_templates[condition]
        similarities = []
        
        # 计算每个试次与其模板的相关系数
        for i, trial_vector in enumerate(condition_data):
            # 计算皮尔逊相关系数
            correlation, _ = pearsonr(trial_vector, template)
            similarities.append(correlation)
            
            if i < 5:  # 只打印前5个试次的结果作为示例
                print(f"    试次 {i+1}: 相关系数 = {correlation:.4f}")
        
        trial_template_similarities[condition] = np.array(similarities)
        
        print(f"  条件 {condition} 统计:")
        print(f"    平均相似度: {np.mean(similarities):.4f} ± {np.std(similarities):.4f}")
        print(f"    相似度范围: [{np.min(similarities):.4f}, {np.max(similarities):.4f}]")
    
    # 第三步：统计检验
    print("\n第三步：统计检验...")
    similarity_values = [trial_template_similarities[cond] for cond in unique_labels]
    
    # ANOVA检验
    try:
        from scipy.stats import f_oneway
        f_stat, p_anova = f_oneway(*similarity_values)
        print(f"ANOVA检验: F = {f_stat:.4f}, p = {p_anova:.6f}")
    except Exception as e:
        print(f"ANOVA检验失败: {e}")
        f_stat, p_anova = np.nan, np.nan
    
    # 两两比较
    pairwise_comparisons = {}
    if len(unique_labels) >= 2:
        print("\n两两比较（t检验）:")
        from scipy.stats import ttest_ind
        
        for i, cond1 in enumerate(unique_labels):
            for j, cond2 in enumerate(unique_labels):
                if i < j:  # 避免重复比较
                    t_stat, p_value = ttest_ind(
                        trial_template_similarities[cond1],
                        trial_template_similarities[cond2]
                    )
                    pairwise_comparisons[f"{cond1}_vs_{cond2}"] = {
                        't_stat': t_stat,
                        'p_value': p_value
                    }
                    
                    significance = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "ns"))
                    print(f"  条件 {cond1} vs 条件 {cond2}: t = {t_stat:.4f}, p = {p_value:.6f} {significance}")
    
    # 计算效应量（Cohen's d）
    effect_sizes = {}
    if len(unique_labels) >= 2:
        print("\n效应量（Cohen's d）:")
        for comparison_name, stats in pairwise_comparisons.items():
            cond1, cond2 = comparison_name.split('_vs_')
            cond1, cond2 = int(cond1), int(cond2)
            
            sim1 = trial_template_similarities[cond1]
            sim2 = trial_template_similarities[cond2]
            
            # Cohen's d计算
            pooled_std = np.sqrt(((len(sim1) - 1) * np.var(sim1, ddof=1) + 
                                 (len(sim2) - 1) * np.var(sim2, ddof=1)) / 
                                (len(sim1) + len(sim2) - 2))
            
            if pooled_std > 0:
                cohens_d = (np.mean(sim1) - np.mean(sim2)) / pooled_std
                effect_sizes[comparison_name] = cohens_d
                
                effect_size_interpretation = ("小" if abs(cohens_d) < 0.5 else 
                                            ("中" if abs(cohens_d) < 0.8 else "大"))
                print(f"  {comparison_name}: Cohen's d = {cohens_d:.4f} ({effect_size_interpretation}效应)")
    
    # 汇总分析结果
    stability_results = {
        'neural_templates': neural_templates,
        'trial_template_similarities': trial_template_similarities,
        'similarity_statistics': {
            condition: {
                'mean': np.mean(similarities),
                'std': np.std(similarities),
                'min': np.min(similarities),
                'max': np.max(similarities),
                'median': np.median(similarities),
                'n_trials': len(similarities)
            } for condition, similarities in trial_template_similarities.items()
        },
        'statistical_tests': {
            'anova_f': f_stat,
            'anova_p': p_anova,
            'pairwise_comparisons': pairwise_comparisons,
            'effect_sizes': effect_sizes
        },
        'analysis_parameters': {
            'n_conditions': len(unique_labels),
            'conditions': list(unique_labels),
            'stimulus_window': stimulus_window.tolist(),
            'n_rr_neurons': len(rr_neurons),
            'feature_dimension': neural_vectors.shape[1]
        }
    }
    
    print("\n表征稳定性分析完成！")
    return stability_results

def visualize_representational_stability(stability_results, save_path=None):
    """
    可视化表征稳定性分析结果
    
    参数:
    stability_results: 稳定性分析结果
    save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 设置专业绘图风格
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'font.family': 'Arial',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
    
    # 提取数据
    similarities = stability_results['trial_template_similarities']
    stats = stability_results['similarity_statistics']
    conditions = stability_results['analysis_parameters']['conditions']
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 专业配色方案
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(conditions)]
    
    # 1. 小提琴图 + 箱形图
    ax1 = axes[0]
    
    # 准备数据
    plot_data = []
    plot_labels = []
    for condition in conditions:
        plot_data.extend(similarities[condition])
        plot_labels.extend([f'Condition {condition}'] * len(similarities[condition]))
    
    # 创建DataFrame用于seaborn
    import pandas as pd
    df = pd.DataFrame({
        'Similarity': plot_data,
        'Condition': plot_labels
    })
    
    # 绘制小提琴图
    violin_parts = ax1.violinplot([similarities[cond] for cond in conditions], 
                                 positions=range(len(conditions)),
                                 showmeans=True, showmedians=True)
    
    # 设置小提琴图颜色
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    # 添加箱形图
    box_parts = ax1.boxplot([similarities[cond] for cond in conditions],
                           positions=range(len(conditions)),
                           patch_artist=True, widths=0.3,
                           boxprops=dict(alpha=0.8),
                           medianprops=dict(color='black', linewidth=2))
    
    # 设置箱形图颜色
    for i, patch in enumerate(box_parts['boxes']):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.9)
    
    ax1.set_xlabel('Stimulus Condition', fontweight='bold')
    ax1.set_ylabel('Trial-Template Similarity\n(Pearson Correlation)', fontweight='bold')
    ax1.set_title('Representational Stability Across Conditions', fontweight='bold', pad=15)
    ax1.set_xticks(range(len(conditions)))
    ax1.set_xticklabels([f'Condition {c}' for c in conditions])
    ax1.grid(True, axis='y', alpha=0.3)
    
    # 2. 统计比较图
    ax2 = axes[1]
    
    # 计算均值和标准误
    means = [stats[cond]['mean'] for cond in conditions]
    stds = [stats[cond]['std'] for cond in conditions]
    sems = [std / np.sqrt(stats[cond]['n_trials']) for cond, std in zip(conditions, stds)]
    
    bars = ax2.bar(range(len(conditions)), means, yerr=sems,
                   color=colors, alpha=0.8, capsize=5,
                   edgecolor='black', linewidth=1.5,
                   error_kw={'linewidth': 2, 'capthick': 2})
    
    # 添加数值标签
    for i, (mean, sem) in enumerate(zip(means, sems)):
        ax2.text(i, mean + sem + 0.01, f'{mean:.3f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 添加显著性标记
    pairwise = stability_results['statistical_tests']['pairwise_comparisons']
    y_max = max([m + s for m, s in zip(means, sems)])
    y_offset = 0.05
    
    comparison_pairs = [(0, 1), (0, 2), (1, 2)] if len(conditions) >= 3 else [(0, 1)] if len(conditions) >= 2 else []
    
    for i, (idx1, idx2) in enumerate(comparison_pairs):
        if idx1 < len(conditions) and idx2 < len(conditions):
            cond1, cond2 = conditions[idx1], conditions[idx2]
            comparison_key = f"{cond1}_vs_{cond2}"
            
            if comparison_key in pairwise:
                p_val = pairwise[comparison_key]['p_value']
                significance = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
                
                if significance != "ns":
                    y_pos = y_max + y_offset * (i + 1)
                    ax2.plot([idx1, idx2], [y_pos, y_pos], 'k-', linewidth=1.5)
                    ax2.plot([idx1, idx1], [y_pos - 0.01, y_pos], 'k-', linewidth=1.5)
                    ax2.plot([idx2, idx2], [y_pos - 0.01, y_pos], 'k-', linewidth=1.5)
                    ax2.text((idx1 + idx2) / 2, y_pos + 0.01, significance, 
                            ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax2.set_xlabel('Stimulus Condition', fontweight='bold')
    ax2.set_ylabel('Mean Trial-Template Similarity\n(± SEM)', fontweight='bold')
    ax2.set_title('Statistical Comparison', fontweight='bold', pad=15)
    ax2.set_xticks(range(len(conditions)))
    ax2.set_xticklabels([f'Condition {c}' for c in conditions])
    ax2.grid(True, axis='y', alpha=0.3)
    
    # 3. 分布密度图
    ax3 = axes[2]
    
    for i, condition in enumerate(conditions):
        sim_data = similarities[condition]
        
        # 绘制密度曲线
        from scipy import stats as scipy_stats
        density = scipy_stats.gaussian_kde(sim_data)
        xs = np.linspace(sim_data.min(), sim_data.max(), 200)
        density_curve = density(xs)
        
        ax3.fill_between(xs, density_curve, alpha=0.6, color=colors[i], 
                        label=f'Condition {condition}')
        ax3.plot(xs, density_curve, color=colors[i], linewidth=2.5)
        
        # 添加均值线
        mean_val = np.mean(sim_data)
        ax3.axvline(mean_val, color=colors[i], linestyle='--', linewidth=2, alpha=0.8)
    
    ax3.set_xlabel('Trial-Template Similarity', fontweight='bold')
    ax3.set_ylabel('Probability Density', fontweight='bold')
    ax3.set_title('Similarity Distribution', fontweight='bold', pad=15)
    ax3.legend(frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3)
    
    # 整体标题
    fig.suptitle('Neural Representational Stability Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # 添加统计信息文本框
    anova_p = stability_results['statistical_tests']['anova_p']
    if not np.isnan(anova_p):
        anova_text = f"ANOVA: p = {anova_p:.4f}"
        fig.text(0.02, 0.02, anova_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"表征稳定性可视化结果已保存: {save_path}")
    plt.show()

def save_representational_stability_results(stability_results, condition_name="all_conditions", save_dir=None):
    """
    保存表征稳定性分析结果
    
    参数:
    stability_results: 稳定性分析结果
    condition_name: 条件名称
    save_dir: 保存目录
    """
    if save_dir is None:
        save_dir = acfg.get_results_dir()
    
    filename = f"representational_stability_{condition_name}.npz"
    
    # 准备保存数据
    save_data = {}
    
    # 保存相似度数据
    for condition, similarities in stability_results['trial_template_similarities'].items():
        save_data[f'similarities_condition_{condition}'] = similarities
    
    # 保存统计数据
    for condition, stats in stability_results['similarity_statistics'].items():
        for stat_name, stat_value in stats.items():
            save_data[f'stats_condition_{condition}_{stat_name}'] = stat_value
    
    # 保存统计检验结果
    save_data['anova_f'] = stability_results['statistical_tests']['anova_f']
    save_data['anova_p'] = stability_results['statistical_tests']['anova_p']
    
    # 保存两两比较结果
    for comparison_name, results in stability_results['statistical_tests']['pairwise_comparisons'].items():
        save_data[f'pairwise_{comparison_name}_t'] = results['t_stat']
        save_data[f'pairwise_{comparison_name}_p'] = results['p_value']
    
    # 保存效应量
    for comparison_name, effect_size in stability_results['statistical_tests']['effect_sizes'].items():
        save_data[f'effect_size_{comparison_name}'] = effect_size
    
    # 保存分析参数
    for param_name, param_value in stability_results['analysis_parameters'].items():
        save_data[f'param_{param_name}'] = param_value
    
    np.savez_compressed(
        os.path.join(save_dir, filename),
        **save_data
    )
    print(f"表征稳定性分析结果已保存: {filename}")

# %% 主分析函数

def run_rich_club_analysis_by_condition(segments, labels, rr_neurons):
    """
    按条件分别进行富人俱乐部分析
    
    参数:
    segments: 神经数据片段
    labels: 标签数组
    rr_neurons: RR神经元索引
    
    返回:
    condition_results: 各条件的富人俱乐部分析结果
    """
    print("按条件进行富人俱乐部分析...")
    
    # 过滤有效数据
    valid_mask = labels != 0
    valid_segments = segments[valid_mask][:, rr_neurons, :]
    valid_labels = labels[valid_mask]
    
    condition_results = {}
    unique_labels = np.unique(valid_labels)
    
    for condition in unique_labels:
        print(f"\n--- 分析条件 {condition} ---")
        
        # 提取该条件的数据
        condition_mask = valid_labels == condition
        condition_segments = valid_segments[condition_mask]
        
        if len(condition_segments) < 10:
            print(f"条件 {condition} 试次数不足，跳过")
            continue
        
        # 提取刺激期活动
        stimulus_window = np.arange(acfg.STIMULUS_START,
                                   min(acfg.STIMULUS_START + acfg.STIMULUS_DURATION,
                                       condition_segments.shape[2]))
        neural_activity = np.mean(condition_segments[:, :, stimulus_window], axis=2)
        
        # 构建功能连接网络
        print(f"构建条件 {condition} 的功能连接网络...")
        corr_matrix, p_matrix = compute_correlation_matrix(neural_activity, method='pearson')
        adj_matrix = threshold_correlation_matrix(corr_matrix, p_matrix,
                                                method='density', network_density=0.1)
        
        # 创建网络
        G = nx.from_numpy_array(adj_matrix)
        
        print(f"网络规模: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
        
        # 富人俱乐部分析
        if G.number_of_edges() > 0:
            rich_club_results = calculate_normalized_rich_club_coefficient(G)
            condition_results[condition] = rich_club_results
            
            # 保存结果
            save_rich_club_analysis(rich_club_results, f"condition_{condition}")
        else:
            print(f"条件 {condition} 网络无连接，跳过富人俱乐部分析")
    
    return condition_results

def run_pid_analysis_by_condition(segments, labels, rr_neurons):
    """
    按条件分别进行信息分解分析
    
    参数:
    segments: 神经数据片段
    labels: 标签数组
    rr_neurons: RR神经元索引
    
    返回:
    condition_results: 各条件的信息分解分析结果
    """
    print("进行按条件分别的信息分解（PID）分析...")
    
    # 过滤有效数据
    valid_mask = labels != 0
    valid_segments = segments[valid_mask][:, rr_neurons, :]
    valid_labels = labels[valid_mask]
    
    unique_labels = np.unique(valid_labels)
    print(f"发现标签类别: {unique_labels}")
    
    # 检查每个类别的试次数
    label_counts = {}
    for label in unique_labels:
        count = np.sum(valid_labels == label)
        label_counts[label] = count
        print(f"标签 {label}: {count} 个试次")
    
    condition_results = {}
    
    # 定义要分析的条件对比
    condition_pairs = [
        ([1, 2], 'condition_1_vs_2'),
        ([1, 3], 'condition_1_vs_3'), 
        ([2, 3], 'condition_2_vs_3'),
        ([1, 2, 3], 'all_conditions')  # 保留原始的全条件比较
    ]
    
    for conditions, condition_name in condition_pairs:
        print(f"\n--- 分析 {condition_name} ---")
        
        # 筛选当前条件的数据
        condition_mask = np.isin(valid_labels, conditions)
        available_conditions = [c for c in conditions if c in unique_labels and label_counts[c] >= 5]
        
        if len(available_conditions) < 2:
            print(f"跳过 {condition_name}: 有效条件不足 (需要至少2个条件，每个条件至少5个试次)")
            continue
            
        # 重新筛选数据
        condition_mask = np.isin(valid_labels, available_conditions)
        condition_segments = valid_segments[condition_mask]
        condition_labels = valid_labels[condition_mask]
        
        print(f"使用条件: {available_conditions}")
        print(f"数据维度: {condition_segments.shape}")
        print(f"标签分布: {dict(zip(*np.unique(condition_labels, return_counts=True)))}")
        
        # 执行信息分解分析
        try:
            pid_results = analyze_hub_peripheral_information_dynamics(
                condition_segments,
                condition_labels,
                list(range(len(rr_neurons)))  # 使用连续索引
            )
            
            if pid_results is not None:
                condition_results[condition_name] = pid_results
                
                # 保存结果
                save_pid_analysis(pid_results, condition_name)
                print(f"+ {condition_name} PID分析完成")
            else:
                print(f"X {condition_name} PID分析失败")
                
        except Exception as e:
            print(f"X {condition_name} PID分析出错: {e}")
            continue
    
    return condition_results

# %% 主脚本

def run_advanced_network_analysis():
    """
    运行高级网络分析（富人俱乐部 + 信息分解）
    """
    print("=" * 60)
    print("高级网络分析：富人俱乐部组织 + 信息分解")
    print("=" * 60)
    
    # 确保结果目录存在（使用版本感知路径）
    results_dir = acfg.get_results_dir()
    print(f"结果将保存到: {results_dir}")
    
    # 1. 加载和预处理数据
    print("\n1. 数据加载与预处理")
    print("-" * 30)
    
    # 加载数据
    if cfg.LOADER_VERSION == 'new':
        neuron_data, neuron_pos, trigger_data, stimulus_data = load_data(cfg.DATA_PATH)
        segments, labels = segment_neuron_data(neuron_data, trigger_data, stimulus_data)
        new_labels = reclassify_labels(stimulus_data)
    elif cfg.LOADER_VERSION == 'old':
        from loaddata import load_old_version_data
        neuron_index, segments, new_labels, neuron_pos = load_old_version_data(
            cfg.OLD_VERSION_PATHS['neurons'],
            cfg.OLD_VERSION_PATHS['trials'],
            cfg.OLD_VERSION_PATHS['location']
        )
        # 对于旧版数据，segments和new_labels已经是处理好的格式
        neuron_pos = neuron_pos[0:2, :] if neuron_pos.shape[0] >= 2 else neuron_pos
        print(f"旧版数据维度: segments={segments.shape}, labels={len(new_labels)}, neuron_pos={neuron_pos.shape}")
        print("已切换到旧版数据加载模式")
    else:
        raise ValueError("无效的 LOADER_VERSION 配置")
    
    # RR神经元筛选
    rr_results = fast_rr_selection(segments, new_labels)
    rr_neurons = rr_results['rr_neurons']
    
    print(f"数据维度: {segments.shape}")
    print(f"RR神经元数量: {len(rr_neurons)}")
    print(f"标签分布: {Counter(new_labels)}")
    
    if len(rr_neurons) < 50:
        print("警告: RR神经元数量过少，可能影响分析质量")
    
    # 2. 富人俱乐部分析
    print("\n2. 富人俱乐部组织分析")
    print("-" * 30)
    
    rich_club_results = run_rich_club_analysis_by_condition(segments, new_labels, rr_neurons)
    
    # 3. 信息分解分析
    print("\n3. 部分信息分解（PID）分析")
    print("-" * 30)
    
    pid_results = run_pid_analysis_by_condition(segments, new_labels, rr_neurons)
    
    # 4. 表征稳定性分析
    print("\n4. 表征稳定性分析")
    print("-" * 30)
    
    stability_results = analyze_representational_stability(segments, new_labels, rr_neurons)
    
    # 保存表征稳定性结果
    save_representational_stability_results(stability_results, save_dir=results_dir)
    
    # 5. 可视化分析结果
    print("\n5. 可视化分析结果")
    print("-" * 30)
    
    # 可视化富人俱乐部分析结果
    for condition, results in rich_club_results.items():
        print(f"可视化条件 {condition} 的富人俱乐部分析...")
        visualize_rich_club_results(
            results, 
            condition_name=f"Condition {condition}",
            save_path=os.path.join(results_dir, f'rich_club_condition_{condition}.png')
        )
    
    # 可视化信息分解分析结果
    for condition, results in pid_results.items():
        print(f"可视化条件 {condition} 的信息分解分析...")
        visualize_pid_results(
            results,
            title=f"Partial Information Decomposition - Condition {condition}",
            save_path=os.path.join(results_dir, f'pid_condition_{condition}.png')
        )
    
    # 生成多条件对比可视化
    if len(pid_results) > 1:
        print("生成多条件PID对比可视化...")
        visualize_pid_conditions_comparison(
            pid_results,
            save_path=os.path.join(results_dir, 'pid_conditions_comparison.png')
        )
    
    # 可视化表征稳定性分析结果
    print("生成表征稳定性可视化...")
    visualize_representational_stability(
        stability_results,
        save_path=os.path.join(results_dir, 'representational_stability_analysis.png')
    )
    
    # 6. 综合分析和报告
    print("\n6. 生成综合分析报告")
    print("-" * 30)
    
    report = generate_advanced_analysis_report(rich_club_results, pid_results, stability_results)
    
    # 保存报告
    report_path = os.path.join(results_dir, 'advanced_analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n高级网络分析完成！")
    print(f"结果保存在: {results_dir}")
    
    return {
        'rich_club_results': rich_club_results,
        'pid_results': pid_results,
        'stability_results': stability_results,
        'report': report
    }

def generate_advanced_analysis_report(rich_club_results, pid_results, stability_results=None):
    """
    生成高级分析报告
    """
    report = []
    report.append("=" * 60)
    report.append("高级网络分析报告")
    report.append("富人俱乐部组织 + 部分信息分解（PID） + 表征稳定性")
    report.append("=" * 60)
    report.append("")
    
    # 富人俱乐部分析结果
    report.append("## 1. 富人俱乐部组织分析")
    report.append("")
    
    if rich_club_results:
        report.append(f"分析了 {len(rich_club_results)} 个条件的富人俱乐部组织")
        
        for condition, results in rich_club_results.items():
            report.append(f"\n### 条件 {condition}:")
            
            rich_club_region = results['rich_club_region']
            if rich_club_region:
                report.append(f"- 检测到富人俱乐部结构")
                report.append(f"- 富人俱乐部度值范围: {min(rich_club_region)} - {max(rich_club_region)}")
                
                # 找到最强的富人俱乐部系数
                max_k = max(rich_club_region, key=lambda k: results['normalized_coefficients'][k])
                max_coeff = results['normalized_coefficients'][max_k]
                report.append(f"- 最强富人俱乐部系数: {max_coeff:.3f} (k={max_k})")
            else:
                report.append("- 未检测到显著的富人俱乐部结构")
    else:
        report.append("未进行富人俱乐部分析")
    
    # 信息分解分析结果
    report.append("\n## 2. 部分信息分解（PID）分析")
    report.append("")
    
    if pid_results:
        report.append(f"分析了 {len(pid_results)} 个条件的信息分解")
        
        for condition, results in pid_results.items():
            report.append(f"\n### 条件 {condition}:")
            
            if 'analysis_summary' in results:
                summary = results['analysis_summary']
                report.append(f"- 枢纽神经元数: {summary['n_hubs']}")
                report.append(f"- 边缘神经元数: {summary['n_peripheral']}")
            
            if 'pid_results' in results:
                pid_data = results['pid_results']
                
                # 分析各种神经元组合的信息模式
                for pair_type, pair_results in pid_data.items():
                    redundancy_mean = pair_results.get('redundancy_mean', 0)
                    synergy_mean = pair_results.get('synergy_mean', 0)
                    unique1_mean = pair_results.get('unique_X1_mean', 0)
                    unique2_mean = pair_results.get('unique_X2_mean', 0)
                    
                    report.append(f"- {pair_type.replace('_', '-')} 信息模式:")
                    report.append(f"  * 冗余信息: {redundancy_mean:.4f}")
                    report.append(f"  * 协同信息: {synergy_mean:.4f}")
                    report.append(f"  * 神经元1唯一信息: {unique1_mean:.4f}")
                    report.append(f"  * 神经元2唯一信息: {unique2_mean:.4f}")
                    
                    # 判断信息模式类型
                    if synergy_mean > redundancy_mean * 1.5:
                        info_pattern = "协同主导"
                    elif redundancy_mean > synergy_mean * 1.5:
                        info_pattern = "冗余主导" 
                    else:
                        info_pattern = "平衡模式"
                    
                    report.append(f"  * 信息模式类型: {info_pattern}")
    else:
        report.append("未进行信息分解分析")
    
    # 表征稳定性分析结果
    report.append("\n## 3. 表征稳定性分析")
    report.append("")
    
    if stability_results:
        # 分析参数
        params = stability_results['analysis_parameters']
        report.append(f"分析了 {params['n_conditions']} 个刺激条件的神经表征稳定性")
        report.append(f"使用了 {params['n_rr_neurons']} 个响应可靠神经元")
        report.append(f"特征向量维度: {params['feature_dimension']}")
        report.append("")
        
        # 统计检验结果
        stats = stability_results['statistical_tests']
        if not np.isnan(stats['anova_p']):
            significance_level = "高度显著" if stats['anova_p'] < 0.001 else ("显著" if stats['anova_p'] < 0.05 else "不显著")
            report.append(f"ANOVA检验结果: F = {stats['anova_f']:.4f}, p = {stats['anova_p']:.6f} ({significance_level})")
        
        # 各条件的稳定性统计
        stability_stats = stability_results['similarity_statistics']
        sorted_conditions = sorted(stability_stats.keys(), key=lambda x: stability_stats[x]['mean'], reverse=True)
        
        report.append("\n各条件表征稳定性排序（按平均相似度）:")
        for i, condition in enumerate(sorted_conditions, 1):
            stats_cond = stability_stats[condition]
            report.append(f"{i}. 条件 {condition}: 平均相似度 = {stats_cond['mean']:.4f} ± {stats_cond['std']:.4f}")
            report.append(f"   相似度范围: [{stats_cond['min']:.4f}, {stats_cond['max']:.4f}], 试次数: {stats_cond['n_trials']}")
        
        # 两两比较的显著性
        pairwise = stats['pairwise_comparisons']
        if pairwise:
            report.append("\n条件间差异显著性:")
            for comparison, results in pairwise.items():
                p_val = results['p_value']
                significance = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
                cond1, cond2 = comparison.split('_vs_')
                report.append(f"- 条件 {cond1} vs 条件 {cond2}: p = {p_val:.6f} {significance}")
        
        # 效应量
        effect_sizes = stats['effect_sizes']
        if effect_sizes:
            report.append("\n效应量分析:")
            for comparison, d_value in effect_sizes.items():
                effect_magnitude = "小" if abs(d_value) < 0.5 else ("中" if abs(d_value) < 0.8 else "大")
                cond1, cond2 = comparison.split('_vs_')
                direction = "条件"+cond1+"更稳定" if d_value > 0 else "条件"+cond2+"更稳定"
                report.append(f"- {comparison}: Cohen's d = {d_value:.4f} ({effect_magnitude}效应, {direction})")
        
        # 核心发现解读
        report.append("\n核心发现:")
        if not np.isnan(stats['anova_p']) and stats['anova_p'] < 0.05:
            report.append("- 不同刺激条件在神经表征稳定性上存在显著差异")
            report.append("- 这支持了'结构化刺激诱发更稳定神经活动模式'的核心假说")
            
            # 找到最稳定和最不稳定的条件
            most_stable = max(stability_stats.keys(), key=lambda x: stability_stats[x]['mean'])
            least_stable = min(stability_stats.keys(), key=lambda x: stability_stats[x]['mean'])
            report.append(f"- 条件 {most_stable} 表现出最高的表征稳定性 (相似度: {stability_stats[most_stable]['mean']:.4f})")
            report.append(f"- 条件 {least_stable} 表现出最低的表征稳定性 (相似度: {stability_stats[least_stable]['mean']:.4f})")
        else:
            report.append("- 不同刺激条件的神经表征稳定性无显著差异")
            report.append("- 这可能表明所有条件都能诱发相对稳定的神经活动模式")
        
    else:
        report.append("未进行表征稳定性分析")
    
    # 综合结论
    report.append("\n## 4. 综合结论")
    report.append("")
    
    # 富人俱乐部结论
    if rich_club_results:
        conditions_with_rich_club = [cond for cond, results in rich_club_results.items() 
                                   if results['rich_club_region']]
        
        if conditions_with_rich_club:
            report.append(f"- {len(conditions_with_rich_club)}/{len(rich_club_results)} 个条件显示富人俱乐部组织")
            report.append("- 这表明V1的功能网络中高度连接的枢纽神经元倾向于形成紧密连接的'核心执行委员会'")
        else:
            report.append("- 所有条件均未显示显著的富人俱乐部组织")
            report.append("- 这可能表明网络的层次化组织不够明显，或连接相对均匀分布")
    
    # PID结论
    if pid_results:
        # 统计各种信息模式
        synergy_dominant_count = 0
        redundancy_dominant_count = 0
        
        for condition, results in pid_results.items():
            if 'pid_results' in results:
                for pair_type, pair_results in results['pid_results'].items():
                    synergy = pair_results.get('synergy_mean', 0)
                    redundancy = pair_results.get('redundancy_mean', 0)
                    
                    if synergy > redundancy * 1.5:
                        synergy_dominant_count += 1
                    elif redundancy > synergy * 1.5:
                        redundancy_dominant_count += 1
        
        total_pairs = synergy_dominant_count + redundancy_dominant_count
        if total_pairs > 0:
            if synergy_dominant_count > redundancy_dominant_count:
                report.append("- 神经元对普遍表现出协同信息处理模式")
                report.append("- 这表明不同神经元组合能够产生单独神经元无法提供的新信息")
            elif redundancy_dominant_count > synergy_dominant_count:
                report.append("- 神经元对普遍表现出冗余信息处理模式")
                report.append("- 这表明多个神经元编码相似信息，有助于提高系统的稳健性")
            else:
                report.append("- 神经元对表现出平衡的信息处理模式")
                report.append("- 协同和冗余信息处理并存")
    
    # 表征稳定性结论
    if stability_results:
        stats = stability_results['statistical_tests']
        if not np.isnan(stats['anova_p']) and stats['anova_p'] < 0.05:
            stability_stats = stability_results['similarity_statistics']
            most_stable = max(stability_stats.keys(), key=lambda x: stability_stats[x]['mean'])
            report.append(f"- 表征稳定性分析验证了核心假说，条件 {most_stable} 诱发了最稳定的神经表征")
            report.append("- 不同刺激条件的神经表征稳定性差异反映了V1对不同视觉输入的编码可靠性")
        else:
            report.append("- 所有条件均能诱发相对稳定的神经表征，表明V1编码的高度稳健性")
    
    # 方法学价值
    report.append("\n## 5. 方法学意义")
    report.append("")
    report.append("- 富人俱乐部分析揭示了网络的层次化拓扑组织原理")
    report.append("- 信息分解分析量化了不同神经元组合的信息贡献模式") 
    report.append("- 表征稳定性分析直接验证了神经编码的可靠性假说")
    report.append("- 三种方法的结合提供了从拓扑结构到信息流动再到编码稳定性的完整视角")
    
    return "\n".join(report)

# %% 主程序执行
if __name__ == "__main__":
    print("开始高级网络分析...")
    
    # %% 运行分析
    results = run_advanced_network_analysis()
    
    print("\n分析完成！主要发现:")
    print(f"- 富人俱乐部分析: {len(results['rich_club_results'])} 个条件")
    print(f"- 信息分解分析: {len(results['pid_results'])} 个条件")
    print(f"- 表征稳定性分析: {results['stability_results']['analysis_parameters']['n_conditions']} 个条件")
    if 'stability_results' in results and results['stability_results']:
        stats = results['stability_results']['statistical_tests']
        if not np.isnan(stats['anova_p']):
            significance = "显著差异" if stats['anova_p'] < 0.05 else "无显著差异"
            print(f"  核心假说验证: 条件间表征稳定性 {significance} (p={stats['anova_p']:.4f})")
    # 获取实际的结果目录路径
    actual_results_dir = acfg.get_results_dir()
    print(f"- 详细报告: {actual_results_dir}/advanced_analysis_report.txt")