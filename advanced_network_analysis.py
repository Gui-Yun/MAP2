# 高级网络分析：富人俱乐部与信息分解
# guiy24@mails.tsinghua.edu.cn
# 2025-01-09
# 实现富人俱乐部组织分析和部分信息分解（PID）分析

# %% 导入必要的库
import numpy as np
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
    
    # 结果保存路径
    RESULTS_DIR = 'results/advanced_analysis'
    
    @classmethod
    def ensure_results_dir(cls):
        """确保结果目录存在"""
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)

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

def save_rich_club_analysis(rich_club_results, condition_name, save_dir=acfg.RESULTS_DIR):
    """
    保存富人俱乐部分析结果
    """
    acfg.ensure_results_dir()
    
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

def save_pid_analysis(pid_results, condition_name, save_dir=acfg.RESULTS_DIR):
    """
    保存信息分解分析结果
    """
    acfg.ensure_results_dir()
    
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
    print("按条件进行信息分解（PID）分析...")
    
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
        condition_labels = valid_labels[condition_mask]
        
        if len(condition_segments) < 20:
            print(f"条件 {condition} 试次数不足，跳过")
            continue
        
        # 执行信息分解分析
        pid_results = analyze_hub_peripheral_information_dynamics(
            condition_segments.reshape(len(condition_segments), len(rr_neurons), -1),
            condition_labels,
            list(range(len(rr_neurons)))  # 使用连续索引
        )
        
        if pid_results is not None:
            condition_results[condition] = pid_results
            
            # 保存结果
            save_pid_analysis(pid_results, f"condition_{condition}")
    
    return condition_results

# %% 主脚本

def run_advanced_network_analysis():
    """
    运行高级网络分析（富人俱乐部 + 信息分解）
    """
    print("=" * 60)
    print("高级网络分析：富人俱乐部组织 + 信息分解")
    print("=" * 60)
    
    # 确保结果目录存在
    acfg.ensure_results_dir()
    
    # 1. 加载和预处理数据
    print("\n1. 数据加载与预处理")
    print("-" * 30)
    
    neuron_data, neuron_pos, trigger_data, stimulus_data = load_data(cfg.DATA_PATH)
    segments, labels = segment_neuron_data(neuron_data, trigger_data, stimulus_data)
    new_labels = reclassify_labels(stimulus_data)
    
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
    
    # 4. 可视化分析结果
    print("\n4. 可视化分析结果")
    print("-" * 30)
    
    # 可视化富人俱乐部分析结果
    for condition, results in rich_club_results.items():
        print(f"可视化条件 {condition} 的富人俱乐部分析...")
        visualize_rich_club_results(
            results, 
            condition_name=f"Condition {condition}",
            save_path=os.path.join(acfg.RESULTS_DIR, f'rich_club_condition_{condition}.png')
        )
    
    # 可视化信息分解分析结果
    for condition, results in pid_results.items():
        print(f"可视化条件 {condition} 的信息分解分析...")
        visualize_pid_results(
            results,
            title=f"Partial Information Decomposition - Condition {condition}",
            save_path=os.path.join(acfg.RESULTS_DIR, f'pid_condition_{condition}.png')
        )
    
    # 5. 综合分析和报告
    print("\n5. 生成综合分析报告")
    print("-" * 30)
    
    report = generate_advanced_analysis_report(rich_club_results, pid_results)
    
    # 保存报告
    report_path = os.path.join(acfg.RESULTS_DIR, 'advanced_analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n高级网络分析完成！")
    print(f"结果保存在: {acfg.RESULTS_DIR}")
    
    return {
        'rich_club_results': rich_club_results,
        'pid_results': pid_results,
        'report': report
    }

def generate_advanced_analysis_report(rich_club_results, pid_results):
    """
    生成高级分析报告
    """
    report = []
    report.append("=" * 60)
    report.append("高级网络分析报告")
    report.append("富人俱乐部组织 + 部分信息分解（PID）")
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
    
    # 综合结论
    report.append("\n## 3. 综合结论")
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
    
    # 方法学价值
    report.append("\n## 4. 方法学意义")
    report.append("")
    report.append("- 富人俱乐部分析揭示了网络的层次化拓扑组织原理")
    report.append("- 信息分解分析量化了不同神经元组合的信息贡献模式")
    report.append("- 两种方法的结合提供了从拓扑结构到信息流动的完整视角")
    
    return "\n".join(report)

# %% 主程序执行
if __name__ == "__main__":
    print("开始高级网络分析...")
    
    # 运行分析
    results = run_advanced_network_analysis()
    
    print("\n分析完成！主要发现:")
    print(f"- 富人俱乐部分析: {len(results['rich_club_results'])} 个条件")
    print(f"- 信息分解分析: {len(results['pid_results'])} 个条件") 
    print(f"- 详细报告: {acfg.RESULTS_DIR}/advanced_analysis_report.txt")