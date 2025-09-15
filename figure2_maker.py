
"""
Figure 2 Maker - 功能网络的拓扑结构对刺激敏感

基于 figures.md 设计规范:
  Panel A: 网络的"富人俱乐部"结构 - 带状阴影曲线图
  Panel B: 刺激重塑网络宏观拓扑 - 分组小提琴图  
  Panel C: 刺激重塑网络核心结构 - 配对散点图
  Panel D: 网络拓扑可视化对比 - 网络图

Usage examples:
  python figure2_maker.py --panel A --mouse m27
  python figure2_maker.py --panel B --mouse m27
  python figure2_maker.py --panel C --mouse m27
  python figure2_maker.py --panel D --mouse m27

Notes:
  - 严格按照 advanced_network_analysis.py 的逻辑进行神经元筛选和网络构建
  - 使用专业的科研绘图风格，参考 figures.md 的配色方案
  - 支持新版和旧版数据格式
"""

import os
import json
import glob
import argparse
from typing import Dict, List, Optional, Tuple
import warnings
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats

from loaddata import (
    cfg as global_cfg,
    reclassify_labels,
    fast_rr_selection,
    load_data, 
    segment_neuron_data, 
    load_old_version_data
)

# Import network analysis functions
from network import (
    compute_correlation_matrix,
    threshold_correlation_matrix,
    compute_network_metrics
)

# Import advanced network analysis functions  
from advanced_network_analysis import (
    calculate_rich_club_coefficient,
    calculate_normalized_rich_club_coefficient,
    generate_random_networks,
    AdvancedAnalysisConfig as acfg
)


# ---------------------------
# 专业绘图风格和配色方案
# ---------------------------
def setup_publication_style():
    """设置专业的科研出版物绘图风格，参考 advanced_network_analysis.py"""
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

# 按照 figures.md 设计的专业配色方案
COLORS = {
    # 主要颜色 (figures.md 指定)
    'ordered': '#FF7F50',      # 珊瑚橙 - 有序光流/核心发现
    'noise': '#4682B4',        # 钢青色 - 随机噪音/对照组  
    'neutral': '#708090',      # 石板灰 - 中性/背景
    
    # 网络可视化颜色
    'hub': '#D2691E',          # 赭石色 - 枢纽节点
    'peripheral': '#2E86AB',    # 节点颜色
    'edge': '#6C757D',         # 边的颜色
    'edge_weak': '#95A5A6',    # 弱连接边
    
    # 科学可视化颜色
    'significant': '#F18F01',   # 显著结果
    'threshold': '#E74C3C',     # 阈值线
    'fill': '#95A5A6',         # 填充区域
    'confidence': '#BDC3C7'     # 置信区间
}


def list_config_files() -> List[str]:
    return sorted(glob.glob(os.path.join('config', '*.json')))


def get_config_by_mouse(mouse: Optional[str]) -> Optional[str]:
    if not mouse:
        return None
    key = str(mouse).lower().strip()
    name = None
    if key in {'m27', '27'}:
        name = 'm27.json'
    elif key in {'m30', '30'}:
        name = 'm30.json'
    elif key in {'m65', '65'}:
        name = 'm65.json'
    elif key in {'m74', '74'}:
        name = 'm74.json'
    if not name:
        return None
    path = os.path.join('config', name)
    return path if os.path.exists(path) else None


def load_session_data(config_path: str):
    """
    严格按照 advanced_network_analysis.py 的逻辑加载数据
    """
    print(f"加载会话数据: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg_json = json.load(f)
    
    # 临时更新全局配置
    global_cfg.LOADER_VERSION = cfg_json.get('LOADER_VERSION', 'new')
    if 'DATA_PATH' in cfg_json:
        global_cfg.DATA_PATH = cfg_json['DATA_PATH']
    if 'OLD_VERSION_PATHS' in cfg_json:
        global_cfg.OLD_VERSION_PATHS = cfg_json['OLD_VERSION_PATHS']
    
    loader = global_cfg.LOADER_VERSION
    print(f"使用数据加载器版本: {loader}")
    
    if loader == 'new':
        print("加载新版数据...")
        neuron_data, neuron_pos, trigger_data, stimulus_data = load_data(global_cfg.DATA_PATH)
        segments, labels = segment_neuron_data(neuron_data, trigger_data, stimulus_data)
        new_labels = reclassify_labels(stimulus_data)
        neuron_pos = neuron_pos[0:2, :] if neuron_pos.shape[0] >= 2 else neuron_pos
        
    elif loader == 'old':
        print("加载旧版数据...")
        neuron_index, segments, labels, neuron_pos = load_old_version_data(
            global_cfg.OLD_VERSION_PATHS['neurons'],
            global_cfg.OLD_VERSION_PATHS['trials'],
            global_cfg.OLD_VERSION_PATHS['location']
        )
        # 对于旧版数据，segments和labels已经是处理好的格式
        new_labels = labels  # 旧版数据已经是重分类后的标签
        neuron_pos = neuron_pos[0:2, :] if neuron_pos.shape[0] >= 2 else neuron_pos
        # 构造stimulus_data用于兼容性
        stimulus_data = np.column_stack([labels, np.zeros(len(labels))])
    else:
        raise ValueError(f"无效的 LOADER_VERSION: {loader}")
    
    print(f"数据加载完成:")
    print(f"  segments: {segments.shape}")
    print(f"  labels: {len(new_labels)}")
    print(f"  neuron_pos: {neuron_pos.shape}")
    print(f"  标签分布: {Counter(new_labels)}")
    
    return segments, new_labels, neuron_pos, stimulus_data

def compute_rr_neurons(segments: np.ndarray, labels: np.ndarray) -> List[int]:
    """
    按照 advanced_network_analysis.py 的逻辑筛选RR神经元
    """
    print("筛选RR神经元...")
    rr_results = fast_rr_selection(segments, labels)
    rr_neurons = rr_results.get('rr_neurons', [])
    
    print(f"RR神经元筛选结果:")
    print(f"  响应性神经元: {len(rr_results.get('response_neurons', []))}")
    print(f"  可靠性神经元: {len(rr_results.get('reliable_neurons', []))}")
    print(f"  最终RR神经元: {len(rr_neurons)}")
    
    if len(rr_neurons) < 50:
        print("警告: RR神经元数量较少，可能影响分析质量")
    
    return list(rr_neurons)


def extract_neural_activity_for_condition(segments: np.ndarray, condition_mask: np.ndarray, 
                                          rr_neurons: List[int]) -> np.ndarray:
    """
    按照 advanced_network_analysis.py 的逻辑提取神经活动
    """
    # 过滤条件和RR神经元
    condition_segments = segments[condition_mask][:, rr_neurons, :]
    
    # 提取刺激期活动 - 严格按照 advanced_network_analysis.py 的参数
    stimulus_window = np.arange(acfg.STIMULUS_START,
                               min(acfg.STIMULUS_START + acfg.STIMULUS_DURATION,
                                   condition_segments.shape[2]))
    
    # 计算刺激期平均活动
    neural_activity = np.mean(condition_segments[:, :, stimulus_window], axis=2)
    
    print(f"  刺激时间窗口: {stimulus_window[0]}-{stimulus_window[-1]}")
    print(f"  神经活动矩阵: {neural_activity.shape}")
    
    return neural_activity

def build_functional_network(neural_activity: np.ndarray, network_density: float = 0.1) -> Tuple[np.ndarray, np.ndarray, nx.Graph]:
    """
    构建功能连接网络，参考 advanced_network_analysis.py
    """
    print(f"构建功能连接网络 (密度={network_density})...")
    
    # 计算相关性矩阵
    corr_matrix, p_matrix = compute_correlation_matrix(neural_activity, method='pearson')
    print(f"  相关系数范围: {np.min(corr_matrix):.3f} ~ {np.max(corr_matrix):.3f}")
    
    # 阈值化处理
    adj_matrix = threshold_correlation_matrix(corr_matrix, p_matrix,
                                            method='density', network_density=network_density)
    
    # 创建网络图
    G = nx.from_numpy_array(adj_matrix)
    print(f"  网络规模: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    print(f"  实际密度: {nx.density(G):.4f}")
    
    return corr_matrix, adj_matrix, G

def calculate_network_metrics(adj_matrix: np.ndarray) -> Dict[str, float]:
    """计算网络指标"""
    metrics = compute_network_metrics(adj_matrix)
    return metrics

def calculate_rich_club_analysis(G: nx.Graph) -> Dict:
    """
    计算富人俱乐部分析，使用 advanced_network_analysis.py 的归一化方法
    """
    print("计算归一化富人俱乐部系数...")
    
    if G.number_of_edges() == 0:
        print("网络无连接，跳过富人俱乐部分析")
        return {'k_values': [], 'coefficients': [], 'normalized_coefficients': []}
    
    # 使用 advanced_network_analysis.py 中的完整归一化方法
    rich_club_results = calculate_normalized_rich_club_coefficient(G)
    
    k_values = rich_club_results['k_values']
    coefficients = [rich_club_results['original_coefficients'][k] for k in k_values]
    normalized_coefficients = [rich_club_results['normalized_coefficients'][k] for k in k_values]
    
    print(f"  度值范围: {min(k_values)} - {max(k_values)}")
    print(f"  原始系数范围: {min(coefficients):.4f} - {max(coefficients):.4f}")
    print(f"  归一化系数范围: {min(normalized_coefficients):.4f} - {max(normalized_coefficients):.4f}")
    
    # 检查是否有显著的富人俱乐部结构
    rich_club_region = rich_club_results.get('rich_club_region', [])
    if rich_club_region:
        print(f"  检测到富人俱乐部结构: k ∈ {rich_club_region}")
    else:
        print("  未检测到显著的富人俱乐部结构")
    
    return {
        'k_values': k_values,
        'coefficients': coefficients,
        'normalized_coefficients': normalized_coefficients,
        'rich_club_region': rich_club_region
    }


def build_condition_masks(labels: np.ndarray) -> Dict[str, np.ndarray]:
    """
    构建条件掩码，按照重分类后的标签
    标签含义: 1=有序, 2=噪音, 3=特殊条件
    """
    print("构建条件掩码...")
    masks = {}
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        if label == 0:  # 跳过无效标签
            continue
            
        mask = (labels == label)
        count = np.sum(mask)
        
        if label == 1:
            masks['ordered'] = mask
            print(f"  有序条件: {count} 个试次")
        elif label == 2:
            masks['noise'] = mask  
            print(f"  噪音条件: {count} 个试次")
        elif label == 3:
            masks['special'] = mask
            print(f"  特殊条件: {count} 个试次")
        else:
            masks[f'condition_{label}'] = mask
            print(f"  条件 {label}: {count} 个试次")
    
    return masks


def build_correlation_network(neural_activity: np.ndarray, method: str = 'pearson') -> Tuple[np.ndarray, np.ndarray]:
    """构建相关网络，使用 network.py 的函数"""
    return compute_correlation_matrix(neural_activity, method=method)


def build_adjacency_matrix(corr_matrix: np.ndarray, p_matrix: np.ndarray, density: float = 0.1) -> np.ndarray:
    """构建邻接矩阵，使用 network.py 的函数"""
    return threshold_correlation_matrix(corr_matrix, p_matrix, method='density', network_density=density)


# ---------------------------
# 网络可视化函数 - 从 network.py 移植并统一风格
# ---------------------------

def visualize_correlation_matrix_compact(corr_matrix, title="Functional Connectivity", ax=None):
    """紧凑版功能连接矩阵可视化，适合作为子图"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    # 计算统计信息
    triu_indices = np.triu_indices_from(corr_matrix, k=1)
    upper_values = corr_matrix[triu_indices]
    
    # 绘制矩阵
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, 
                   aspect='auto', interpolation='nearest')
    
    # 设置标题和标签
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Neuron Index', fontweight='bold')
    ax.set_ylabel('Neuron Index', fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Correlation', rotation=270, labelpad=15, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)
    
    # 添加统计信息
    n_connections = len(upper_values)
    mean_corr = np.mean(upper_values)
    std_corr = np.std(upper_values)
    
    stats_text = f'n = {n_connections}\nμ = {mean_corr:.3f}\nσ = {std_corr:.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9, 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
            verticalalignment='top')
    
    return ax


def visualize_adjacency_matrix_compact(adj_matrix, title="Network Adjacency", ax=None):
    """紧凑版邻接矩阵可视化"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    # 计算网络基本统计
    n_nodes = adj_matrix.shape[0]
    n_edges = np.sum(adj_matrix > 0) // 2
    density = n_edges / ((n_nodes * (n_nodes - 1)) / 2) if n_nodes > 1 else 0
    
    # 绘制邻接矩阵
    im = ax.imshow(adj_matrix, cmap='viridis', aspect='auto', interpolation='nearest')
    
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Neuron Index', fontweight='bold')
    ax.set_ylabel('Neuron Index', fontweight='bold')
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Connection', rotation=270, labelpad=15, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)
    
    # 网络统计信息
    network_type = 'Binary' if np.max(adj_matrix) <= 1 else 'Weighted'
    info_text = f'{network_type}\n{n_nodes} nodes\n{n_edges} edges\nρ = {density:.3f}'
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
            verticalalignment='top')
    
    return ax


def visualize_degree_distribution_compact(adj_matrix, title="Degree Distribution", ax=None):
    """紧凑版度分布可视化"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    # 计算度
    G = nx.from_numpy_array(adj_matrix)
    degrees = np.array([d for n, d in G.degree()])
    
    if len(degrees) == 0 or np.max(degrees) == 0:
        ax.text(0.5, 0.5, 'No Connections', ha='center', va='center',
                transform=ax.transAxes, fontsize=16, fontweight='bold',
                color=COLORS['neutral'])
        ax.set_title(title, fontsize=12, fontweight='bold')
        return ax
    
    # 绘制直方图
    n_bins = min(15, len(np.unique(degrees)))
    n, bins, patches = ax.hist(degrees, bins=n_bins, alpha=0.7, 
                              color=COLORS['ordered'], edgecolor='white', linewidth=1)
    
    # 统计线
    mean_degree = np.mean(degrees)
    ax.axvline(mean_degree, color=COLORS['significant'], linestyle='--', 
               linewidth=2, alpha=0.8, label=f'Mean: {mean_degree:.1f}')
    
    ax.set_xlabel('Degree', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.legend(frameon=True, fancybox=True)
    
    # 添加统计信息
    stats_text = (f'Max: {np.max(degrees)}\n'
                 f'Std: {np.std(degrees):.1f}\n'
                 f'Density: {nx.density(G):.3f}')
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8),
            verticalalignment='top', horizontalalignment='right')
    
    return ax


def visualize_connectivity_comparison(corr_matrix_ordered, adj_matrix_ordered,
                                    corr_matrix_noise, adj_matrix_noise,
                                    title="Connectivity Comparison", ax=None):
    """连接性对比可视化"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # 获取上三角数据
    triu_indices_ord = np.triu_indices_from(corr_matrix_ordered, k=1)
    triu_indices_noise = np.triu_indices_from(corr_matrix_noise, k=1)
    
    corr_ordered = corr_matrix_ordered[triu_indices_ord]
    adj_ordered = adj_matrix_ordered[triu_indices_ord]
    
    corr_noise = corr_matrix_noise[triu_indices_noise]
    adj_noise = adj_matrix_noise[triu_indices_noise]
    
    # 保留的连接
    connected_ord = adj_ordered != 0
    connected_noise = adj_noise != 0
    
    # 散点图对比
    ax.scatter(corr_ordered[~connected_ord], np.zeros(np.sum(~connected_ord)), 
               alpha=0.3, s=8, color=COLORS['neutral'], label='Removed (Ordered)')
    
    ax.scatter(corr_noise[~connected_noise], np.ones(np.sum(~connected_noise)), 
               alpha=0.3, s=8, color=COLORS['neutral'])
    
    ax.scatter(corr_ordered[connected_ord], np.zeros(np.sum(connected_ord)), 
               alpha=0.7, s=15, color=COLORS['ordered'], label='Kept (Ordered)')
    
    ax.scatter(corr_noise[connected_noise], np.ones(np.sum(connected_noise)), 
               alpha=0.7, s=15, color=COLORS['noise'], label='Kept (Noise)')
    
    ax.set_xlabel('Original Correlation', fontweight='bold')
    ax.set_ylabel('Condition', fontweight='bold')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Ordered', 'Noise'])
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.legend(frameon=True, fancybox=True, fontsize=10)
    
    return ax


# ---------------------------
# Panel 函数 - 严格按照 figures.md 设计规范
# ---------------------------

def panel_a(output_dir: str, cfg_path: str, density: float = 0.1):
    """
    Panel A: 网络的"富人俱乐部"结构 - 带状阴影曲线图
    X轴为度(k)，Y轴为归一化富人俱乐部系数(Φ_norm)
    """
    setup_publication_style()
    
    print("=" * 50)
    print("生成 Panel A: 富人俱乐部结构分析")
    print("=" * 50)
    
    # 加载数据
    segments, labels, neuron_pos, stimulus_data = load_session_data(cfg_path)
    rr_neurons = compute_rr_neurons(segments, labels)
    
    if len(rr_neurons) == 0:
        print("错误: 无RR神经元，无法进行分析")
        return

    # 构建条件掩码
    masks = build_condition_masks(labels)
    
    # 检查是否有有序和噪音条件
    if 'ordered' not in masks or 'noise' not in masks:
        print("错误: 缺少有序或噪音条件数据")
        return

    print(f"\n分析有序和噪音条件的富人俱乐部结构...")
    
    # 分析两种条件
    condition_results = {}
    
    for condition_name, condition_mask in [('ordered', masks['ordered']), ('noise', masks['noise'])]:
        print(f"\n--- 分析 {condition_name} 条件 ---")
        
        # 提取神经活动
        neural_activity = extract_neural_activity_for_condition(segments, condition_mask, rr_neurons)
        
        # 构建功能网络
        corr_matrix, adj_matrix, G = build_functional_network(neural_activity, density)
        
        # 富人俱乐部分析
        rich_club_results = calculate_rich_club_analysis(G)
        condition_results[condition_name] = rich_club_results
    
    # 绘制富人俱乐部曲线
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for condition_name, results in condition_results.items():
        if not results['k_values']:
            continue
            
        k_values = results['k_values']
        coefficients = results['normalized_coefficients']
        
        color = COLORS['ordered'] if condition_name == 'ordered' else COLORS['noise']
        label = 'Ordered' if condition_name == 'ordered' else 'Noise'
        
        # 主曲线
        ax.plot(k_values, coefficients, 'o-', color=color, linewidth=3, 
               markersize=6, alpha=0.9, label=label,
               markerfacecolor='white', markeredgewidth=2)
    
    # 添加显著性阈值线
    ax.axhline(y=1.0, color=COLORS['threshold'], linestyle='--', 
              linewidth=2, alpha=0.8, label='Rich-Club Threshold')
    
    # 标记富人俱乐部区域
    for condition_name, results in condition_results.items():
        if results.get('rich_club_region'):
            rich_club_region = results['rich_club_region']
            # 高亮富人俱乐部区域的点
            for k in rich_club_region:
                if k in results['k_values']:
                    k_idx = results['k_values'].index(k)
                    coeff = results['normalized_coefficients'][k_idx]
                    color = COLORS['ordered'] if condition_name == 'ordered' else COLORS['noise']
                    ax.plot(k, coeff, 'o', color=color, markersize=10, 
                           alpha=0.8, markeredgecolor='white', markeredgewidth=2)
    
    # 添加富人俱乐部阈值区域
    y_max = ax.get_ylim()[1]
    if y_max > 1.0:
        ax.axhspan(1.0, y_max, alpha=0.1, color=COLORS['significant'], 
                  label='Rich-Club Regime')
    
    ax.set_xlabel('Degree k', fontweight='bold')
    ax.set_ylabel('Normalized Rich Club Coefficient φ_norm(k)', fontweight='bold')
    ax.set_title('Figure 2A. Rich Club Organization', fontsize=14, fontweight='bold', pad=15)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'figure2_panel_a.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'[已保存] {out_path}')


def panel_b(output_dir: str, cfg_path: str, density: float = 0.1):
    """
    Panel B: 刺激重塑网络宏观拓扑 - 分组小提琴图
    X轴为网络指标（网络密度、聚类系数），分组对比有序vs噪音
    """
    setup_publication_style()
    
    print("=" * 50)
    print("生成 Panel B: 网络宏观拓扑对比")
    print("=" * 50)
    
    # 加载数据
    segments, labels, neuron_pos, stimulus_data = load_session_data(cfg_path)
    rr_neurons = compute_rr_neurons(segments, labels)
    
    if len(rr_neurons) == 0:
        print("错误: 无RR神经元，无法进行分析")
        return
    
    # 构建条件掩码
    masks = build_condition_masks(labels)
    
    if 'ordered' not in masks or 'noise' not in masks:
        print("错误: 缺少有序或噪音条件数据")
        return
    
    print("计算网络拓扑指标...")
    
    # 计算每个条件的网络指标
    metrics_data = []
    
    for condition_name, condition_mask in [('ordered', masks['ordered']), ('noise', masks['noise'])]:
        print(f"\n--- 计算 {condition_name} 条件指标 ---")
        
        # 提取神经活动
        neural_activity = extract_neural_activity_for_condition(segments, condition_mask, rr_neurons)
        
        # 构建功能网络
        corr_matrix, adj_matrix, G = build_functional_network(neural_activity, density)
        
        # 计算网络指标
        metrics = calculate_network_metrics(adj_matrix)
        
        # 添加到数据列表
        group_name = 'Ordered' if condition_name == 'ordered' else 'Noise'
        metrics_data.append({'group': group_name, 'metric': 'Network Density', 'value': metrics['density']})
        metrics_data.append({'group': group_name, 'metric': 'Avg Clustering', 'value': metrics['avg_clustering']})
        
        print(f"  网络密度: {metrics['density']:.4f}")
        print(f"  平均聚类: {metrics['avg_clustering']:.4f}")
    
    # 创建DataFrame用于绘图
    import pandas as pd
    df = pd.DataFrame(metrics_data)
    
    # 绘制分组小提琴图
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 使用seaborn绘制小提琴图，内嵌箱线图
    sns.violinplot(data=df, x='metric', y='value', hue='group', 
                   palette=[COLORS['ordered'], COLORS['noise']], 
                   inner='box', ax=ax, alpha=0.8)
    
    ax.set_xlabel('Network Metrics', fontweight='bold')
    ax.set_ylabel('Metric Value', fontweight='bold')
    ax.set_title('Figure 2B. Network Topology Remodeling', fontsize=14, fontweight='bold', pad=15)
    ax.legend(title='Stimulus Type', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'figure2_panel_b.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'[已保存] {out_path}')


def panel_c(output_dir: str, cfg_path: str, density: float = 0.1):
    """
    Panel C: 刺激重塑网络核心结构 - 配对散点图
    X轴为刺激类型（有序 vs. 噪音），Y轴为峰值富人俱乐部系数
    """
    setup_publication_style()
    
    print("=" * 50) 
    print("生成 Panel C: 富人俱乐部系数配对比较")
    print("=" * 50)
    
    # 加载数据
    segments, labels, neuron_pos, stimulus_data = load_session_data(cfg_path)
    rr_neurons = compute_rr_neurons(segments, labels)
    
    if len(rr_neurons) == 0:
        print("错误: 无RR神经元，无法进行分析")
        return
    
    # 构建条件掩码
    masks = build_condition_masks(labels)
    
    if 'ordered' not in masks or 'noise' not in masks:
        print("错误: 缺少有序或噪音条件数据")
        return
    
    print("计算峰值富人俱乐部系数...")
    
    # 计算两个条件的峰值富人俱乐部系数
    peak_coefficients = {}
    
    for condition_name, condition_mask in [('ordered', masks['ordered']), ('noise', masks['noise'])]:
        print(f"\n--- 计算 {condition_name} 条件 ---")
        
        # 提取神经活动
        neural_activity = extract_neural_activity_for_condition(segments, condition_mask, rr_neurons)
        
        # 构建功能网络
        corr_matrix, adj_matrix, G = build_functional_network(neural_activity, density)
        
        # 富人俱乐部分析
        rich_club_results = calculate_rich_club_analysis(G)
        
        # 计算峰值归一化系数
        if rich_club_results['normalized_coefficients']:
            peak_coeff = max(rich_club_results['normalized_coefficients'])
        else:
            peak_coeff = 0.0
            
        peak_coefficients[condition_name] = peak_coeff
        print(f"  峰值富人俱乐部系数: {peak_coeff:.4f}")
    
    # 绘制配对散点图
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # 准备数据点 
    x_positions = [0, 1]
    y_values = [peak_coefficients['ordered'], peak_coefficients['noise']]
    colors = [COLORS['ordered'], COLORS['noise']]
    labels_text = ['Ordered', 'Noise']
    
    # 绘制连接线（显示变化趋势）
    ax.plot(x_positions, y_values, color=COLORS['neutral'], 
           alpha=0.6, linewidth=2, linestyle='-')
    
    # 绘制散点
    ax.scatter(x_positions, y_values, s=120, c=colors, 
              edgecolors='white', linewidths=2, alpha=0.8, zorder=5)
    
    # 设置坐标轴
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels_text, fontweight='bold')
    ax.set_ylabel('Peak Normalized Rich-Club Coefficient', fontweight='bold')
    ax.set_title('Figure 2C. Core Structure Remodeling', fontsize=14, fontweight='bold', pad=15)
    
    # 添加数值标签
    for i, (x, y) in enumerate(zip(x_positions, y_values)):
        ax.text(x, y + max(y_values) * 0.05, f'{y:.3f}', 
               ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0, top=max(y_values) * 1.2)
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'figure2_panel_c.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'[已保存] {out_path}')


def panel_d(output_dir: str, cfg_path: str, density: float = 0.1, hub_top_percent: float = 0.85):
    """
    Panel D: 有序网络拓扑可视化 (简洁版，适合拼图)
    """
    setup_publication_style()
    
    print("=" * 50)
    print("生成 Panel D: 有序网络拓扑可视化")
    print("=" * 50)
    
    # 加载数据
    segments, labels, neuron_pos, stimulus_data = load_session_data(cfg_path)
    rr_neurons = compute_rr_neurons(segments, labels)
    
    if len(rr_neurons) == 0:
        print("未找到RR神经元，使用所有神经元")
        rr_neurons = list(range(segments.shape[1]))
    
    # 构建条件掩码，只需要有序条件
    masks = build_condition_masks(labels)
    
    if 'ordered' not in masks:
        print("未找到有序条件数据")
        return

    print("构建有序网络...")
    
    # 提取神经活动（有序条件）
    neural_activity = extract_neural_activity_for_condition(segments, masks['ordered'], rr_neurons)
    
    # 构建功能网络
    corr_matrix, adj_matrix, G = build_functional_network(neural_activity, density)
    
    # 简洁的单图布局，正方形适合拼图
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # 使用真实神经元物理位置
    if neuron_pos is not None and neuron_pos.shape[0] >= 2 and len(rr_neurons) >= G.number_of_nodes():
        pos = {i: (neuron_pos[0, rr_neurons[i]], neuron_pos[1, rr_neurons[i]]) 
               for i in range(G.number_of_nodes())}
        print(f"使用真实神经元物理位置 ({G.number_of_nodes()} 个节点)")
    else:
        pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
        print("使用spring布局算法")
    
    # 计算度中心性
    degrees = dict(G.degree())
    degree_values = np.array(list(degrees.values()))
    
    if len(degree_values) > 0:
        # 识别枢纽节点
        hub_threshold = np.quantile(degree_values, hub_top_percent)
        hub_nodes = [n for n, d in degrees.items() if d >= hub_threshold]
        peripheral_nodes = [n for n in G.nodes() if n not in hub_nodes]
        
        # 绘制网络 - 简洁风格
        # 边（最先绘制，在节点下方）
        nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.3, 
                             edge_color='#666666', ax=ax)
        
        # 外围节点
        if peripheral_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=peripheral_nodes, 
                                 node_color=COLORS['peripheral'], node_size=12, 
                                 alpha=0.7, ax=ax, edgecolors='none')
        
        # 枢纽节点  
        if hub_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=hub_nodes,
                                 node_color=COLORS['hub'], node_size=25,
                                 alpha=0.9, ax=ax, edgecolors='white', linewidths=0.8)
    
    # 极简样式设置
    ax.set_aspect('equal')
        ax.axis('off')

    # 去掉所有标题和注解，纯净的网络图
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'figure2_panel_d.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', 
                pad_inches=0.02)  # 极小边距
    plt.close(fig)
    print(f'[已保存] {out_path}')


def panel_e(output_dir: str, cfg_path: str, density: float = 0.1):
    """
    Panel E: 网络模块化分析
    1×3子图：聚类相关矩阵 + 聚类邻接矩阵 + 度分布
    """
    setup_publication_style()
    
    print("=" * 50)
    print("生成 Panel E: 网络模块化分析")
    print("=" * 50)
    
    # 加载数据
    segments, labels, neuron_pos, stimulus_data = load_session_data(cfg_path)
    rr_neurons = compute_rr_neurons(segments, labels)
    
    if len(rr_neurons) == 0:
        print("错误: 无RR神经元，无法进行分析")
        return
    
    # 构建条件掩码
    masks = build_condition_masks(labels)
    
    if 'ordered' not in masks:
        print("错误: 缺少有序条件数据")
        return
    
    print("构建功能连接网络...")
    
    # 只使用有序条件
    neural_activity = extract_neural_activity_for_condition(segments, masks['ordered'], rr_neurons)
    
    # 构建相关矩阵和邻接矩阵
    corr_matrix, p_matrix = build_correlation_network(neural_activity, method='pearson')
    adj_matrix = build_adjacency_matrix(corr_matrix, p_matrix, density=density)
    
    # 网络模块检测
    print("进行网络模块检测...")
    G = nx.from_numpy_array(adj_matrix)
    communities = nx.community.greedy_modularity_communities(G)
    
    # 过滤掉小模块（<10个神经元）
    min_module_size = 10
    large_communities = [community for community in communities if len(community) >= min_module_size]
    
    print(f"初始检测到 {len(communities)} 个模块")
    print(f"过滤后保留 {len(large_communities)} 个主要模块（≥{min_module_size}个神经元）")
    
    # 打印主要模块信息
    for i, community in enumerate(large_communities):
        print(f"  模块 {i+1}: {len(community)} 个神经元")
    
    # 创建模块标签（只对大模块）
    module_labels = np.full(adj_matrix.shape[0], -1)  # -1表示不属于任何主要模块
    node_to_module = {}
    
    for i, community in enumerate(large_communities):
        for node in community:
            module_labels[node] = i
            node_to_module[node] = i
    
    # 按模块重新排序（主要模块的节点排在前面）
    # 先按模块分组，模块内部按节点编号排序
    sorted_indices = []
    
    # 添加属于主要模块的节点
    for i, community in enumerate(large_communities):
        community_nodes = sorted(list(community))
        sorted_indices.extend(community_nodes)
    
    # 添加不属于主要模块的节点
    remaining_nodes = [j for j in range(adj_matrix.shape[0]) if module_labels[j] == -1]
    remaining_nodes.sort()
    sorted_indices.extend(remaining_nodes)
    
    sorted_indices = np.array(sorted_indices)
    corr_matrix_sorted = corr_matrix[sorted_indices][:, sorted_indices]
    adj_matrix_sorted = adj_matrix[sorted_indices][:, sorted_indices]
    
    # 计算主要模块边界
    module_boundaries = []
    current_pos = 0
    for community in large_communities:
        current_pos += len(community)
        module_boundaries.append(current_pos)
    
    n_major_modules = len(large_communities)
    total_nodes_in_major_modules = sum(len(c) for c in large_communities)
    minor_module_nodes = adj_matrix.shape[0] - total_nodes_in_major_modules
    
    print(f"主要模块节点: {total_nodes_in_major_modules}, 其他节点: {minor_module_nodes}")
    
    # 创建1×3子图布局
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 子图A: 模块化相关矩阵
    ax1 = axes[0]
    im1 = ax1.imshow(corr_matrix_sorted, cmap='RdBu_r', vmin=-1, vmax=1, 
                     aspect='auto', interpolation='nearest')
    
    # 添加主要模块分割线
    for boundary in module_boundaries[:-1]:
        ax1.axhline(boundary-0.5, color='white', linewidth=2)
        ax1.axvline(boundary-0.5, color='white', linewidth=2)
    
    # 添加主要模块区域与其他节点的分割线
    if total_nodes_in_major_modules < adj_matrix.shape[0]:
        ax1.axhline(total_nodes_in_major_modules-0.5, color='yellow', linewidth=3, alpha=0.8)
        ax1.axvline(total_nodes_in_major_modules-0.5, color='yellow', linewidth=3, alpha=0.8)
    
    ax1.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Neuron Index')
    ax1.set_ylabel('Neuron Index')
    
    # 颜色条
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Correlation', rotation=270, labelpad=15)
    
    # 子图B: 模块化邻接矩阵
    ax2 = axes[1]
    im2 = ax2.imshow(adj_matrix_sorted, cmap='viridis', aspect='auto', interpolation='nearest')
    
    # 添加主要模块分割线
    for boundary in module_boundaries[:-1]:
        ax2.axhline(boundary-0.5, color='red', linewidth=2)
        ax2.axvline(boundary-0.5, color='red', linewidth=2)
    
    # 添加主要模块区域与其他节点的分割线
    if total_nodes_in_major_modules < adj_matrix.shape[0]:
        ax2.axhline(total_nodes_in_major_modules-0.5, color='yellow', linewidth=3, alpha=0.8)
        ax2.axvline(total_nodes_in_major_modules-0.5, color='yellow', linewidth=3, alpha=0.8)
    
    ax2.set_title('Adjacency Matrix', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Neuron Index')
    ax2.set_ylabel('Neuron Index')
    
    # 颜色条
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Connection', rotation=270, labelpad=15)
    
    # 子图C: 度分布
    ax3 = axes[2]
    degrees = np.array([d for n, d in G.degree()])
    
    n_bins = min(15, len(np.unique(degrees)))
    n, bins, patches = ax3.hist(degrees, bins=n_bins, alpha=0.7, 
                              color=COLORS['ordered'], edgecolor='white', linewidth=1)
    
    mean_degree = np.mean(degrees)
    ax3.axvline(mean_degree, color=COLORS['significant'], linestyle='--', 
               linewidth=2, alpha=0.8, label=f'Mean: {mean_degree:.1f}')
    
    ax3.set_xlabel('Degree', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title('Degree Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    
    # 网络统计信息
    n_edges = G.number_of_edges()
    network_density = nx.density(G)
    
    # 计算模块度（使用所有原始模块，包括小模块）
    try:
        modularity = nx.community.modularity(G, communities)
    except:
        # 如果计算失败，使用大模块的覆盖率作为替代指标
        modularity = total_nodes_in_major_modules / adj_matrix.shape[0]
    
    stats_text = (f'Nodes: {adj_matrix.shape[0]}\n'
                 f'Edges: {n_edges}\n'
                 f'Major Modules: {n_major_modules}\n'
                 f'Modularity: {modularity:.3f}')
    
    ax3.text(0.98, 0.98, stats_text, transform=ax3.transAxes, fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8),
            verticalalignment='top', horizontalalignment='right')
    
    # 无整体标题，布局调整
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'figure2_panel_e.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Panel E 保存完成: {save_path}")
    print(f"网络统计: {adj_matrix.shape[0]} 节点, {n_edges} 边, {n_major_modules} 主要模块, 模块度 {modularity:.3f}")


def panel_f(output_dir: str, density: float = 0.1):
    """
    Panel F: 跨动物富人俱乐部分析对比
    Figure 3C: 刺激结构重塑了V1网络核心的组织强度
    分组柱状图 + 散点图，展示四只动物的平均归一化富人俱乐部系数
    (计算富人俱乐部区间Φ_norm > 1的平均值)
    """
    setup_publication_style()
    
    print("=" * 60)
    print("生成 Panel F: 跨动物富人俱乐部分析对比")
    print("=" * 60)
    
    # 四只动物的配置
    mice_configs = {
        'm27': 'config/m27.json',
        'm30': 'config/m30.json', 
        'm65': 'config/m65.json',
        'm74': 'config/m74.json'
    }
    
    # 存储每只动物的结果
    results = {
        'ordered': [],    # 有序条件的平均Φ_norm
        'noise': [],      # 噪音条件的平均Φ_norm
        'mice': []        # 动物标识
    }
    
    print("分析四只动物的富人俱乐部数据...")
    
    for mouse_id, cfg_path in mice_configs.items():
        print(f"\\n--- 分析 {mouse_id} ---")
        
        if not os.path.exists(cfg_path):
            print(f"  配置文件不存在: {cfg_path}")
            continue
            
        try:
            # 加载数据
            segments, labels, neuron_pos, stimulus_data = load_session_data(cfg_path)
            rr_neurons = compute_rr_neurons(segments, labels)
            
            if len(rr_neurons) == 0:
                print(f"  {mouse_id}: 无RR神经元，跳过")
                continue
            
            # 构建条件掩码
            masks = build_condition_masks(labels)
            
            mean_values = {}
            
            # 分析两个条件
            for condition_name, condition_key in [('ordered', 'ordered'), ('noise', 'noise')]:
                if condition_key not in masks:
                    print(f"  {mouse_id}: 缺少{condition_name}条件数据")
                    mean_values[condition_name] = np.nan
                    continue
                
                # 提取神经活动
                neural_activity = extract_neural_activity_for_condition(segments, masks[condition_key], rr_neurons)
                
                # 构建网络
                corr_matrix, p_matrix = build_correlation_network(neural_activity, method='pearson')
                adj_matrix = build_adjacency_matrix(corr_matrix, p_matrix, density=density)
                
                # 富人俱乐部分析
                G = nx.from_numpy_array(adj_matrix)
                rich_club_results = calculate_rich_club_analysis(G)
                
                # 计算富人俱乐部系数的平均值
                if rich_club_results['normalized_coefficients']:
                    normalized_coeffs = np.array(rich_club_results['normalized_coefficients'])
                    
                    # 方法1: 所有k值的平均值
                    mean_phi_norm_all = np.mean(normalized_coeffs)
                    
                    # 方法2: 富人俱乐部区间(Φ_norm > 1)的平均值
                    rich_club_mask = normalized_coeffs > 1.0
                    if np.any(rich_club_mask):
                        mean_phi_norm_rich = np.mean(normalized_coeffs[rich_club_mask])
                        n_rich_points = np.sum(rich_club_mask)
                    else:
                        mean_phi_norm_rich = np.nan
                        n_rich_points = 0
                else:
                    mean_phi_norm_all = np.nan
                    mean_phi_norm_rich = np.nan
                    n_rich_points = 0
                
                # 使用富人俱乐部区间的平均值（如果存在），否则使用全体平均值
                if not np.isnan(mean_phi_norm_rich):
                    final_value = mean_phi_norm_rich
                    method_used = f"富人俱乐部区间均值({n_rich_points}点)"
                else:
                    final_value = mean_phi_norm_all
                    method_used = "全体均值"
                
                mean_values[condition_name] = final_value
                print(f"  {mouse_id} {condition_name}: {method_used} Φ_norm = {final_value:.3f}")
            
            # 只有当两个条件都有有效数据时才加入结果
            if not np.isnan(mean_values['ordered']) and not np.isnan(mean_values['noise']):
                results['ordered'].append(mean_values['ordered'])
                results['noise'].append(mean_values['noise'])
                results['mice'].append(mouse_id)
                
        except Exception as e:
            print(f"  {mouse_id}: 分析失败 - {str(e)}")
            continue
    
    # 检查是否有足够的数据
    if len(results['mice']) < 2:
        print("错误: 至少需要2只动物的有效数据")
        return
        
    print(f"\\n成功分析了 {len(results['mice'])} 只动物: {results['mice']}")
    
    # 转换为numpy数组
    ordered_values = np.array(results['ordered'])
    noise_values = np.array(results['noise'])
    
    # 计算统计量
    ordered_mean = np.mean(ordered_values)
    ordered_sem = np.std(ordered_values) / np.sqrt(len(ordered_values))
    noise_mean = np.mean(noise_values)
    noise_sem = np.std(noise_values) / np.sqrt(len(noise_values))
    
    print(f"\\n统计结果:")
    print(f"  有序条件: {ordered_mean:.3f} ± {ordered_sem:.3f}")
    print(f"  噪音条件: {noise_mean:.3f} ± {noise_sem:.3f}")
    
    # 统计检验
    from scipy import stats
    
    # Wilcoxon Signed-Rank Test (配对非参数检验)
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(ordered_values, noise_values, alternative='two-sided')
    
    # Permutation Test (置换检验)
    def permutation_test(x, y, n_permutations=10000):
        """执行配对样本的置换检验"""
        observed_diff = np.mean(x) - np.mean(y)
        differences = x - y  # 配对差值
        
        # 生成置换分布
        perm_diffs = []
        np.random.seed(42)  # 保证可重复性
        
        for _ in range(n_permutations):
            # 对每个配对的差值随机翻转符号
            signs = np.random.choice([-1, 1], size=len(differences))
            perm_diff = np.mean(differences * signs)
            perm_diffs.append(perm_diff)
        
        # 计算p值 (双尾检验)
        perm_diffs = np.array(perm_diffs)
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
        
        return observed_diff, p_value
    
    perm_diff, perm_p = permutation_test(ordered_values, noise_values)
    
    print(f"\\n统计检验结果:")
    print(f"  Wilcoxon Signed-Rank Test: W = {wilcoxon_stat:.3f}, p = {wilcoxon_p:.3f}")
    print(f"  Permutation Test: diff = {perm_diff:.3f}, p = {perm_p:.3f}")
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # 设置x轴位置
    x_pos = [0, 1]
    x_labels = ['Ordered', 'Noise']
    means = [ordered_mean, noise_mean]
    sems = [ordered_sem, noise_sem]
    
    # 绘制柱状图
    bars = ax.bar(x_pos, means, yerr=sems, capsize=5, 
                  color=[COLORS['ordered'], COLORS['noise']], 
                  alpha=0.7, edgecolor='white', linewidth=1.5,
                  error_kw={'linewidth': 2, 'capthick': 2})
    
    # 添加数据点（抖动散点）
    np.random.seed(42)  # 保证可重复性
    jitter_strength = 0.1
    
    for i, (x, values) in enumerate([(0, ordered_values), (1, noise_values)]):
        # 生成抖动位置
        jittered_x = x + np.random.normal(0, jitter_strength, len(values))
        
        # 绘制散点
        ax.scatter(jittered_x, values, 
                  color='white', s=60, alpha=0.9, 
                  edgecolors=COLORS['ordered'] if i == 0 else COLORS['noise'],
                  linewidth=2, zorder=10)
        
        # 添加动物标签
        for j, (jx, val, mouse) in enumerate(zip(jittered_x, values, results['mice'])):
            ax.annotate(mouse, (jx, val), xytext=(0, 8), 
                       textcoords='offset points', ha='center', va='bottom',
                       fontsize=8, fontweight='bold')
    
    # 添加统计检验结果标注
    y_max = max(max(ordered_values), max(noise_values))
    y_offset = y_max * 0.1
    line_height = y_max + y_offset
    
    # 统计检验连接线
    ax.plot([0, 1], [line_height, line_height], 'k-', linewidth=1.5)
    ax.plot([0, 0], [line_height - y_offset/3, line_height], 'k-', linewidth=1.5)
    ax.plot([1, 1], [line_height - y_offset/3, line_height], 'k-', linewidth=1.5)
    
    # 显示p值（使用Wilcoxon检验结果）
    p_text = f'p = {wilcoxon_p:.3f}'
    ax.text(0.5, line_height + y_offset/4, p_text, 
           ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 设置坐标轴
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, line_height + y_offset/2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontweight='bold')
    ax.set_ylabel('Mean Normalized Rich-Club Coefficient (Φ_norm)', fontweight='bold')
    
    # 添加网格
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # 标题
    title_en = 'Stimulus structure reshapes the organizational strength\\nof the V1 network core'
    ax.set_title(title_en, fontsize=14, fontweight='bold', pad=20)
    
    # 添加统计信息文本框
    stats_text = f'n = {len(results["mice"])} mice\\nWilcoxon: p = {wilcoxon_p:.3f}\\nPermutation: p = {perm_p:.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8),
           verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'figure2_panel_f.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\\nPanel F 保存完成: {save_path}")
    print(f"统计结果: {len(results['mice'])} 只动物")
    print(f"  Wilcoxon Signed-Rank Test: p = {wilcoxon_p:.3f}")
    print(f"  Permutation Test: p = {perm_p:.3f}")
    print(f"分析方法: 富人俱乐部区间(Φ_norm > 1)的平均值")


def panel_g(output_dir: str, density: float = 0.1):
    """
    Panel G: 跨动物峰值k值分析对比
    检验峰值k值(k_peak)在两种条件下是否存在显著不同
    分组柱状图 + 散点图，展示四只动物的峰值k值
    """
    setup_publication_style()
    
    print("=" * 60)
    print("生成 Panel G: 跨动物峰值k值分析对比")
    print("=" * 60)
    
    # 四只动物的配置
    mice_configs = {
        'm27': 'config/m27.json',
        'm30': 'config/m30.json', 
        'm65': 'config/m65.json',
        'm74': 'config/m74.json'
    }
    
    # 存储每只动物的结果
    results = {
        'ordered': [],    # 有序条件的峰值k
        'noise': [],      # 噪音条件的峰值k
        'mice': []        # 动物标识
    }
    
    for mouse_id, config_path in mice_configs.items():
        print(f"\n分析动物 {mouse_id}...")
        
        try:
            # 加载数据和配置
            segments, labels, neuron_pos, stimulus_data = load_session_data(config_path)
            if segments is None:
                continue
            
            # 计算RR神经元
            rr_neurons = compute_rr_neurons(segments, labels)
            print(f"  找到 {len(rr_neurons)} 个RR神经元")
            
            # 构建条件掩码
            masks = build_condition_masks(labels)
            
            if 'ordered' not in masks or 'noise' not in masks:
                print("  错误: 缺少有序或噪音条件数据")
                continue
            
            k_peak_values = {}
            
            for condition_name, mask in [('ordered', masks['ordered']), 
                                       ('noise', masks['noise'])]:
                print(f"  分析条件: {condition_name}")
                
                # 提取神经活动
                activity = extract_neural_activity_for_condition(
                    segments, mask, rr_neurons
                )
                
                if activity.shape[0] < 10:
                    print(f"    {condition_name}: 试次数不足，跳过")
                    k_peak_values[condition_name] = np.nan
                    continue
                
                # 构建功能连接网络
                corr_matrix, p_matrix = compute_correlation_matrix(activity, method='pearson')
                adj_matrix = threshold_correlation_matrix(corr_matrix, p_matrix, method='density', network_density=density)
                
                # 富人俱乐部分析
                G = nx.from_numpy_array(adj_matrix)
                rich_club_results = calculate_rich_club_analysis(G)
                
                # 获取峰值k值
                if rich_club_results['normalized_coefficients']:
                    normalized_coeffs = rich_club_results['normalized_coefficients']
                    k_values = rich_club_results['k_values']
                    
                    # 找到最大归一化系数对应的k值
                    max_idx = np.argmax(normalized_coeffs)
                    k_peak = k_values[max_idx]
                    max_coeff = normalized_coeffs[max_idx]
                else:
                    k_peak = np.nan
                    max_coeff = np.nan
                
                k_peak_values[condition_name] = k_peak
                print(f"    {mouse_id} {condition_name}: k_peak = {k_peak}, Φ_norm_max = {max_coeff:.3f}")
            
            # 只有当两个条件都有有效数据时才加入结果
            if not np.isnan(k_peak_values['ordered']) and not np.isnan(k_peak_values['noise']):
                results['ordered'].append(k_peak_values['ordered'])
                results['noise'].append(k_peak_values['noise'])
                results['mice'].append(mouse_id)
                
        except Exception as e:
            print(f"  动物 {mouse_id} 处理失败: {e}")
            continue
    
    # 检查是否有足够的数据
    if len(results['mice']) < 3:
        print(f"错误: 只有 {len(results['mice'])} 只动物有有效数据，无法进行统计分析")
        return
    
    print(f"\n成功分析 {len(results['mice'])} 只动物，开始可视化...")
    
    # 转换为numpy数组
    ordered_values = np.array(results['ordered'])
    noise_values = np.array(results['noise'])
    
    # 统计检验
    from scipy import stats
    
    # Wilcoxon Signed-Rank Test (配对非参数检验)
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(ordered_values, noise_values, alternative='two-sided')
    
    # Permutation Test (置换检验)
    def permutation_test(x, y, n_permutations=10000):
        """执行配对样本的置换检验"""
        observed_diff = np.mean(x) - np.mean(y)
        differences = x - y  # 配对差值
        
        # 生成置换分布
        perm_diffs = []
        np.random.seed(42)  # 保证可重复性
        
        for _ in range(n_permutations):
            # 对每个配对的差值随机翻转符号
            signs = np.random.choice([-1, 1], size=len(differences))
            perm_diff = np.mean(differences * signs)
            perm_diffs.append(perm_diff)
        
        # 计算p值 (双侧检验)
        perm_diffs = np.array(perm_diffs)
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
        
        return observed_diff, p_value
    
    obs_diff, perm_p = permutation_test(ordered_values, noise_values)
    
    print(f"\n统计检验结果:")
    print(f"  有序刺激: {np.mean(ordered_values):.1f} ± {np.std(ordered_values)/np.sqrt(len(ordered_values)):.1f}")
    print(f"  噪音刺激: {np.mean(noise_values):.1f} ± {np.std(noise_values)/np.sqrt(len(noise_values)):.1f}")
    print(f"  Wilcoxon Signed-Rank Test: p = {wilcoxon_p:.3f}")
    print(f"  Permutation Test: p = {perm_p:.3f}")
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    
    # 分组柱状图数据
    conditions = ['Ordered', 'Noise']
    means = [np.mean(ordered_values), np.mean(noise_values)]
    sems = [np.std(ordered_values)/np.sqrt(len(ordered_values)),
            np.std(noise_values)/np.sqrt(len(noise_values))]
    
    # 绘制柱状图
    bars = ax.bar(conditions, means, yerr=sems, 
                  color=[COLORS['ordered'], COLORS['noise']], 
                  alpha=0.7, capsize=5, width=0.6,
                  edgecolor='black', linewidth=1.5)
    
    # 添加散点图 (显示各动物的原始数据)
    x_jitter = 0.1
    for i, (condition, values) in enumerate(zip(conditions, [ordered_values, noise_values])):
        x_pos = np.random.normal(i, x_jitter, len(values))
        ax.scatter(x_pos, values, color='black', alpha=0.6, s=50, zorder=3)
        
        # 连接配对数据点
        if i == 0:  # 只在第一个条件时画连接线
            for j in range(len(values)):
                ax.plot([x_pos[j], 1 + np.random.normal(0, x_jitter)], 
                       [ordered_values[j], noise_values[j]], 
                       'k-', alpha=0.3, linewidth=1)
    
    # 设置样式
    ax.set_ylabel('Peak k value (k_peak)', fontweight='bold')
    ax.set_xlabel('Stimulus Condition', fontweight='bold')
    ax.set_title('Peak k Value Comparison Across Animals', fontweight='bold', pad=20)
    
    # 添加统计检验结果标注
    y_max = max(max(ordered_values), max(noise_values))
    y_offset = y_max * 0.1
    line_height = y_max + y_offset
    
    # 统计检验连接线
    ax.plot([0, 1], [line_height, line_height], 'k-', linewidth=1.5)
    ax.plot([0, 0], [line_height - y_offset/3, line_height], 'k-', linewidth=1.5)
    ax.plot([1, 1], [line_height - y_offset/3, line_height], 'k-', linewidth=1.5)
    
    # p值标注
    if min(wilcoxon_p, perm_p) < 0.001:
        p_text = f'p < 0.001'
    else:
        p_text = f'p = {min(wilcoxon_p, perm_p):.3f}'
    
    ax.text(0.5, line_height + y_offset/4, p_text, 
           ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 添加统计信息文本框
    stats_text = f'n = {len(results["mice"])} mice\nWilcoxon: p = {wilcoxon_p:.3f}\nPermutation: p = {perm_p:.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8),
           verticalalignment='top', fontsize=10)
    
    # 美化图形
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 保存图形
    save_path = os.path.join(output_dir, 'figure2_panel_g.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPanel G 保存完成: {save_path}")
    print(f"统计结果: {len(results['mice'])} 只动物")
    print(f"  Wilcoxon Signed-Rank Test: p = {wilcoxon_p:.3f}")
    print(f"  Permutation Test: p = {perm_p:.3f}")
    print(f"分析方法: 归一化富人俱乐部系数峰值对应的k值")


# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description='Generate Figure 2 panels')
    parser.add_argument('--panel', type=str, required=True, help='Panel letter (A/B/C/D/E/F/G)')
    parser.add_argument('--mouse', type=str, default='m27', help='Mouse/session key (m27/m30/m65/m74)')
    parser.add_argument('--config', type=str, default=None, help='Optional config JSON (overrides --mouse)')
    parser.add_argument('--outdir', type=str, default='figures', help='Output directory')
    parser.add_argument('--density', type=float, default=0.1, help='Network density for thresholding (0-1)')
    args = parser.parse_args()

    cfg_path = args.config if args.config else get_config_by_mouse(args.mouse)
    if cfg_path is None:
        raise RuntimeError('No valid config found. Please provide --config or a known --mouse key.')

    panel = args.panel.strip().upper()
    if panel == 'A':
        panel_a(args.outdir, cfg_path, density=args.density)
    elif panel == 'B':
        panel_b(args.outdir, cfg_path, density=args.density)
    elif panel == 'C':
        panel_c(args.outdir, cfg_path, density=args.density)
    elif panel == 'D':
        panel_d(args.outdir, cfg_path, density=args.density)
    elif panel == 'E':
        panel_e(args.outdir, cfg_path, density=args.density)
    elif panel == 'F':
        panel_f(args.outdir, density=args.density)  # Panel F不需要特定的config
    elif panel == 'G':
        panel_g(args.outdir, density=args.density)  # Panel G不需要特定的config
    else:
        raise ValueError('Unsupported panel. Use A/B/C/D/E/F/G.')


if __name__ == '__main__':
    main()
