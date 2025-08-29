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
    
    # 相关值处理方法
    CORRELATION_SIGN = 'positive'   # 'absolute', 'positive', 'negative', 'both'
    
    # 网络类型
    BINARIZE_NETWORK = True         # 是否二值化网络（False保留权重）
    
    # 可视化参数
    FIGURE_SIZE = (12, 8)
    FIGURE_SIZE_LARGE = (15, 10)
    COLORMAP = 'RdYlBu_r'
    
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

def visualize_correlation_matrix(corr_matrix, title="Correlation Matrix", save_path=None):
    """
    可视化相关系数矩阵
    
    参数:
    corr_matrix: 相关系数矩阵
    title: 图标题
    save_path: 保存路径
    """
    plt.figure(figsize=net_cfg.FIGURE_SIZE)
    
    # 创建热图
    sns.heatmap(corr_matrix, 
                cmap=net_cfg.COLORMAP, 
                center=0,
                square=True,
                cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title(title)
    plt.xlabel('Neuron Index')
    plt.ylabel('Neuron Index')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"相关矩阵图已保存: {save_path}")
    
    plt.show()

def visualize_network_graph(adj_matrix, neuron_pos=None, title="Network Graph", save_path=None):
    """
    可视化网络图
    
    参数:
    adj_matrix: 邻接矩阵
    neuron_pos: 神经元位置坐标 (2, n_neurons)
    title: 图标题
    save_path: 保存路径
    """
    plt.figure(figsize=net_cfg.FIGURE_SIZE_LARGE)
    
    # 创建networkx图
    G = nx.from_numpy_array(adj_matrix)
    
    # 设置节点位置
    if neuron_pos is not None and neuron_pos.shape[0] >= 2:
        pos = {i: (neuron_pos[0, i], neuron_pos[1, i]) for i in range(len(G.nodes()))}
    else:
        pos = nx.spring_layout(G, k=1, iterations=50)
    
    # 计算节点度数用于调整大小
    degrees = dict(G.degree())
    node_sizes = [degrees[node] * 20 + 50 for node in G.nodes()]
    
    # 绘制网络
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                          node_color='lightblue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"网络图已保存: {save_path}")
    
    plt.show()

def plot_degree_distribution(adj_matrix, title="Degree Distribution", save_path=None):
    """
    绘制度分布
    
    参数:
    adj_matrix: 邻接矩阵
    title: 图标题
    save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    
    # 创建networkx图并计算度
    G = nx.from_numpy_array(adj_matrix)
    degrees = [d for n, d in G.degree()]
    
    # 绘制直方图
    plt.subplot(1, 2, 1)
    plt.hist(degrees, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Histogram')
    plt.grid(True, alpha=0.3)
    
    # 绘制度分布（对数坐标）
    plt.subplot(1, 2, 2)
    degree_counts = Counter(degrees)
    degrees_unique = sorted(degree_counts.keys())
    counts = [degree_counts[d] for d in degrees_unique]
    
    plt.loglog(degrees_unique, counts, 'o-', alpha=0.7)
    plt.xlabel('Degree (log)')
    plt.ylabel('Count (log)')
    plt.title('Degree Distribution (log-log)')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"度分布图已保存: {save_path}")
    
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
    
    # %% 分条件构建功能连接矩阵
    print("\n=== 构建功能连接矩阵 ===")
    
    # 为每个类别分别构建网络
    category_networks = {}
    category_metrics = {}
    
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
    
    # %% 网络可视化
    print("\n=== 网络可视化 ===")
    
    # 创建保存目录
    results_dir = os.path.join(cfg.DATA_PATH, 'network_results')
    os.makedirs(results_dir, exist_ok=True)
    
    for category in category_networks.keys():
        print(f"\n可视化类别 {category} 的网络...")
        
        # 可视化相关矩阵
        visualize_correlation_matrix(
            category_networks[category]['correlation_matrix'],
            title=f"Category {category} Correlation Matrix",
            save_path=os.path.join(results_dir, f"category_{category}_correlation_matrix.png")
        )
        
        # 可视化网络图
        visualize_network_graph(
            category_networks[category]['adjacency_matrix'],
            neuron_pos=network_neuron_pos,
            title=f"Category {category} Network Graph",
            save_path=os.path.join(results_dir, f"category_{category}_network_graph.png")
        )
        
        # 可视化度分布
        plot_degree_distribution(
            category_networks[category]['adjacency_matrix'],
            title=f"Category {category} Degree Distribution",
            save_path=os.path.join(results_dir, f"category_{category}_degree_distribution.png")
        )
    
    # %% 比较不同类别的网络特征
    print("\n=== 网络特征比较 ===")
    
    if len(category_metrics) >= 2:
        # 创建比较表格
        comparison_df = pd.DataFrame(category_metrics).T
        print("\n网络指标比较:")
        print(comparison_df.round(3))
        
        # 保存比较结果
        comparison_df.to_csv(os.path.join(results_dir, 'network_metrics_comparison.csv'))
        print(f"\n网络指标比较已保存到: {os.path.join(results_dir, 'network_metrics_comparison.csv')}")
        
        # 可视化关键指标比较
        key_metrics = ['density', 'avg_clustering', 'modularity', 'avg_degree']
        
        plt.figure(figsize=(12, 8))
        for i, metric in enumerate(key_metrics, 1):
            plt.subplot(2, 2, i)
            categories = list(category_metrics.keys())
            values = [category_metrics[cat][metric] for cat in categories]
            
            plt.bar([f"Category {cat}" for cat in categories], values, alpha=0.7)
            plt.title(f'{metric.replace("_", " ").title()}')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'network_metrics_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
    
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
        
        # 可视化多条件比较
        key_metrics = ['density', 'avg_clustering', 'modularity', 'avg_degree', 'small_world_sigma']
        n_metrics = len(key_metrics)
        
        plt.figure(figsize=(16, 10))
        for i, metric in enumerate(key_metrics, 1):
            plt.subplot(2, 3, i)
            conditions = list(condition_metrics.keys())
            values = [condition_metrics[cond][metric] for cond in conditions]
            
            # 为不同类型的条件设置不同颜色
            colors = []
            for cond in conditions:
                if cond == 'Resting':
                    colors.append('gray')
                elif 'Cat1' in cond:
                    if 'Int0.0' in cond:
                        colors.append('navy')
                    elif 'Int0.2' in cond:
                        colors.append('blue')
                    elif 'Int0.5' in cond:
                        colors.append('lightblue')
                    elif 'Int1.0' in cond:
                        colors.append('skyblue')
                    else:
                        colors.append('blue')
                elif 'Cat2' in cond:
                    if 'Int0.0' in cond:
                        colors.append('darkred')
                    elif 'Int0.2' in cond:
                        colors.append('red')
                    elif 'Int0.5' in cond:
                        colors.append('lightcoral')
                    elif 'Int1.0' in cond:
                        colors.append('pink')
                    else:
                        colors.append('red')
                else:
                    colors.append('green')
            
            plt.bar(range(len(conditions)), values, color=colors, alpha=0.7)
            plt.xticks(range(len(conditions)), conditions, rotation=45, ha='right')
            plt.title(f'{metric.replace("_", " ").title()}')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'detailed_network_metrics_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        # 专门可视化类别2的强度效应
        if len(cat2_conditions) >= 2:
            plt.figure(figsize=(15, 10))
            metrics_for_intensity = ['density', 'avg_clustering', 'modularity', 'avg_degree']
            
            # 获取类别2的所有强度条件，按强度值排序
            cat2_data = []
            for cond in cat2_conditions:
                intensity_str = cond.split('_Int')[1]
                intensity_val = float(intensity_str)
                cat2_data.append((intensity_val, cond))
            cat2_data = sorted(cat2_data)
            
            n_intensities = len(cat2_data)
            n_metrics = len(metrics_for_intensity)
            
            # 为每个强度设置颜色
            colors = plt.cm.Reds(np.linspace(0.4, 0.9, n_intensities))
            
            # 创建分组柱状图
            x = np.arange(n_metrics)
            width = 0.8 / n_intensities
            
            for i, (intensity_val, condition) in enumerate(cat2_data):
                values = [condition_metrics[condition][m] for m in metrics_for_intensity]
                offset = (i - n_intensities/2 + 0.5) * width
                plt.bar(x + offset, values, width, 
                       label=f'强度 {intensity_val}', 
                       alpha=0.8, color=colors[i])
            
            plt.xlabel('网络指标')
            plt.ylabel('指标值')
            plt.title('类别2: 4种强度的网络指标比较')
            plt.xticks(x, [m.replace('_', ' ').title() for m in metrics_for_intensity])
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(os.path.join(results_dir, 'category2_intensity_comparison.png'), 
                       dpi=150, bbox_inches='tight')
            plt.show()
            
            # 添加强度趋势图
            plt.figure(figsize=(12, 8))
            
            for i, metric in enumerate(metrics_for_intensity):
                plt.subplot(2, 2, i+1)
                
                intensities = [data[0] for data in cat2_data]
                values = [condition_metrics[data[1]][metric] for data in cat2_data]
                
                plt.plot(intensities, values, 'o-', linewidth=2, markersize=8, alpha=0.7)
                plt.xlabel('强度')
                plt.ylabel(metric.replace('_', ' ').title())
                plt.title(f'类别2: {metric.replace("_", " ").title()} vs 强度')
                plt.grid(True, alpha=0.3)
                
                # 添加数值标签
                for j, (intensity, value) in enumerate(zip(intensities, values)):
                    plt.annotate(f'{value:.3f}', (intensity, value), 
                               textcoords="offset points", xytext=(0,10), ha='center')
            
            plt.suptitle('类别2网络指标随强度变化趋势')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'category2_intensity_trends.png'), 
                       dpi=150, bbox_inches='tight')
            plt.show()
        
        # 可视化静息状态 vs 任务状态对比
        task_conditions = [k for k in condition_metrics.keys() if k != 'Resting']
        if task_conditions:
            plt.figure(figsize=(14, 10))
            
            # 选择关键指标进行比较
            comparison_metrics = ['density', 'avg_clustering', 'modularity', 'avg_degree']
            
            for i, metric in enumerate(comparison_metrics, 1):
                plt.subplot(2, 2, i)
                
                # 获取静息状态和任务状态的值
                resting_value = condition_metrics['Resting'][metric]
                task_values = [condition_metrics[cond][metric] for cond in task_conditions]
                
                # 绘制静息状态基线
                plt.axhline(y=resting_value, color='gray', linestyle='--', 
                           label=f'静息状态: {resting_value:.3f}', alpha=0.8)
                
                # 绘制任务状态
                colors = ['blue', 'lightblue', 'red', 'pink'][:len(task_conditions)]
                plt.bar(range(len(task_conditions)), task_values, 
                       color=colors, alpha=0.7, label='任务状态')
                
                plt.xticks(range(len(task_conditions)), task_conditions, rotation=45, ha='right')
                plt.title(f'{metric.replace("_", " ").title()}')
                plt.ylabel(metric.replace('_', ' ').title())
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.suptitle('静息状态 vs 任务状态网络比较')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'resting_vs_task_comparison.png'), 
                       dpi=150, bbox_inches='tight')
            plt.show()
    
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