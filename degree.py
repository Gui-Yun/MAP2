# 网络拓扑中心性与神经信息关系分析
# guiy24@mails.tsinghua.edu.cn
# 2025-08-29
# 分析网络拓扑中心性指标与Fisher信息、分类准确率的关系
# %% 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats
from scipy.stats import f_oneway, pearsonr, spearmanr
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
from collections import defaultdict
import warnings
import os
warnings.filterwarnings('ignore')

# 导入项目模块
from loaddata import load_data, segment_neuron_data, reclassify_labels, fast_rr_selection, cfg

# %% 配置参数
class CentralityConfig:
    """拓扑中心性分析配置"""
    
    # 网络构建参数
    CORRELATION_THRESHOLD = 0.3    # 相关性阈值
    P_VALUE_THRESHOLD = 0.05       # p值阈值
    
    # 分层参数
    N_LEVELS = 5                   # 默认分层数量
    MULTI_LEVELS = [5, 7, 10]      # 多种分层数量用于比较
    
    # 可视化参数
    FIGURE_SIZE = (15, 10)
    FIGURE_SIZE_LARGE = (18, 12)
    FIGURE_SIZE_EXTRA_LARGE = (20, 15)
    SMALL_FIGURE_SIZE = (8, 6)
    
    # 科研绘图配置
    VISUALIZATION_DPI = 300         # 图像分辨率
    VISUALIZATION_STYLE = 'seaborn-v0_8-whitegrid'  # 科研绘图风格
    
    # 中心性分析配色方案
    CENTRALITY_COLORS = {
        'degree': '#2E86AB',           # 度中心性（蓝色）
        'betweenness': '#A23B72',      # 介数中心性（紫色）
        'closeness': '#F18F01',        # 接近中心性（橙色）
        'eigenvector': '#C73E1D',      # 特征向量中心性（红色）
        'pagerank': '#6C757D',         # PageRank中心性（灰色）
        'clustering': '#2ECC71',       # 聚类系数（绿色）
        'primary': '#2E86AB',          # 主要颜色
        'secondary': '#A23B72',        # 次要颜色
        'accent': '#F18F01',           # 强调色
        'success': '#C73E1D',          # 成功色
        'neutral': '#6C757D',          # 中性色
        'background': '#F8F9FA'        # 背景色
    }
    
    # 层级分析配色
    LEVEL_COLORS = ['#E74C3C', '#F39C12', '#F1C40F', '#2ECC71', '#3498DB']
    
    # 分析参数
    CV_FOLDS = 5                   # 交叉验证折数
    RANDOM_STATE = 42              # 随机种子

ccfg = CentralityConfig()

# %% 科研绘图风格设置
def setup_centrality_plot_style():
    """设置中心性分析科研绘图风格"""
    plt.style.use(ccfg.VISUALIZATION_STYLE)
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'font.family': 'Arial',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.2,
        'axes.edgecolor': '#2C3E50',
        'grid.alpha': 0.3,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })

# %% 网络构建函数
def build_correlation_network(neural_activity, method='density', correlation_threshold=ccfg.CORRELATION_THRESHOLD, 
                            network_density=0.1, p_threshold=ccfg.P_VALUE_THRESHOLD, 
                            correlation_sign='absolute', binarize=True):
    """
    基于network.py方法构建神经元相关性网络
    
    参数:
    neural_activity: 神经活动数据 (trials, neurons)
    method: 阈值化方法 ('absolute', 'density', 'significance_only')
    correlation_threshold: 相关性阈值
    network_density: 网络密度（用于density方法）
    p_threshold: p值阈值
    correlation_sign: 相关值符号处理 ('absolute', 'positive', 'negative', 'both')
    binarize: 是否二值化网络
    
    返回:
    G: NetworkX图对象
    correlation_matrix: 相关性矩阵
    adjacency_matrix: 邻接矩阵
    """
    print(f"构建相关性网络...")
    print(f"方法: {method}, p值阈值: {p_threshold}, 符号处理: {correlation_sign}")
    
    n_neurons = neural_activity.shape[1]
    
    # 计算相关系数矩阵和p值矩阵
    correlation_matrix = np.zeros((n_neurons, n_neurons))
    p_matrix = np.zeros((n_neurons, n_neurons))
    
    print(f"计算 {n_neurons} × {n_neurons} 相关矩阵...")
    for i in range(n_neurons):
        for j in range(i, n_neurons):
            if i == j:
                correlation_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
            else:
                corr, p_val = pearsonr(neural_activity[:, i], neural_activity[:, j])
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr
                p_matrix[i, j] = p_val
                p_matrix[j, i] = p_val
        
        if (i + 1) % 100 == 0:
            print(f"已完成 {i + 1}/{n_neurons} 个神经元")
    
    print(f"相关系数范围: {correlation_matrix.min():.3f} ~ {correlation_matrix.max():.3f}")
    
    # 使用network.py的阈值化方法
    adjacency_matrix = threshold_correlation_matrix(
        correlation_matrix, p_matrix, method, correlation_threshold, 
        network_density, p_threshold, correlation_sign, binarize)
    
    # 创建NetworkX图
    G = nx.from_numpy_array(adjacency_matrix)
    
    print(f"网络构建完成: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
    print(f"网络密度: {nx.density(G):.4f}")
    
    return G, correlation_matrix, adjacency_matrix

def threshold_correlation_matrix(corr_matrix, p_matrix, method='density',
                               corr_threshold=0.3, network_density=0.1,
                               p_threshold=0.05, correlation_sign='absolute',
                               binarize=True):
    """
    参考network.py的阈值化处理方法
    """
    print(f"应用阈值化处理: {method}")
    
    # 复制矩阵
    adj_matrix = corr_matrix.copy()
    n_nodes = adj_matrix.shape[0]
    
    # 移除自连接
    np.fill_diagonal(adj_matrix, 0)
    
    # 步骤1: 应用显著性阈值
    significant_mask = p_matrix < p_threshold
    adj_matrix[~significant_mask] = 0
    print(f"显著性过滤后保留 {np.sum(significant_mask)} 个连接")
    
    # 步骤2: 处理相关系数符号
    if correlation_sign == 'absolute':
        adj_matrix = np.abs(adj_matrix)
    elif correlation_sign == 'positive':
        adj_matrix[adj_matrix < 0] = 0
    elif correlation_sign == 'negative':
        adj_matrix[adj_matrix > 0] = 0
        adj_matrix = np.abs(adj_matrix)
    elif correlation_sign == 'both':
        pass  # 保持原始符号
    
    # 步骤3: 应用不同的阈值化方法
    if method == 'absolute':
        print(f"使用绝对阈值: {corr_threshold}")
        if correlation_sign == 'both':
            threshold_mask = np.abs(adj_matrix) > corr_threshold
        else:
            threshold_mask = adj_matrix > corr_threshold
        adj_matrix[~threshold_mask] = 0
        
    elif method == 'density':
        print(f"使用密度阈值: {network_density}")
        
        # 获取上三角部分的非零值
        triu_indices = np.triu_indices_from(adj_matrix, k=1)
        triu_values = adj_matrix[triu_indices]
        nonzero_values = triu_values[triu_values != 0]
        
        if len(nonzero_values) > 0:
            if correlation_sign == 'both':
                sort_values = np.abs(nonzero_values)
            else:
                sort_values = nonzero_values
            
            n_keep = int(len(nonzero_values) * network_density)
            n_keep = max(1, n_keep)
            
            threshold = np.partition(sort_values, -n_keep)[-n_keep]
            print(f"密度阈值: {threshold:.3f} (保留{n_keep}/{len(nonzero_values)}个连接)")
            
            if correlation_sign == 'both':
                density_mask = np.abs(adj_matrix) >= threshold
            else:
                density_mask = adj_matrix >= threshold
            adj_matrix[~density_mask] = 0
            
    elif method == 'significance_only':
        print("仅使用显著性阈值")
    
    # 步骤4: 二值化处理
    if binarize:
        adj_matrix = (adj_matrix != 0).astype(int)
        print("网络已二值化")
    
    # 确保对角线为0
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

# %% 中心性指标计算
def calculate_centrality_metrics(G):
    """
    计算各种中心性指标
    
    参数:
    G: NetworkX图对象
    
    返回:
    centrality_dict: 包含各种中心性指标的字典
    """
    print("计算中心性指标...")
    
    centrality_dict = {}
    
    # 1. 度中心性 (Degree Centrality)
    centrality_dict['degree'] = nx.degree_centrality(G)
    
    # 2. 介数中心性 (Betweenness Centrality)
    centrality_dict['betweenness'] = nx.betweenness_centrality(G)
    
    # 3. 接近中心性 (Closeness Centrality) - 只对连通图计算
    if nx.is_connected(G):
        centrality_dict['closeness'] = nx.closeness_centrality(G)
    else:
        # 对每个连通分量计算
        closeness = {}
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            component_closeness = nx.closeness_centrality(subgraph)
            closeness.update(component_closeness)
        # 对孤立节点设置为0
        for node in G.nodes():
            if node not in closeness:
                closeness[node] = 0.0
        centrality_dict['closeness'] = closeness
    
    # 4. 特征向量中心性 (Eigenvector Centrality)
    try:
        centrality_dict['eigenvector'] = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        print("特征向量中心性计算失败，使用度中心性替代")
        centrality_dict['eigenvector'] = centrality_dict['degree'].copy()
    
    # 5. PageRank中心性
    centrality_dict['pagerank'] = nx.pagerank(G)
    
    # 6. 聚类系数 (Clustering Coefficient)
    centrality_dict['clustering'] = nx.clustering(G)
    
    print(f"计算完成，包含 {len(centrality_dict)} 种指标")
    
    return centrality_dict

# %% Fisher信息计算
def calculate_multivariate_fisher_per_level(segments, labels, neuron_indices):
    """
    计算指定神经元集合的多变量Fisher信息
    
    参数:
    segments: 神经数据片段 (trials, neurons, timepoints)
    labels: 标签数组
    neuron_indices: 神经元索引列表
    
    返回:
    fisher_score: 多变量Fisher信息分数
    """
    if len(neuron_indices) == 0:
        return 0.0
        
    # 过滤有效数据
    valid_mask = labels != 0
    valid_segments = segments[valid_mask]
    valid_labels = labels[valid_mask]
    
    unique_labels = np.unique(valid_labels)
    if len(unique_labels) < 2:
        return 0.0
    
    # 使用刺激期数据的平均值
    stimulus_window = np.arange(cfg.T_STIMULUS, 
                               min(cfg.T_STIMULUS + cfg.L_STIMULUS, valid_segments.shape[2]))
    
    # 提取指定神经元的刺激期平均活动
    neuron_data = np.mean(valid_segments[:, neuron_indices, :][:, :, stimulus_window], axis=2)  # (trials, neurons)
    
    n_trials, n_neurons = neuron_data.shape
    
    # 检查样本数是否足够
    min_samples_per_class = min([np.sum(valid_labels == label) for label in unique_labels])
    if min_samples_per_class < 2 or n_trials < n_neurons + len(unique_labels):
        return 0.0
    
    # 数据标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    neuron_data_scaled = scaler.fit_transform(neuron_data)
    
    # 如果神经元数量过多，使用PCA降维
    if n_neurons > n_trials * 0.5:
        from sklearn.decomposition import PCA
        target_dim = max(2, min(10, n_trials // 3))
        pca = PCA(n_components=target_dim, random_state=42)
        neuron_data_scaled = pca.fit_transform(neuron_data_scaled)
        n_neurons = target_dim
        print(f"PCA降维: {len(neuron_indices)}维 -> {target_dim}维")
    
    # 计算总体均值
    grand_mean = np.mean(neuron_data_scaled, axis=0)
    
    # 计算类别均值和样本数
    class_means = []
    class_sizes = []
    
    for label in unique_labels:
        label_mask = valid_labels == label
        label_data = neuron_data_scaled[label_mask]
        if len(label_data) > 0:
            class_means.append(np.mean(label_data, axis=0))
            class_sizes.append(len(label_data))
        else:
            class_means.append(grand_mean)
            class_sizes.append(0)
    
    class_means = np.array(class_means)  # (n_classes, n_neurons)
    class_sizes = np.array(class_sizes)
    
    # 计算类间散布矩阵 (Between-class scatter matrix)
    S_b = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    for i, (class_mean, n_i) in enumerate(zip(class_means, class_sizes)):
        if n_i > 0:
            diff = (class_mean - grand_mean).reshape(-1, 1).astype(np.float64)
            S_b += n_i * np.dot(diff, diff.T).astype(np.float64)
    
    # 计算类内散布矩阵 (Within-class scatter matrix)
    S_w = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    for label in unique_labels:
        label_mask = valid_labels == label
        label_data = neuron_data_scaled[label_mask]
        if len(label_data) > 1:
            class_mean = np.mean(label_data, axis=0).astype(np.float64)
            centered_data = (label_data - class_mean).astype(np.float64)
            S_w += np.dot(centered_data.T, centered_data).astype(np.float64)
    
    # 正则化避免奇异矩阵
    regularization = 1e-6 * np.eye(n_neurons, dtype=np.float64)
    S_w += regularization
    
    try:
        # 计算多变量Fisher信息: trace(S_w^(-1) * S_b)
        from scipy.linalg import pinv
        S_w_inv = pinv(S_w).astype(np.float64)
        fisher_matrix = np.dot(S_w_inv, S_b).astype(np.float64)
        fisher_score = float(np.trace(fisher_matrix).real)
        
        # 确保返回非负有限值
        fisher_score = max(0.0, fisher_score)
        if not np.isfinite(fisher_score):
            fisher_score = 0.0
            
    except Exception as e:
        fisher_score = 0.0
        print(f"多变量Fisher信息计算失败: {e}")
    
    return fisher_score

def calculate_fisher_information_per_neuron(segments, labels, neuron_indices):
    """
    计算指定神经元的Fisher信息
    
    参数:
    segments: 神经数据片段 (trials, neurons, timepoints)
    labels: 标签数组
    neuron_indices: 神经元索引列表
    
    返回:
    fisher_scores: 每个神经元的Fisher信息分数
    """
    print(f"计算 {len(neuron_indices)} 个神经元的Fisher信息...")
    
    # 过滤有效数据
    valid_mask = labels != 0
    valid_segments = segments[valid_mask]
    valid_labels = labels[valid_mask]
    
    unique_labels = np.unique(valid_labels)
    fisher_scores = {}
    
    # 使用刺激期数据
    stimulus_window = np.arange(cfg.T_STIMULUS, 
                               min(cfg.T_STIMULUS + cfg.L_STIMULUS, valid_segments.shape[2]))
    
    for neuron_idx in neuron_indices:
        # 提取该神经元的刺激期平均活动
        neuron_data = np.mean(valid_segments[:, neuron_idx, stimulus_window], axis=1)
        
        # 计算类间和类内方差
        class_means = []
        class_vars = []
        
        for label in unique_labels:
            label_mask = valid_labels == label
            label_data = neuron_data[label_mask]
            
            if len(label_data) > 0:
                class_means.append(np.mean(label_data))
                class_vars.append(np.var(label_data))
        
        if len(class_means) < 2:
            fisher_scores[neuron_idx] = 0.0
            continue
        
        # Fisher比率：类间方差 / 类内方差
        between_class_var = np.var(class_means)
        within_class_var = np.mean(class_vars)
        
        if within_class_var > 0:
            fisher_scores[neuron_idx] = between_class_var / within_class_var
        else:
            fisher_scores[neuron_idx] = 0.0
    
    return fisher_scores

# %% 分类准确率计算
def calculate_classification_accuracy_per_neuron(segments, labels, neuron_indices):
    """
    计算单个神经元的分类准确率
    
    参数:
    segments: 神经数据片段 (trials, neurons, timepoints)
    labels: 标签数组
    neuron_indices: 神经元索引列表
    
    返回:
    accuracy_scores: 每个神经元的分类准确率
    """
    print(f"计算 {len(neuron_indices)} 个神经元的分类准确率...")
    
    # 过滤有效数据
    valid_mask = labels != 0
    valid_segments = segments[valid_mask]
    valid_labels = labels[valid_mask]
    
    accuracy_scores = {}
    
    # 使用刺激期数据
    stimulus_window = np.arange(cfg.T_STIMULUS, 
                               min(cfg.T_STIMULUS + cfg.L_STIMULUS, valid_segments.shape[2]))
    
    cv = StratifiedKFold(n_splits=ccfg.CV_FOLDS, shuffle=True, random_state=ccfg.RANDOM_STATE)
    
    for neuron_idx in neuron_indices:
        # 提取该神经元的刺激期数据
        X = valid_segments[:, neuron_idx, stimulus_window]  # (trials, timepoints)
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # SVM分类
        clf = SVC(kernel=cfg.SVM_KERNEL, C=cfg.SVM_C, 
                 random_state=ccfg.RANDOM_STATE, class_weight='balanced')
        
        try:
            scores = cross_val_score(clf, X_scaled, valid_labels, cv=cv, scoring='accuracy')
            accuracy_scores[neuron_idx] = scores.mean()
        except:
            accuracy_scores[neuron_idx] = 1.0 / len(np.unique(valid_labels))  # 随机水平
    
    return accuracy_scores

# %% 分层分析
def stratify_neurons_by_centrality(centrality_scores, n_levels=ccfg.N_LEVELS):
    """
    根据中心性分数将神经元分层
    
    参数:
    centrality_scores: 中心性分数字典
    n_levels: 分层数量
    
    返回:
    level_groups: 各层神经元索引列表
    level_ranges: 各层数值范围
    """
    # 转换为数组
    neuron_indices = list(centrality_scores.keys())
    scores = list(centrality_scores.values())
    
    # 计算分位数
    percentiles = np.linspace(0, 100, n_levels + 1)
    thresholds = np.percentile(scores, percentiles)
    
    level_groups = []
    level_ranges = []
    
    for i in range(n_levels):
        if i == 0:
            # 最低层
            mask = np.array(scores) <= thresholds[i+1]
        elif i == n_levels - 1:
            # 最高层
            mask = np.array(scores) > thresholds[i]
        else:
            # 中间层
            mask = (np.array(scores) > thresholds[i]) & (np.array(scores) <= thresholds[i+1])
        
        level_neurons = [neuron_indices[j] for j in range(len(neuron_indices)) if mask[j]]
        level_groups.append(level_neurons)
        
        if len(level_neurons) > 0:
            level_scores = [scores[j] for j in range(len(scores)) if mask[j]]
            level_ranges.append((min(level_scores), max(level_scores)))
        else:
            level_ranges.append((0, 0))
    
    print(f"神经元分层完成: {[len(group) for group in level_groups]}")
    
    return level_groups, level_ranges

# %% 可视化函数
def save_centrality_vs_information_data(centrality_name, level_groups, level_ranges, 
                                      fisher_scores, accuracy_scores, centrality_scores, save_dir="results"):
    """
    保存中心性与信息量关系数据
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备数据
    level_names = [f"Level {i+1}" for i in range(len(level_groups))]
    level_centralities = []
    level_fishers = []
    level_accuracies = []
    
    for group in level_groups:
        if len(group) > 0:
            group_centrality = [centrality_scores[n] for n in group]
            group_fisher = [fisher_scores[n] for n in group if n in fisher_scores]
            group_accuracy = [accuracy_scores[n] for n in group if n in accuracy_scores]
            
            level_centralities.append(np.mean(group_centrality))
            level_fishers.append(np.mean(group_fisher) if group_fisher else 0)
            level_accuracies.append(np.mean(group_accuracy) if group_accuracy else 0)
        else:
            level_centralities.append(0)
            level_fishers.append(0)
            level_accuracies.append(0)
    
    # 准备分布数据
    centrality_data = []
    level_labels = []
    for i, group in enumerate(level_groups):
        if len(group) > 0:
            group_scores = [centrality_scores[n] for n in group]
            centrality_data.extend(group_scores)
            level_labels.extend([f"L{i+1}"] * len(group_scores))
    
    # 计算相关性
    correlations = {}
    if len(level_centralities) > 2:
        corr_fisher, p_fisher = pearsonr(level_centralities, level_fishers)
        corr_acc, p_acc = pearsonr(level_centralities, level_accuracies)
        
        correlations = {
            'centrality_fisher_corr': corr_fisher,
            'centrality_fisher_p': p_fisher,
            'centrality_accuracy_corr': corr_acc,
            'centrality_accuracy_p': p_acc
        }
        
        print(f"\n=== {centrality_name} 相关性分析 ===")
        print(f"中心性 vs Fisher信息: r={corr_fisher:.3f}, p={p_fisher:.3f}")
        print(f"中心性 vs 分类准确率: r={corr_acc:.3f}, p={p_acc:.3f}")
    
    # 保存数据
    centrality_analysis_data = {
        'centrality_name': centrality_name,
        'level_names': level_names,
        'level_groups_sizes': [len(group) for group in level_groups],
        'level_ranges': level_ranges,
        'level_centralities': level_centralities,
        'level_fishers': level_fishers,
        'level_accuracies': level_accuracies,
        'centrality_distribution': centrality_data,
        'level_distribution_labels': level_labels,
        **correlations
    }
    
    filename = f"{centrality_name.lower()}_analysis.npz"
    np.savez_compressed(
        os.path.join(save_dir, filename),
        **centrality_analysis_data
    )
    print(f"{centrality_name} 中心性分析数据已保存: {filename}")

# %% 主分析函数
def analyze_centrality_information_relationship(neuron_data, segments, labels, rr_neurons=None):
    """
    分析拓扑中心性与信息量的关系
    
    参数:
    neuron_data: 神经活动数据 (time_points, neurons)
    segments: 分割后的数据 (trials, neurons, timepoints)
    labels: 标签数组
    rr_neurons: RR神经元索引（可选）
    """
    print("=== 开始拓扑中心性与信息关系分析 ===")
    
    # 如果提供了RR神经元，只分析这些神经元
    if rr_neurons is not None:
        print(f"使用 {len(rr_neurons)} 个RR神经元进行分析")
        neuron_data_subset = neuron_data[:, rr_neurons]
        segments_subset = segments[:, rr_neurons, :]
        neuron_mapping = {i: rr_neurons[i] for i in range(len(rr_neurons))}
    else:
        print(f"使用所有 {neuron_data.shape[1]} 个神经元进行分析")
        neuron_data_subset = neuron_data
        segments_subset = segments
        neuron_mapping = {i: i for i in range(neuron_data.shape[1])}
    
    # 1. 构建网络
    G, correlation_matrix, adjacency_matrix = build_correlation_network(neuron_data_subset)
    
    # 2. 计算中心性指标
    centrality_dict = calculate_centrality_metrics(G)
    
    # 3. 计算Fisher信息
    neuron_indices = list(range(neuron_data_subset.shape[1]))
    fisher_scores = calculate_fisher_information_per_neuron(segments_subset, labels, neuron_indices)
    
    # 4. 计算分类准确率
    accuracy_scores = calculate_classification_accuracy_per_neuron(segments_subset, labels, neuron_indices)
    
    # 5. 对每种中心性指标进行分析
    results = {}
    
    for centrality_name, centrality_scores in centrality_dict.items():
        print(f"\n--- 分析 {centrality_name} 中心性 ---")
        
        # 分层
        level_groups, level_ranges = stratify_neurons_by_centrality(centrality_scores)
        
        # 保存分析数据
        save_centrality_vs_information_data(centrality_name, level_groups, level_ranges,
                                           fisher_scores, accuracy_scores, centrality_scores, 
                                           save_dir=os.path.join(cfg.DATA_PATH, 'results'))
        
        # 多变量Fisher信息分析
        print(f"\n--- {centrality_name} 多变量Fisher信息分析 ---")
        os.makedirs(os.path.join(cfg.DATA_PATH, 'centrality_results'), exist_ok=True)
        multivariate_results = visualize_centrality_vs_multivariate_fisher(
            segments_subset, labels, centrality_name, level_groups, centrality_scores,
            save_path=os.path.join(cfg.DATA_PATH, 'centrality_results', f'{centrality_name}_multivariate_fisher_regression.png')
        )
        
        # 保存结果
        results[centrality_name] = {
            'centrality_scores': centrality_scores,
            'level_groups': level_groups,
            'level_ranges': level_ranges,
            'fisher_scores': fisher_scores,
            'accuracy_scores': accuracy_scores,
            'multivariate_analysis': multivariate_results
        }
    
    return results, G, correlation_matrix, adjacency_matrix

# %% 可视化函数
def visualize_centrality_distributions(centrality_dict, title="Centrality Metrics Distribution Analysis", save_path=None):
    """
    Professional visualization of multiple centrality metrics distributions
    
    Parameters:
    centrality_dict: Dictionary of centrality metrics
    title: Plot title
    save_path: Path to save the figure
    """
    setup_centrality_plot_style()
    
    n_metrics = len(centrality_dict)
    cols = min(3, n_metrics)
    rows = (n_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=ccfg.FIGURE_SIZE_LARGE)
    if n_metrics == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten() if n_metrics > 1 else [axes]
    else:
        axes = axes.flatten()
    
    # Color mapping for different centrality metrics
    color_map = {
        'degree': ccfg.CENTRALITY_COLORS['degree'],
        'betweenness': ccfg.CENTRALITY_COLORS['betweenness'], 
        'closeness': ccfg.CENTRALITY_COLORS['closeness'],
        'eigenvector': ccfg.CENTRALITY_COLORS['eigenvector'],
        'pagerank': ccfg.CENTRALITY_COLORS['pagerank'],
        'clustering': ccfg.CENTRALITY_COLORS['clustering']
    }
    
    for i, (metric_name, scores) in enumerate(centrality_dict.items()):
        ax = axes[i]
        values = np.array(list(scores.values()))
        
        # Get color for this metric
        color = color_map.get(metric_name, ccfg.CENTRALITY_COLORS['primary'])
        
        # Professional histogram with statistical overlay
        n_bins = min(30, max(10, len(values) // 10))
        counts, bins, patches = ax.hist(values, bins=n_bins, alpha=0.7, 
                                       color=color, edgecolor='black', linewidth=0.8)
        
        # Add statistical lines
        mean_val = np.mean(values)
        std_val = np.std(values)
        median_val = np.median(values)
        
        ax.axvline(mean_val, color='darkred', linestyle='--', linewidth=2, 
                  alpha=0.8, label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='darkblue', linestyle=':', linewidth=2, 
                  alpha=0.8, label=f'Median: {median_val:.3f}')
        
        # Kernel density estimation overlay
        try:
            from scipy.stats import gaussian_kde
            if len(values) > 3 and std_val > 0:
                kde = gaussian_kde(values)
                x_range = np.linspace(values.min(), values.max(), 100)
                kde_values = kde(x_range)
                # Scale KDE to match histogram
                kde_scaled = kde_values * len(values) * (bins[1] - bins[0])
                ax.plot(x_range, kde_scaled, color='black', linewidth=2, alpha=0.8, label='KDE')
        except:
            pass
        
        ax.set_xlabel(f'{metric_name.replace("_", " ").title()} Centrality')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{metric_name.replace("_", " ").title()} Distribution')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title, y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=ccfg.VISUALIZATION_DPI, bbox_inches='tight')
        print(f"Centrality distributions saved to: {save_path}")
    plt.show()

def visualize_centrality_relationships(centrality_dict, title="Centrality Metrics Correlation Analysis", save_path=None):
    """
    Professional visualization of correlations between different centrality metrics
    
    Parameters:
    centrality_dict: Dictionary of centrality metrics
    title: Plot title
    save_path: Path to save the figure
    """
    setup_centrality_plot_style()
    
    # Extract data
    metrics = list(centrality_dict.keys())
    n_metrics = len(metrics)
    
    if n_metrics < 2:
        print("Need at least 2 centrality metrics for comparison")
        return
    
    # Calculate correlation matrix
    correlation_matrix = np.zeros((n_metrics, n_metrics))
    p_value_matrix = np.zeros((n_metrics, n_metrics))
    
    for i, metric1 in enumerate(metrics):
        for j, metric2 in enumerate(metrics):
            values1 = list(centrality_dict[metric1].values())
            values2 = list(centrality_dict[metric2].values())
            
            if len(values1) > 1 and len(values2) > 1:
                corr, p_val = pearsonr(values1, values2)
                correlation_matrix[i, j] = corr
                p_value_matrix[i, j] = p_val
            else:
                correlation_matrix[i, j] = 1.0 if i == j else 0.0
                p_value_matrix[i, j] = 0.0 if i == j else 1.0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=ccfg.FIGURE_SIZE_LARGE)
    
    # 1. Correlation heatmap
    mask_upper = np.triu(np.ones_like(correlation_matrix), k=1)
    
    # Create custom annotations with significance indicators
    annotations = np.empty_like(correlation_matrix, dtype=object)
    for i in range(n_metrics):
        for j in range(n_metrics):
            if i <= j:  # Upper triangle and diagonal
                if i == j:
                    annotations[i, j] = '1.000'
                else:
                    corr_val = correlation_matrix[i, j]
                    p_val = p_value_matrix[i, j]
                    sig_mark = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                    annotations[i, j] = f'{corr_val:.3f}{sig_mark}'
            else:
                annotations[i, j] = ''
    
    im1 = sns.heatmap(correlation_matrix, mask=mask_upper, annot=annotations, fmt='', 
                      cmap='RdBu_r', center=0, square=True, ax=ax1,
                      xticklabels=[m.replace('_', ' ').title() for m in metrics], 
                      yticklabels=[m.replace('_', ' ').title() for m in metrics],
                      cbar_kws={"shrink": .8, "label": "Pearson Correlation"})
    
    ax1.set_title('Centrality Metrics Correlation Matrix\n(Upper Triangle)')
    
    # 2. Scatter plot matrix for top correlations
    # Find the highest correlations (excluding diagonal)
    corr_pairs = []
    for i in range(n_metrics):
        for j in range(i+1, n_metrics):
            corr_pairs.append((abs(correlation_matrix[i, j]), i, j, correlation_matrix[i, j]))
    
    corr_pairs.sort(reverse=True)
    
    # Show top 3 correlations in scatter plots
    if len(corr_pairs) > 0:
        ax2.clear()
        
        # Get the strongest correlation for main plot
        _, idx1, idx2, corr_val = corr_pairs[0]
        metric1, metric2 = metrics[idx1], metrics[idx2]
        
        values1 = np.array(list(centrality_dict[metric1].values()))
        values2 = np.array(list(centrality_dict[metric2].values()))
        
        # Scatter plot
        ax2.scatter(values1, values2, alpha=0.6, s=50, 
                   color=ccfg.CENTRALITY_COLORS['primary'],
                   edgecolors='black', linewidth=0.5)
        
        # Add regression line
        z = np.polyfit(values1, values2, 1)
        p = np.poly1d(z)
        ax2.plot(sorted(values1), p(sorted(values1)), 
                color=ccfg.CENTRALITY_COLORS['accent'], 
                linestyle='--', linewidth=2, alpha=0.8)
        
        ax2.set_xlabel(f'{metric1.replace("_", " ").title()} Centrality')
        ax2.set_ylabel(f'{metric2.replace("_", " ").title()} Centrality')
        ax2.set_title(f'Strongest Correlation: r = {corr_val:.3f}')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No correlations to display', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Correlation Scatter Plot')
    
    plt.suptitle(title, y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=ccfg.VISUALIZATION_DPI, bbox_inches='tight')
        print(f"Centrality relationships saved to: {save_path}")
    plt.show()

def visualize_centrality_vs_multivariate_fisher(segments, labels, centrality_name, level_groups, 
                                               centrality_scores, save_path=None):
    """
    可视化中心性层级与多变量Fisher信息的回归关系
    
    参数:
    segments: 神经数据片段 (trials, neurons, timepoints)
    labels: 标签数组
    centrality_name: 中心性指标名称
    level_groups: 各层神经元索引列表
    centrality_scores: 中心性分数字典
    save_path: 保存路径
    """
    setup_centrality_plot_style()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=ccfg.FIGURE_SIZE_LARGE)
    
    # 获取中心性对应的颜色
    centrality_color = ccfg.CENTRALITY_COLORS.get(centrality_name.lower(), ccfg.CENTRALITY_COLORS['primary'])
    
    # 计算每个level的数据
    level_names = []
    level_centralities = []
    level_multivariate_fishers = []
    level_neuron_counts = []
    level_individual_fishers = []
    
    print(f"\\n计算各层级的多变量Fisher信息...")
    
    for level_idx, group in enumerate(level_groups):
        if len(group) == 0:
            continue
            
        level_name = f"Level {level_idx + 1}"
        level_names.append(level_name)
        level_neuron_counts.append(len(group))
        
        # 平均中心度
        group_centrality = [centrality_scores[n] for n in group]
        avg_centrality = np.mean(group_centrality)
        level_centralities.append(avg_centrality)
        
        # 多变量Fisher信息
        multivariate_fisher = calculate_multivariate_fisher_per_level(segments, labels, group)
        level_multivariate_fishers.append(multivariate_fisher)
        
        # 个体Fisher信息均值（用于对比）
        individual_fishers = []
        for neuron in group:
            neuron_fisher = calculate_fisher_information_per_neuron(segments, labels, [neuron])
            if neuron in neuron_fisher:
                individual_fishers.append(neuron_fisher[neuron])
        
        avg_individual_fisher = np.mean(individual_fishers) if individual_fishers else 0
        level_individual_fishers.append(avg_individual_fisher)
        
        print(f"  {level_name}: {len(group)}个神经元, 中心度={avg_centrality:.3f}, "
              f"多变量FI={multivariate_fisher:.3f}, 个体FI均值={avg_individual_fisher:.3f}")
    
    # 1. 主要回归图：中心性 vs 多变量Fisher信息
    if len(level_centralities) > 1:
        ax1.scatter(level_centralities, level_multivariate_fishers, 
                   s=np.array(level_neuron_counts) * 5,  # 点大小反映神经元数量
                   c=range(len(level_centralities)), cmap='viridis',
                   alpha=0.8, edgecolors='black', linewidth=2)
        
        # 添加回归线
        if len(level_centralities) > 2:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(level_centralities, level_multivariate_fishers)
            line_x = np.linspace(min(level_centralities), max(level_centralities), 100)
            line_y = slope * line_x + intercept
            
            ax1.plot(line_x, line_y, '--', color='red', linewidth=2.5, alpha=0.8,
                    label=f'Regression (r={r_value:.3f}, p={p_value:.3f})')
            
            # 添加回归方程
            ax1.text(0.05, 0.95, f'y = {slope:.3f}x + {intercept:.3f}\\nr = {r_value:.3f}, p = {p_value:.3f}',
                    transform=ax1.transAxes, fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # 添加level标签
        for i, (x, y, name) in enumerate(zip(level_centralities, level_multivariate_fishers, level_names)):
            ax1.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)
        
        ax1.set_xlabel(f'{centrality_name.replace("_", " ").title()} Centrality')
        ax1.set_ylabel('Multivariate Fisher Information')
        ax1.set_title('Centrality vs Multivariate Fisher Information\\n(Level-wise Regression)')
        ax1.grid(True, alpha=0.3)
        if len(level_centralities) > 2:
            ax1.legend()
    
    # 2. 对比图：多变量FI vs 个体FI均值
    if len(level_multivariate_fishers) > 1:
        ax2.scatter(level_individual_fishers, level_multivariate_fishers,
                   s=np.array(level_neuron_counts) * 5,
                   c=range(len(level_multivariate_fishers)), cmap='plasma',
                   alpha=0.8, edgecolors='black', linewidth=2)
        
        # 对角线参考线
        min_val = min(min(level_individual_fishers), min(level_multivariate_fishers))
        max_val = max(max(level_individual_fishers), max(level_multivariate_fishers))
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x line')
        
        # 回归分析
        if len(level_individual_fishers) > 2:
            slope, intercept, r_value, p_value, std_err = linregress(level_individual_fishers, level_multivariate_fishers)
            line_x = np.linspace(min(level_individual_fishers), max(level_individual_fishers), 100)
            line_y = slope * line_x + intercept
            ax2.plot(line_x, line_y, '--', color='red', linewidth=2.5, alpha=0.8,
                    label=f'Regression (r={r_value:.3f})')
        
        ax2.set_xlabel('Average Individual Fisher Information')
        ax2.set_ylabel('Multivariate Fisher Information')
        ax2.set_title('Multivariate vs Individual Fisher Information\\n(Information Integration Effect)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    # 3. 柱状图：各层级多变量FI对比
    bars = ax3.bar(range(len(level_names)), level_multivariate_fishers,
                   color=[plt.cm.viridis(i/len(level_names)) for i in range(len(level_names))],
                   alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # 添加神经元数量标签
    for i, (bar, count, fisher) in enumerate(zip(bars, level_neuron_counts, level_multivariate_fishers)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height + max(level_multivariate_fishers) * 0.02,
                f'{fisher:.3f}\\n(n={count})', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')
    
    ax3.set_xlabel('Centrality Level')
    ax3.set_ylabel('Multivariate Fisher Information')
    ax3.set_title('Multivariate Fisher Information by Level')
    ax3.set_xticks(range(len(level_names)))
    ax3.set_xticklabels(level_names, rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. 增强效应分析：多变量FI与神经元数量的关系
    if len(level_neuron_counts) > 1:
        ax4.scatter(level_neuron_counts, level_multivariate_fishers,
                   c=level_centralities, s=100, cmap='viridis',
                   alpha=0.8, edgecolors='black', linewidth=2)
        
        # 回归分析
        if len(level_neuron_counts) > 2:
            slope, intercept, r_value, p_value, std_err = linregress(level_neuron_counts, level_multivariate_fishers)
            line_x = np.linspace(min(level_neuron_counts), max(level_neuron_counts), 100)
            line_y = slope * line_x + intercept
            ax4.plot(line_x, line_y, '--', color='red', linewidth=2.5, alpha=0.8,
                    label=f'Regression (r={r_value:.3f}, p={p_value:.3f})')
            ax4.legend()
        
        # 添加colorbar
        scatter = ax4.scatter(level_neuron_counts, level_multivariate_fishers,
                             c=level_centralities, s=0, cmap='viridis')  # 隐藏点，只为colorbar
        cbar = plt.colorbar(scatter, ax=ax4, shrink=0.8)
        cbar.set_label('Average Centrality', rotation=270, labelpad=20)
        
        ax4.set_xlabel('Number of Neurons in Level')
        ax4.set_ylabel('Multivariate Fisher Information')
        ax4.set_title('Information vs Population Size\\n(Network Integration Effect)')
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'{centrality_name.replace("_", " ").title()} Centrality: Multivariate Fisher Information Analysis', 
                 y=0.98, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=ccfg.VISUALIZATION_DPI, bbox_inches='tight')
        print(f"Multivariate Fisher regression analysis saved to: {save_path}")
    plt.show()
    
    # 返回分析结果
    return {
        'level_names': level_names,
        'level_centralities': level_centralities,
        'level_multivariate_fishers': level_multivariate_fishers,
        'level_individual_fishers': level_individual_fishers,
        'level_neuron_counts': level_neuron_counts
    }

def visualize_centrality_vs_information(centrality_name, level_groups, level_ranges, 
                                      fisher_scores, accuracy_scores, centrality_scores, save_path=None):
    """Professional visualization highlighting centrality-FI relationship"""
    setup_centrality_plot_style()
    
    fig = plt.figure(figsize=ccfg.FIGURE_SIZE_EXTRA_LARGE)
    
    # Create complex layout: main plot takes most space, supporting plots are smaller
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3, 
                         width_ratios=[2, 1, 1], height_ratios=[2, 1, 1])
    
    # Main plot: Direct correlation scatter plot
    ax_main = fig.add_subplot(gs[0, :])
    
    # Supporting plots
    ax_levels = fig.add_subplot(gs[1, 0])  # Level stratification
    ax_corr = fig.add_subplot(gs[1, 1:])   # Correlation strength
    ax_dist = fig.add_subplot(gs[2, :])    # Distribution comparison
    
    # Prepare data
    level_names = [f"Level {i+1}" for i in range(len(level_groups))]
    level_centralities = []
    level_fishers = []
    level_accuracies = []
    
    for group in level_groups:
        if len(group) > 0:
            group_centrality = [centrality_scores[n] for n in group]
            group_fisher = [fisher_scores[n] for n in group if n in fisher_scores]
            group_accuracy = [accuracy_scores[n] for n in group if n in accuracy_scores]
            
            level_centralities.append(np.mean(group_centrality))
            level_fishers.append(np.mean(group_fisher) if group_fisher else 0)
            level_accuracies.append(np.mean(group_accuracy) if group_accuracy else 0)
        else:
            level_centralities.append(0)
            level_fishers.append(0)
            level_accuracies.append(0)
    
    # Color scheme
    centrality_color = ccfg.CENTRALITY_COLORS.get(centrality_name.lower(), ccfg.CENTRALITY_COLORS['primary'])
    
    # === MAIN PLOT: Individual neuron scatter plot ===
    if len(centrality_scores) > 0 and len(fisher_scores) > 0:
        # Get individual neuron data
        neuron_centralities = []
        neuron_fishers = []
        neuron_levels = []
        
        for level_idx, group in enumerate(level_groups):
            for neuron in group:
                if neuron in centrality_scores and neuron in fisher_scores:
                    neuron_centralities.append(centrality_scores[neuron])
                    neuron_fishers.append(fisher_scores[neuron])
                    neuron_levels.append(level_idx)
        
        if len(neuron_centralities) > 0:
            # Scatter plot of individual neurons colored by level
            scatter = ax_main.scatter(neuron_centralities, neuron_fishers, 
                                    c=neuron_levels, s=80, alpha=0.7, 
                                    cmap='viridis', edgecolors='black', linewidth=0.8)
            
            # Add trend line
            if len(neuron_centralities) > 3:
                try:
                    z = np.polyfit(neuron_centralities, neuron_fishers, 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(min(neuron_centralities), max(neuron_centralities), 100)
                    ax_main.plot(x_range, p(x_range), color='red', linestyle='--', 
                               linewidth=3, alpha=0.8, label='Trend Line')
                    
                    # Calculate and display correlation
                    corr_coef, p_val = pearsonr(neuron_centralities, neuron_fishers)
                    ax_main.text(0.05, 0.95, f'r = {corr_coef:.3f}, p = {p_val:.3f}',
                               transform=ax_main.transAxes, fontsize=14, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                except:
                    pass
            
            # Highlight level means as larger points
            level_scatter = ax_main.scatter(level_centralities, level_fishers, 
                                          s=300, c=range(len(level_centralities)), 
                                          cmap='viridis', edgecolors='red', linewidth=3,
                                          marker='s', alpha=0.9, label='Level Means')
            
            ax_main.set_xlabel(f'{centrality_name.replace("_", " ").title()} Centrality', fontsize=14)
            ax_main.set_ylabel('Fisher Information', fontsize=14)
            ax_main.set_title(f'{centrality_name.replace("_", " ").title()} Centrality vs Fisher Information\n(Individual Neurons and Level Averages)', 
                            fontsize=16, fontweight='bold')
            ax_main.grid(True, alpha=0.3)
            ax_main.legend(loc='upper right')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax_main, shrink=0.8)
            cbar.set_label('Centrality Level', rotation=270, labelpad=20, fontsize=12)
    
    # === SUPPORTING PLOT 1: Level stratification ===
    if len(level_centralities) > 0:
        # Dual y-axis plot for levels
        line1 = ax_levels.plot(range(len(level_centralities)), level_centralities, 'o-', 
                              color=centrality_color, linewidth=3, markersize=10, 
                              alpha=0.8, markerfacecolor='white', markeredgewidth=2,
                              label=f'{centrality_name.replace("_", " ").title()}')
        
        ax_levels_twin = ax_levels.twinx()
        line2 = ax_levels_twin.plot(range(len(level_fishers)), level_fishers, 's-', 
                                   color='red', linewidth=3, markersize=10,
                                   alpha=0.8, markerfacecolor='white', markeredgewidth=2,
                                   label='Fisher Info')
        
        ax_levels.set_xlabel('Centrality Level', fontsize=12)
        ax_levels.set_ylabel(f'{centrality_name.replace("_", " ").title()}', color=centrality_color, fontsize=12)
        ax_levels_twin.set_ylabel('Fisher Information', color='red', fontsize=12)
        ax_levels.set_title('Level Progression', fontsize=14, fontweight='bold')
        ax_levels.set_xticks(range(len(level_names)))
        ax_levels.set_xticklabels(level_names, rotation=45)
        ax_levels.grid(True, alpha=0.3)
        ax_levels.tick_params(axis='y', labelcolor=centrality_color)
        ax_levels_twin.tick_params(axis='y', labelcolor='red')
    
    # === SUPPORTING PLOT 2: Correlation analysis ===
    if len(level_centralities) > 2:
        # Calculate correlations for different level numbers
        level_numbers = [3, 5, 7, 10]
        correlations = []
        p_values = []
        
        for n_levels in level_numbers:
            temp_groups, _ = stratify_neurons_by_centrality(centrality_scores, n_levels)
            temp_centralities = []
            temp_fishers = []
            
            for group in temp_groups:
                if len(group) > 0:
                    group_centrality = [centrality_scores[n] for n in group]
                    group_fisher = [fisher_scores[n] for n in group if n in fisher_scores]
                    
                    temp_centralities.append(np.mean(group_centrality))
                    temp_fishers.append(np.mean(group_fisher) if group_fisher else 0)
            
            if len(temp_centralities) > 2:
                corr, p_val = pearsonr(temp_centralities, temp_fishers)
                correlations.append(abs(corr))
                p_values.append(p_val)
            else:
                correlations.append(0)
                p_values.append(1)
        
        bars = ax_corr.bar(range(len(level_numbers)), correlations, 
                          color=[plt.cm.viridis(i/len(level_numbers)) for i in range(len(level_numbers))],
                          alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add significance markers
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            height = bar.get_height()
            if p_val < 0.001:
                sig_mark = '***'
            elif p_val < 0.01:
                sig_mark = '**'
            elif p_val < 0.05:
                sig_mark = '*'
            else:
                sig_mark = 'ns'
            
            ax_corr.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                        sig_mark, ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax_corr.set_xlabel('Number of Levels', fontsize=12)
        ax_corr.set_ylabel('|Correlation|', fontsize=12)
        ax_corr.set_title('Correlation Strength by Level Number', fontsize=14, fontweight='bold')
        ax_corr.set_xticks(range(len(level_numbers)))
        ax_corr.set_xticklabels([f'{n}' for n in level_numbers])
        ax_corr.set_ylim(0, 1.1)
        ax_corr.grid(True, alpha=0.3)
    
    # === SUPPORTING PLOT 3: Distribution comparison ===
    if len(level_groups) > 0:
        # Create violin plots for centrality and fisher info distributions by level
        centrality_data = []
        fisher_data = []
        level_labels = []
        
        for level_idx, group in enumerate(level_groups):
            level_label = f'L{level_idx+1}'
            for neuron in group:
                if neuron in centrality_scores:
                    centrality_data.append(centrality_scores[neuron])
                    level_labels.append(level_label)
                if neuron in fisher_scores:
                    fisher_data.append(fisher_scores[neuron])
        
        if len(centrality_data) > 0:
            import pandas as pd
            df = pd.DataFrame({
                'centrality': centrality_data,
                'level': level_labels
            })
            
            # Violin plot for centrality distribution by level
            parts = ax_dist.violinplot([df[df['level'] == f'L{i+1}']['centrality'].values 
                                      for i in range(len(level_groups)) 
                                      if len(df[df['level'] == f'L{i+1}']) > 0],
                                     positions=range(len(level_groups)), 
                                     widths=0.6, showmeans=True, showmedians=True)
            
            # Color the violins
            for pc, color in zip(parts['bodies'], plt.cm.viridis(np.linspace(0, 1, len(parts['bodies'])))):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            ax_dist.set_xlabel('Centrality Level', fontsize=12)
            ax_dist.set_ylabel(f'{centrality_name.replace("_", " ").title()} Distribution', fontsize=12)
            ax_dist.set_title('Centrality Distribution by Level', fontsize=14, fontweight='bold')
            ax_dist.set_xticks(range(len(level_groups)))
            ax_dist.set_xticklabels([f'L{i+1}' for i in range(len(level_groups))])
            ax_dist.grid(True, alpha=0.3)
    
    plt.suptitle(f'{centrality_name.replace("_", " ").title()} Centrality - Fisher Information Relationship Analysis', 
                 y=0.98, fontsize=18, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=ccfg.VISUALIZATION_DPI, bbox_inches='tight')
        print(f"Enhanced centrality vs information analysis saved to: {save_path}")
    plt.show()

def visualize_network_with_centrality(G, centrality_scores, neuron_pos=None, centrality_name="degree", save_path=None):
    """Professional network visualization with centrality-based node coloring and sizing"""
    setup_centrality_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=ccfg.FIGURE_SIZE_EXTRA_LARGE)
    
    # Get centrality color
    centrality_color = ccfg.CENTRALITY_COLORS.get(centrality_name.lower(), ccfg.CENTRALITY_COLORS['primary'])
    
    # Set node positions
    if neuron_pos is not None and neuron_pos.shape[0] >= 2:
        pos = {i: (neuron_pos[0, i], neuron_pos[1, i]) for i in range(neuron_pos.shape[1])}
    else:
        pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42)
    
    # Prepare centrality values and node sizes
    centrality_values = [centrality_scores[node] for node in G.nodes()]
    
    if centrality_values:
        # Normalize for better visualization
        min_val, max_val = min(centrality_values), max(centrality_values)
        if max_val > min_val:
            normalized_values = [(val - min_val) / (max_val - min_val) for val in centrality_values]
        else:
            normalized_values = [0.5] * len(centrality_values)
        
        node_sizes = [50 + 400 * norm_val for norm_val in normalized_values]
    else:
        node_sizes = [100] * len(G.nodes())
        normalized_values = [0.5] * len(G.nodes())
    
    # 1. Main network visualization
    ax1.set_title(f'Network Topology Colored by {centrality_name.replace("_", " ").title()} Centrality')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.4, width=0.8, 
                          edge_color=ccfg.CENTRALITY_COLORS['neutral'])
    
    # Draw nodes with centrality coloring
    if centrality_values:
        nodes = nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=node_sizes,
                                      node_color=centrality_values, cmap='viridis',
                                      alpha=0.8, edgecolors='black', linewidths=1.0)
        
        # Add colorbar
        cbar1 = plt.colorbar(nodes, ax=ax1, shrink=0.8)
        cbar1.set_label(f'{centrality_name.replace("_", " ").title()} Centrality', rotation=270, labelpad=20)
    else:
        nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=100,
                              node_color=centrality_color, alpha=0.8)
    
    ax1.axis('off')
    
    
    # 2. Hub nodes emphasis
    ax2.set_title(f'Hub Nodes by {centrality_name.replace("_", " ").title()} Centrality')
    
    if centrality_values:
        # Identify hub nodes (top 20%)
        threshold = np.percentile(centrality_values, 80) if centrality_values else 0
        hub_nodes = [node for node in G.nodes() if centrality_scores[node] >= threshold]
        regular_nodes = [node for node in G.nodes() if centrality_scores[node] < threshold]
        
        # Draw all edges
        nx.draw_networkx_edges(G, pos, ax=ax2, alpha=0.3, width=0.6,
                              edge_color=ccfg.CENTRALITY_COLORS['neutral'])
        
        # Draw regular nodes (smaller, faded)
        if regular_nodes:
            regular_sizes = [node_sizes[node] * 0.6 for node in regular_nodes]
            nx.draw_networkx_nodes(G, pos, ax=ax2, nodelist=regular_nodes,
                                  node_size=regular_sizes, 
                                  node_color=ccfg.CENTRALITY_COLORS['background'],
                                  alpha=0.4, edgecolors='gray', linewidths=0.5)
        
        # Draw hub nodes (larger, highlighted)
        if hub_nodes:
            hub_sizes = [node_sizes[node] * 1.2 for node in hub_nodes]
            hub_colors = [centrality_scores[node] for node in hub_nodes]
            
            hub_scatter = nx.draw_networkx_nodes(G, pos, ax=ax2, nodelist=hub_nodes,
                                               node_size=hub_sizes, 
                                               node_color=hub_colors, cmap='viridis',
                                               alpha=0.9, edgecolors='black', linewidths=2.0)
            
            # Add colorbar for hubs
            cbar2 = plt.colorbar(hub_scatter, ax=ax2, shrink=0.8)
            cbar2.set_label(f'Hub {centrality_name.replace("_", " ").title()}', rotation=270, labelpad=20)
            
    
    ax2.axis('off')
    
    plt.suptitle(f'{centrality_name.replace("_", " ").title()} Centrality Network Analysis', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=ccfg.VISUALIZATION_DPI, bbox_inches='tight')
        print(f"Network centrality visualization saved to: {save_path}")
    plt.show()

def visualize_multi_level_comparison(centrality_scores, fisher_scores, accuracy_scores, 
                                   centrality_name, levels_list=None, save_path=None):
    """
    科研级别多分档层次比较可视化
    
    参数:
    centrality_scores: 中心性分数字典
    fisher_scores: Fisher信息分数字典
    accuracy_scores: 分类准确率分数字典
    centrality_name: 中心性指标名称
    levels_list: 分档数量列表
    save_path: 保存路径
    """
    if levels_list is None:
        levels_list = ccfg.MULTI_LEVELS
    
    setup_centrality_plot_style()
    
    fig, axes = plt.subplots(2, 3, figsize=ccfg.FIGURE_SIZE_EXTRA_LARGE)
    axes = axes.flatten()
    
    # 获取中心性对应的颜色
    centrality_color = ccfg.CENTRALITY_COLORS.get(centrality_name.lower(), ccfg.CENTRALITY_COLORS['primary'])
    
    # 为每种分档数量进行分析
    all_results = {}
    for n_levels in levels_list:
        level_groups, level_ranges = stratify_neurons_by_centrality(centrality_scores, n_levels)
        
        # 计算每层的平均值
        level_centralities = []
        level_fishers = []  
        level_accuracies = []
        
        for group in level_groups:
            if len(group) > 0:
                group_centrality = [centrality_scores[n] for n in group]
                group_fisher = [fisher_scores[n] for n in group if n in fisher_scores]
                group_accuracy = [accuracy_scores[n] for n in group if n in accuracy_scores]
                
                level_centralities.append(np.mean(group_centrality))
                level_fishers.append(np.mean(group_fisher) if group_fisher else 0)
                level_accuracies.append(np.mean(group_accuracy) if group_accuracy else 0)
            else:
                level_centralities.append(0)
                level_fishers.append(0) 
                level_accuracies.append(0)
        
        all_results[n_levels] = {
            'centralities': level_centralities,
            'fishers': level_fishers,
            'accuracies': level_accuracies,
            'groups': level_groups
        }
    
    # 1. 中心性分层分布对比
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(levels_list)))
    
    for i, (n_levels, color) in enumerate(zip(levels_list, colors)):
        centralities = all_results[n_levels]['centralities']
        x_positions = np.arange(len(centralities)) + i * 0.25
        
        bars = ax1.bar(x_positions, centralities, width=0.2, alpha=0.7,
                      color=color, edgecolor='black', linewidth=1,
                      label=f'{n_levels} Levels')
    
    ax1.set_xlabel('Centrality Level')
    ax1.set_ylabel(f'{centrality_name.replace("_", " ").title()} Centrality')
    ax1.set_title('Multi-Level Centrality Stratification Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Fisher信息层次关系
    ax2 = axes[1]
    for i, (n_levels, color) in enumerate(zip(levels_list, colors)):
        fishers = all_results[n_levels]['fishers']
        x_range = np.arange(len(fishers))
        ax2.plot(x_range, fishers, 'o-', color=color, linewidth=2.5, 
                markersize=6, alpha=0.8, label=f'{n_levels} Levels')
    
    ax2.set_xlabel('Centrality Level')
    ax2.set_ylabel('Fisher Information')
    ax2.set_title('Fisher Information Across Different Level Numbers')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 分类准确率层次关系
    ax3 = axes[2]
    for i, (n_levels, color) in enumerate(zip(levels_list, colors)):
        accuracies = all_results[n_levels]['accuracies'] 
        x_range = np.arange(len(accuracies))
        ax3.plot(x_range, accuracies, '^-', color=color, linewidth=2.5,
                markersize=6, alpha=0.8, label=f'{n_levels} Levels')
    
    ax3.set_xlabel('Centrality Level')
    ax3.set_ylabel('Classification Accuracy')
    ax3.set_title('Classification Accuracy Across Different Level Numbers')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 相关性强度比较 (中心性 vs Fisher信息)
    ax4 = axes[3]
    correlations_fisher = []
    p_values_fisher = []
    
    for n_levels in levels_list:
        centralities = all_results[n_levels]['centralities']
        fishers = all_results[n_levels]['fishers']
        
        if len(centralities) > 2:
            corr, p_val = pearsonr(centralities, fishers)
            correlations_fisher.append(abs(corr))
            p_values_fisher.append(p_val)
        else:
            correlations_fisher.append(0)
            p_values_fisher.append(1)
    
    bars = ax4.bar(range(len(levels_list)), correlations_fisher, 
                  color=[colors[i] for i in range(len(levels_list))],
                  alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # 添加显著性标记
    for i, (bar, p_val) in enumerate(zip(bars, p_values_fisher)):
        height = bar.get_height()
        if p_val < 0.001:
            sig_mark = '***'
        elif p_val < 0.01:
            sig_mark = '**'
        elif p_val < 0.05:
            sig_mark = '*'
        else:
            sig_mark = 'ns'
        
        ax4.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                sig_mark, ha='center', va='bottom', fontweight='bold')
    
    ax4.set_xlabel('Number of Levels')
    ax4.set_ylabel('|Correlation| with Fisher Information')
    ax4.set_title('Centrality-Fisher Correlation Strength by Level Number')
    ax4.set_xticks(range(len(levels_list)))
    ax4.set_xticklabels([f'{n}' for n in levels_list])
    ax4.grid(True, alpha=0.3)
    
    # 5. 相关性强度比较 (中心性 vs 分类准确率)
    ax5 = axes[4]
    correlations_acc = []
    p_values_acc = []
    
    for n_levels in levels_list:
        centralities = all_results[n_levels]['centralities']
        accuracies = all_results[n_levels]['accuracies']
        
        if len(centralities) > 2:
            corr, p_val = pearsonr(centralities, accuracies)
            correlations_acc.append(abs(corr))
            p_values_acc.append(p_val)
        else:
            correlations_acc.append(0)
            p_values_acc.append(1)
    
    bars = ax5.bar(range(len(levels_list)), correlations_acc,
                  color=[colors[i] for i in range(len(levels_list))],
                  alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # 添加显著性标记
    for i, (bar, p_val) in enumerate(zip(bars, p_values_acc)):
        height = bar.get_height()
        if p_val < 0.001:
            sig_mark = '***'
        elif p_val < 0.01:
            sig_mark = '**' 
        elif p_val < 0.05:
            sig_mark = '*'
        else:
            sig_mark = 'ns'
        
        ax5.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                sig_mark, ha='center', va='bottom', fontweight='bold')
    
    ax5.set_xlabel('Number of Levels')
    ax5.set_ylabel('|Correlation| with Classification Accuracy')
    ax5.set_title('Centrality-Accuracy Correlation Strength by Level Number')
    ax5.set_xticks(range(len(levels_list)))
    ax5.set_xticklabels([f'{n}' for n in levels_list])
    ax5.grid(True, alpha=0.3)
    
    # 6. 统计总结 - 移除文字介绍，仅保留空白子图
    ax6 = axes[5]
    ax6.axis('off')
    
    plt.suptitle(f'{centrality_name.replace("_", " ").title()} Centrality: Multi-Level Stratification Analysis', 
                 y=0.98, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=ccfg.VISUALIZATION_DPI, bbox_inches='tight')
        print(f"Multi-level comparison saved to: {save_path}")
    plt.show()
    
    return all_results

def visualize_centrality_heatmap(centrality_dict, save_path=None):
    """
    科研级别中心性指标热力图可视化
    """
    setup_centrality_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=ccfg.FIGURE_SIZE_LARGE)
    
    # 准备数据
    metrics = list(centrality_dict.keys())
    neurons = list(next(iter(centrality_dict.values())).keys())
    n_neurons = len(neurons)
    n_metrics = len(metrics)
    
    # 创建中心性矩阵
    centrality_matrix = np.zeros((n_metrics, n_neurons))
    for i, metric in enumerate(metrics):
        for j, neuron in enumerate(neurons):
            centrality_matrix[i, j] = centrality_dict[metric][neuron]
    
    # 1. 原始中心性热力图
    im1 = sns.heatmap(centrality_matrix, 
                     xticklabels=False,
                     yticklabels=[m.replace('_', ' ').title() for m in metrics],
                     cmap='viridis', ax=ax1, cbar_kws={'label': 'Centrality Value'})
    ax1.set_title('Raw Centrality Values Heatmap')
    ax1.set_xlabel('Neurons')
    ax1.set_ylabel('Centrality Metrics')
    
    # 2. 标准化中心性热力图
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    centrality_matrix_normalized = scaler.fit_transform(centrality_matrix.T).T
    
    im2 = sns.heatmap(centrality_matrix_normalized,
                     xticklabels=False, 
                     yticklabels=[m.replace('_', ' ').title() for m in metrics],
                     cmap='RdBu_r', center=0, ax=ax2,
                     cbar_kws={'label': 'Normalized Centrality (Z-score)'})
    ax2.set_title('Normalized Centrality Values Heatmap')
    ax2.set_xlabel('Neurons')
    ax2.set_ylabel('Centrality Metrics')
    
    plt.suptitle('Centrality Metrics Heatmap Analysis', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=ccfg.VISUALIZATION_DPI, bbox_inches='tight')
        print(f"Centrality heatmap saved to: {save_path}")
    plt.show()

def visualize_centrality_radar_chart(centrality_dict, top_n=10, save_path=None):
    """
    科研级别雷达图：展示顶级神经元的多维中心性
    """
    setup_centrality_plot_style()
    
    # 获取度中心性最高的前N个神经元
    degree_scores = centrality_dict.get('degree', centrality_dict[list(centrality_dict.keys())[0]])
    top_neurons = sorted(degree_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_neuron_ids = [n[0] for n in top_neurons]
    
    metrics = list(centrality_dict.keys())
    n_metrics = len(metrics)
    
    # 设置雷达图
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # 闭合
    
    fig, ax = plt.subplots(figsize=ccfg.FIGURE_SIZE, subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.tab10(np.linspace(0, 1, min(top_n, 10)))
    
    # 标准化数据用于雷达图
    all_values = {}
    for metric in metrics:
        values = list(centrality_dict[metric].values())
        min_val, max_val = min(values), max(values)
        if max_val > min_val:
            all_values[metric] = {k: (v - min_val) / (max_val - min_val) 
                                for k, v in centrality_dict[metric].items()}
        else:
            all_values[metric] = {k: 0.5 for k in centrality_dict[metric].keys()}
    
    # 绘制每个顶级神经元
    for i, (neuron_id, color) in enumerate(zip(top_neuron_ids, colors)):
        values = [all_values[metric][neuron_id] for metric in metrics]
        values += values[:1]  # 闭合
        
        ax.plot(angles, values, 'o-', linewidth=2.5, color=color, 
               alpha=0.7, label=f'Neuron {neuron_id}')
        ax.fill(angles, values, color=color, alpha=0.1)
    
    # 设置标签和格式
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', '\n').title() for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_title(f'Top {top_n} Neurons: Multi-Dimensional Centrality Profile', 
                y=1.08, fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=ccfg.VISUALIZATION_DPI, bbox_inches='tight')
        print(f"Centrality radar chart saved to: {save_path}")
    plt.show()

def visualize_centrality_summary(results, G, save_dir=None):
    """Professional comprehensive analysis visualization of multiple centrality metrics"""
    centrality_dict = {name: data['centrality_scores'] for name, data in results.items()}
    
    print("\n=== Centrality Metrics Visualization Analysis ===")
    
    # Set up save directory if provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 1. Centrality metrics distributions
    print("\nGenerating centrality distributions visualization...")
    dist_save_path = os.path.join(save_dir, "centrality_distributions.png") if save_dir else None
    visualize_centrality_distributions(
        centrality_dict, 
        "Centrality Metrics Distribution Analysis",
        save_path=dist_save_path
    )
    
    # 2. Centrality metrics correlations
    print("Generating centrality relationships visualization...")
    rel_save_path = os.path.join(save_dir, "centrality_relationships.png") if save_dir else None
    visualize_centrality_relationships(
        centrality_dict, 
        "Centrality Metrics Correlation Analysis",
        save_path=rel_save_path
    )
    
    # 3. Detailed analysis for each centrality metric
    for centrality_name, data in results.items():
        print(f"\nAnalyzing {centrality_name} centrality...")
        
        # Centrality vs information measures
        info_save_path = os.path.join(save_dir, f"{centrality_name}_vs_information.png") if save_dir else None
        visualize_centrality_vs_information(
            centrality_name, 
            data['level_groups'], 
            data['level_ranges'],
            data['fisher_scores'], 
            data['accuracy_scores'], 
            data['centrality_scores'],
            save_path=info_save_path
        )
        
        # Multivariate Fisher regression analysis (if available)
        if 'multivariate_analysis' in data:
            print(f"Multivariate Fisher analysis results for {centrality_name}:")
            mv_results = data['multivariate_analysis']
            print(f"  Level centralities: {[f'{x:.3f}' for x in mv_results['level_centralities']]}")
            print(f"  Multivariate Fisher: {[f'{x:.3f}' for x in mv_results['level_multivariate_fishers']]}")
            print(f"  Individual Fisher means: {[f'{x:.3f}' for x in mv_results['level_individual_fishers']]}")
            print(f"  Neuron counts: {mv_results['level_neuron_counts']}")
        
        # Network topology visualization (for reasonable network sizes)
        if G.number_of_nodes() <= 300:
            print(f"Visualizing {centrality_name} network topology...")
            network_save_path = os.path.join(save_dir, f"{centrality_name}_network_topology.png") if save_dir else None
            visualize_network_with_centrality(
                G, data['centrality_scores'], 
                centrality_name=centrality_name,
                save_path=network_save_path
            )
        else:
            print(f"Skipping network topology for {centrality_name} (network too large: {G.number_of_nodes()} nodes)")
    
    # 4. 多分档比较分析
    print("\nGenerating multi-level comparison analysis...")
    for centrality_name, data in results.items():
        multi_save_path = os.path.join(save_dir, f"{centrality_name}_multi_level_comparison.png") if save_dir else None
        visualize_multi_level_comparison(
            data['centrality_scores'],
            data['fisher_scores'], 
            data['accuracy_scores'],
            centrality_name,
            levels_list=ccfg.MULTI_LEVELS,
            save_path=multi_save_path
        )
    
    # 5. 中心性热力图
    print("\nGenerating centrality heatmap...")
    heatmap_save_path = os.path.join(save_dir, "centrality_heatmap.png") if save_dir else None
    visualize_centrality_heatmap(centrality_dict, save_path=heatmap_save_path)
    
    # 6. 顶级神经元雷达图
    print("\nGenerating top neurons radar chart...")
    radar_save_path = os.path.join(save_dir, "top_neurons_radar.png") if save_dir else None
    visualize_centrality_radar_chart(centrality_dict, top_n=10, save_path=radar_save_path)
    
    print(f"\n=== Centrality Analysis Visualization Complete ===")
    if save_dir:
        print(f"All visualizations saved to: {save_dir}")

# %% 主脚本
if __name__ == "__main__":
    print("=== 拓扑中心性与神经信息关系分析 ===")
    
    # 加载数据
    print("\n1. 数据加载...")
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
        # 为了兼容性，创建模拟的neuron_data
        neuron_data = np.mean(segments, axis=0).T  # 转换为 (timepoints, neurons) 格式
        print(f"旧版数据维度: segments={segments.shape}, labels={len(new_labels)}, neuron_pos={neuron_pos.shape}")
        print(f"模拟neuron_data维度: {neuron_data.shape}")
        print("已切换到旧版数据加载模式")
    else:
        raise ValueError("无效的 LOADER_VERSION 配置")
    
    # RR神经元筛选
    print("\n2. RR神经元筛选...")
    rr_results = fast_rr_selection(segments, new_labels)
    rr_neurons = rr_results['rr_neurons'] if len(rr_results['rr_neurons']) > 0 else None
    
    # 主分析
    print("\n3. 拓扑中心性分析...")
    results, G, correlation_matrix, adjacency_matrix = analyze_centrality_information_relationship(
        neuron_data, segments, new_labels, rr_neurons)
    
    # 可视化分析结果
    print("\n4. 可视化分析结果...")
    results_dir = os.path.join(cfg.DATA_PATH, 'centrality_results')
    visualize_centrality_summary(results, G, save_dir=results_dir)
    
    print("\n=== 分析完成 ===")
    print(f"网络节点数: {G.number_of_nodes()}")
    print(f"网络边数: {G.number_of_edges()}")
    print(f"分析的中心性指标: {list(results.keys())}")