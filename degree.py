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
    N_LEVELS = 5                   # 分层数量
    
    # 可视化参数
    FIGURE_SIZE = (15, 10)
    SMALL_FIGURE_SIZE = (8, 6)
    
    # 分析参数
    CV_FOLDS = 5                   # 交叉验证折数
    RANDOM_STATE = 42              # 随机种子

ccfg = CentralityConfig()

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
def plot_centrality_vs_information(centrality_name, level_groups, level_ranges, 
                                  fisher_scores, accuracy_scores, centrality_scores):
    """
    绘制中心性与信息量关系图
    """
    fig, axes = plt.subplots(2, 2, figsize=ccfg.FIGURE_SIZE)
    
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
    
    # 1. 中心性分布箱线图
    ax1 = axes[0, 0]
    centrality_data = []
    labels = []
    for i, group in enumerate(level_groups):
        if len(group) > 0:
            group_scores = [centrality_scores[n] for n in group]
            centrality_data.extend(group_scores)
            labels.extend([f"L{i+1}"] * len(group_scores))
    
    if centrality_data:
        df = pd.DataFrame({'Centrality': centrality_data, 'Level': labels})
        sns.boxplot(data=df, x='Level', y='Centrality', ax=ax1)
        ax1.set_title(f'{centrality_name} Distribution by Level')
        ax1.set_ylabel(f'{centrality_name}')
    
    # 2. 中心性 vs Fisher信息
    ax2 = axes[0, 1]
    ax2.plot(level_centralities, level_fishers, 'bo-', linewidth=2, markersize=8)
    for i, (x, y) in enumerate(zip(level_centralities, level_fishers)):
        ax2.annotate(f'L{i+1}', (x, y), xytext=(5, 5), textcoords='offset points')
    ax2.set_xlabel(f'{centrality_name}')
    ax2.set_ylabel('Fisher Information')
    ax2.set_title(f'{centrality_name} vs Fisher Information')
    ax2.grid(True, alpha=0.3)
    
    # 3. 中心性 vs 分类准确率
    ax3 = axes[1, 0]
    ax3.plot(level_centralities, level_accuracies, 'ro-', linewidth=2, markersize=8)
    for i, (x, y) in enumerate(zip(level_centralities, level_accuracies)):
        ax3.annotate(f'L{i+1}', (x, y), xytext=(5, 5), textcoords='offset points')
    ax3.set_xlabel(f'{centrality_name}')
    ax3.set_ylabel('Classification Accuracy')
    ax3.set_title(f'{centrality_name} vs Classification Accuracy')
    ax3.grid(True, alpha=0.3)
    
    # 4. Fisher信息 vs 分类准确率
    ax4 = axes[1, 1]
    ax4.scatter(level_fishers, level_accuracies, c=level_centralities, 
               s=100, cmap='viridis', alpha=0.7)
    for i, (x, y) in enumerate(zip(level_fishers, level_accuracies)):
        ax4.annotate(f'L{i+1}', (x, y), xytext=(5, 5), textcoords='offset points')
    ax4.set_xlabel('Fisher Information')
    ax4.set_ylabel('Classification Accuracy')
    ax4.set_title('Fisher Information vs Classification Accuracy')
    ax4.grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label(f'{centrality_name}')
    
    plt.tight_layout()
    plt.show()
    
    # 计算相关性
    if len(level_centralities) > 2:
        corr_fisher, p_fisher = pearsonr(level_centralities, level_fishers)
        corr_acc, p_acc = pearsonr(level_centralities, level_accuracies)
        
        print(f"\n=== {centrality_name} 相关性分析 ===")
        print(f"中心性 vs Fisher信息: r={corr_fisher:.3f}, p={p_fisher:.3f}")
        print(f"中心性 vs 分类准确率: r={corr_acc:.3f}, p={p_acc:.3f}")

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
        
        # 可视化
        plot_centrality_vs_information(centrality_name, level_groups, level_ranges,
                                     fisher_scores, accuracy_scores, centrality_scores)
        
        # 保存结果
        results[centrality_name] = {
            'centrality_scores': centrality_scores,
            'level_groups': level_groups,
            'level_ranges': level_ranges,
            'fisher_scores': fisher_scores,
            'accuracy_scores': accuracy_scores
        }
    
    return results, G, correlation_matrix, adjacency_matrix

# %% 主脚本
if __name__ == "__main__":
    print("=== 拓扑中心性与神经信息关系分析 ===")
    
    # 加载数据
    print("\n1. 数据加载...")
    neuron_data, neuron_pos, trigger_data, stimulus_data = load_data(cfg.DATA_PATH)
    
    # 数据分割
    segments, labels = segment_neuron_data(neuron_data, trigger_data, stimulus_data)
    
    # 重新分类标签
    new_labels = reclassify_labels(stimulus_data)
    
    # RR神经元筛选
    print("\n2. RR神经元筛选...")
    rr_results = fast_rr_selection(segments, new_labels)
    rr_neurons = rr_results['rr_neurons'] if len(rr_results['rr_neurons']) > 0 else None
    
    # 主分析
    print("\n3. 拓扑中心性分析...")
    results, G, correlation_matrix, adjacency_matrix = analyze_centrality_information_relationship(
        neuron_data, segments, new_labels, rr_neurons)
    
    print("\n=== 分析完成 ===")
    print(f"网络节点数: {G.number_of_nodes()}")
    print(f"网络边数: {G.number_of_edges()}")
    print(f"分析的中心性指标: {list(results.keys())}")