# 噪音相关性与神经元置换分析
# guiy24@mails.tsinghua.edu.cn
# 2025-08-31
# 分析神经元置换对噪音相关性的影响，及其与功能连接的关系

# %% 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind, wilcoxon
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import networkx as nx
import warnings
from collections import defaultdict
from itertools import combinations
import os
warnings.filterwarnings('ignore')

# 导入项目模块
from loaddata import (
    load_data, segment_neuron_data, reclassify_labels, 
    fast_rr_selection, preprocess_neural_data, calculate_fisher_information, cfg
)
from network import NetworkConfig, compute_network_metrics

# %% 配置参数
class NoiseCorrelationConfig:
    """噪音相关性分析配置"""
    
    # 噪音相关性参数
    NOISE_THRESHOLD_PERCENTILE = 80    # 噪音相关性阈值百分位
    MIN_TRIALS_PER_CONDITION = 10      # 每个条件最少试次数
    
    # 神经元置换参数
    SHUFFLE_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 置换比例
    
    # Fisher信息计算参数
    FISHER_START_TIME = 20             # Fisher信息时间窗口开始点
    FISHER_END_TIME = 30               # Fisher信息时间窗口结束点
    
    # 网络构建参数
    NETWORK_METHOD = 'density'         # 网络构建方法: 'threshold' 或 'density'
    NETWORK_THRESHOLD = 0.3            # 绝对阈值（当method='threshold'时）
    NETWORK_DENSITY = 0.1              # 网络密度（当method='density'时）
    
    # 枢纽神经元定义参数
    HUB_PERCENTILE = 90                # 枢纽神经元度数百分位阈值
    PERIPHERAL_PERCENTILE = 25         # 边缘神经元度数百分位阈值
    
    # 可视化参数
    FIGSIZE = (12, 8)                  # 图形大小
    FIGURE_SIZE_LARGE = (18, 12)       # 大图尺寸
    FIGURE_SIZE_EXTRA_LARGE = (20, 15) # 特大图尺寸
    DPI = 300                          # 图形分辨率
    VISUALIZATION_STYLE = 'seaborn-v0_8-whitegrid'  # 科研绘图风格
    
    # 科研绘图配色方案
    COLORS = {
        'noise': '#E74C3C',            # 噪音相关性（红色）
        'signal': '#3498DB',           # 信号相关性（蓝色）
        'fisher': '#2ECC71',           # Fisher信息（绿色）
        'network': '#F39C12',          # 网络指标（橙色）
        'hub': '#9B59B6',             # 枢纽神经元（紫色）
        'peripheral': '#95A5A6',       # 边缘神经元（灰色）
        'primary': '#2E86AB',          # 主要颜色
        'secondary': '#A23B72',        # 次要颜色
        'accent': '#F18F01',           # 强调色
        'neutral': '#6C757D',          # 中性色
        'background': '#F8F9FA'        # 背景色
    }
    
    # 结果保存路径
    RESULTS_DIR = 'results/noise_correlation'
    
    @classmethod
    def ensure_results_dir(cls):
        """确保结果目录存在"""
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)

# 实例化配置
ncfg = NoiseCorrelationConfig()

# %% 科研绘图风格设置
def setup_noise_plot_style():
    """设置噪音相关性分析科研绘图风格"""
    plt.style.use(ncfg.VISUALIZATION_STYLE)
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

# %% 核心函数定义

def calculate_noise_correlation(neural_data, labels):
    """
    计算每个条件下的噪音相关性矩阵
    
    噪音定义为每个神经元在当前试次的活动与该神经元在该条件下平均响应的差值
    
    Parameters:
    -----------
    neural_data : ndarray, shape (n_trials, n_neurons, n_timepoints)
        神经活动数据
    labels : ndarray, shape (n_trials,)
        试次标签
        
    Returns:
    --------
    noise_correlations : dict
        每个条件的噪音相关性矩阵 {condition: correlation_matrix}
    """
    print("计算噪音相关性矩阵...")
    
    unique_labels = np.unique(labels)
    noise_correlations = {}
    
    for condition in unique_labels:
        condition_indices = labels == condition
        condition_data = neural_data[condition_indices]  # (n_trials_cond, n_neurons, n_timepoints)
        
        if condition_data.shape[0] < ncfg.MIN_TRIALS_PER_CONDITION:
            print(f"警告: 条件 {condition} 试次数不足 ({condition_data.shape[0]} < {ncfg.MIN_TRIALS_PER_CONDITION})")
            continue
        
        # 计算该条件下每个神经元的平均响应
        mean_response = np.mean(condition_data, axis=0)  # (n_neurons, n_timepoints)
        
        # 计算每个试次的噪音（残差）
        noise_activities = condition_data - mean_response  # (n_trials_cond, n_neurons, n_timepoints)
        
        # 将时间维度展平，计算神经元间的噪音相关性
        # 每个神经元的噪音时间序列: (n_trials_cond * n_timepoints,)
        n_trials_cond, n_neurons, n_timepoints = noise_activities.shape
        noise_flattened = noise_activities.transpose(1, 0, 2).reshape(n_neurons, -1)
        
        # 计算噪音相关性矩阵
        noise_corr_matrix = np.corrcoef(noise_flattened)
        noise_corr_matrix = np.nan_to_num(noise_corr_matrix)  # 处理NaN值
        
        noise_correlations[condition] = noise_corr_matrix
        print(f"  条件 {condition}: 试次数 {n_trials_cond}, 矩阵维度 {noise_corr_matrix.shape}")
    
    return noise_correlations


def calculate_signal_correlation(neural_data, labels):
    """
    计算信号相关性（功能连接）
    信号相关性是每个条件下，先计算每个试次的相关性矩阵，然后对所有试次的相关性矩阵取平均
    
    Parameters:
    -----------
    neural_data : ndarray, shape (n_trials, n_neurons, n_timepoints)
        神经活动数据
    labels : ndarray, shape (n_trials,)
        试次标签
        
    Returns:
    --------
    signal_correlations : dict
        每个条件的信号相关性矩阵 {condition: correlation_matrix}
    """
    print("计算信号相关性矩阵（试次相关矩阵平均）...")
    
    unique_labels = np.unique(labels)
    signal_correlations = {}
    
    for condition in unique_labels:
        condition_indices = labels == condition
        condition_data = neural_data[condition_indices]
        
        if condition_data.shape[0] < ncfg.MIN_TRIALS_PER_CONDITION:
            continue
        
        n_trials_cond, n_neurons, n_timepoints = condition_data.shape
        correlation_matrices = []
        
        # 对每个试次计算相关性矩阵
        for trial_idx in range(n_trials_cond):
            trial_data = condition_data[trial_idx]  # (n_neurons, n_timepoints)
            
            # 计算该试次中神经元间的相关性
            trial_corr_matrix = np.corrcoef(trial_data)
            trial_corr_matrix = np.nan_to_num(trial_corr_matrix)  # 处理NaN值
            
            correlation_matrices.append(trial_corr_matrix)
        
        # 对所有试次的相关性矩阵取平均
        signal_corr_matrix = np.mean(correlation_matrices, axis=0)
        
        signal_correlations[condition] = signal_corr_matrix
        print(f"  条件 {condition}: 试次数 {n_trials_cond}, 信号相关矩阵维度 {signal_corr_matrix.shape}")
    
    return signal_correlations


def shuffle_within_condition(neural_data, labels, fraction_to_shuffle=0.5, neurons_to_shuffle=None):
    """
    在每个条件内对指定比例或指定神经元进行试次间的活动打乱
    
    Parameters:
    -----------
    neural_data : ndarray, shape (n_trials, n_neurons, n_timepoints)
        原始神经活动数据
    labels : ndarray, shape (n_trials,)
        试次标签
    fraction_to_shuffle : float
        要打乱的神经元比例 (0-1)
    neurons_to_shuffle : array-like, optional
        指定要打乱的神经元索引列表
        
    Returns:
    --------
    shuffled_data : ndarray
        打乱后的神经活动数据
    """
    shuffled_data = neural_data.copy()
    n_neurons = neural_data.shape[1]
    
    # 确定要打乱的神经元
    if neurons_to_shuffle is None:
        n_to_shuffle = int(n_neurons * fraction_to_shuffle)
        neurons_to_shuffle = np.random.choice(n_neurons, size=n_to_shuffle, replace=False)
    
    # 对每个条件分别进行打乱
    for condition in np.unique(labels):
        condition_indices = np.where(labels == condition)[0]
        
        # 对选中的神经元进行打乱
        for neuron_idx in neurons_to_shuffle:
            original_activity = shuffled_data[condition_indices, neuron_idx, :]
            shuffled_indices = np.random.permutation(len(condition_indices))
            shuffled_data[condition_indices, neuron_idx, :] = original_activity[shuffled_indices, :]
    
    return shuffled_data


def analyze_shuffle_effect_with_fisher(neural_data, labels, rr_indices, shuffle_fractions=None, n_iterations=10):
    """
    分析不同程度神经元置换对Fisher信息的影响（多次置换取平均）
    
    Parameters:
    -----------
    neural_data : ndarray, shape (n_trials, n_neurons, n_timepoints)
        神经活动数据
    labels : ndarray, shape (n_trials,)
        试次标签
    rr_indices : list
        RR神经元索引
    shuffle_fractions : list, optional
        置换比例列表
    n_iterations : int, optional
        每个置换比例的重复次数，默认10次
        
    Returns:
    --------
    results : dict
        分析结果，包括原始Fisher信息和不同置换程度的Fisher信息均值和标准差
    """
    if shuffle_fractions is None:
        shuffle_fractions = ncfg.SHUFFLE_FRACTIONS
    
    print(f"分析神经元置换对Fisher信息的影响（时间窗口{ncfg.FISHER_START_TIME}-{ncfg.FISHER_END_TIME}）...")
    print(f"每个置换比例重复{n_iterations}次取平均")
    
    # 计算原始数据的Fisher信息
    original_fi_mean = calculate_fisher_information_stimulus_window(neural_data, labels, rr_indices)
    
    print(f"原始Fisher信息: {original_fi_mean:.4f}")
    
    # 分析不同置换程度的影响（多次置换取平均）
    shuffle_results = {}
    
    for fraction in shuffle_fractions:
        print(f"分析置换比例: {fraction:.1f} ({n_iterations}次迭代)")
        
        fisher_values = []
        
        # 多次置换取平均
        for iteration in range(n_iterations):
            # 打乱数据（每次随机）
            shuffled_data = shuffle_within_condition(neural_data, labels, fraction)
            
            # 计算打乱后的Fisher信息
            shuffled_fisher = calculate_fisher_information_stimulus_window(
                shuffled_data, labels, rr_indices)
            fisher_values.append(shuffled_fisher)
        
        # 计算统计量
        fisher_mean = np.mean(fisher_values)
        fisher_std = np.std(fisher_values)
        degradation_mean = (original_fi_mean - fisher_mean) / original_fi_mean * 100 if original_fi_mean > 0 else 0.0
        
        shuffle_results[fraction] = {
            'fisher_value': fisher_mean,
            'fisher_std': fisher_std,
            'fisher_values': fisher_values,  # 保存所有值用于进一步分析
            'degradation_percent': degradation_mean
        }
        
        print(f"  置换后Fisher信息: {fisher_mean:.4f} ± {fisher_std:.4f} (下降 {degradation_mean:.1f}%)")
    
    results = {
        'original_fisher_mean': original_fi_mean,
        'shuffle_results': shuffle_results,
        'shuffle_fractions': shuffle_fractions,
        'n_iterations': n_iterations
    }
    
    return results


def compare_noise_signal_correlations(noise_correlations, signal_correlations):
    """
    比较噪音相关性和信号相关性的关系
    
    Parameters:
    -----------
    noise_correlations : dict
        噪音相关性矩阵
    signal_correlations : dict
        信号相关性矩阵
        
    Returns:
    --------
    comparison_results : dict
        比较结果
    """
    print("比较噪音相关性与信号相关性的关系...")
    
    comparison_results = {}
    
    for condition in noise_correlations.keys():
        if condition not in signal_correlations:
            continue
        
        noise_matrix = noise_correlations[condition]
        signal_matrix = signal_correlations[condition]
        
        if noise_matrix.shape != signal_matrix.shape:
            print(f"警告: 条件 {condition} 的矩阵维度不匹配")
            continue
        
        # 提取上三角矩阵（排除对角线）
        n_neurons = noise_matrix.shape[0]
        triu_indices = np.triu_indices(n_neurons, k=1)
        
        noise_values = noise_matrix[triu_indices]
        signal_values = signal_matrix[triu_indices]
        
        # 计算相关系数
        r_noise_signal, p_value = pearsonr(noise_values, signal_values)
        
        comparison_results[condition] = {
            'correlation': r_noise_signal,
            'p_value': p_value,
            'noise_values': noise_values,
            'signal_values': signal_values
        }
        
        print(f"条件 {condition}: 噪音-信号相关系数 r = {r_noise_signal:.4f} (p = {p_value:.4f})")
    
    return comparison_results


def calculate_fisher_information_stimulus_window(segments, labels, rr_neurons, 
                                               start_time=None, end_time=None):
    """
    计算指定时间窗口的多变量Fisher信息（包含PCA降维，与loaddata.py一致）
    
    参数:
    segments: 神经数据片段 (trials, neurons, timepoints)
    labels: 标签数组
    rr_neurons: RR神经元索引
    start_time: 开始时间点（默认使用配置参数）
    end_time: 结束时间点（默认使用配置参数）
    
    返回:
    fisher_score: 该时间窗口的多变量Fisher信息分数
    """
    from scipy.linalg import pinv, eigvals
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # 使用配置参数作为默认值
    if start_time is None:
        start_time = ncfg.FISHER_START_TIME
    if end_time is None:
        end_time = ncfg.FISHER_END_TIME
    
    # 过滤有效数据和RR神经元
    valid_mask = labels != 0
    valid_segments = segments[valid_mask][:, rr_neurons, :]
    valid_labels = labels[valid_mask]
    
    n_trials, n_neurons, n_timepoints = valid_segments.shape
    
    # 检查时间窗口是否合理
    if end_time > n_timepoints:
        end_time = n_timepoints
    if start_time >= end_time:
        start_time = max(0, end_time - 1)
    
    print(f"计算多变量Fisher信息，时间窗口: {start_time}-{end_time}, 神经元数: {n_neurons}")
    
    # 提取指定时间窗口的数据并对时间维度取平均
    window_data = valid_segments[:, :, start_time:end_time]  # (trials, neurons, window_length)
    mean_data = np.mean(window_data, axis=2)  # (trials, neurons)
    
    # 只使用类别1和2进行Fisher信息计算，排除类别3（噪音条件）
    target_labels = [1, 2]
    target_mask = np.isin(valid_labels, target_labels)
    
    if np.sum(target_mask) < 10:  # 至少需要10个样本
        return 0.0
        
    # 过滤数据和标签
    filtered_data = mean_data[target_mask]
    filtered_labels = valid_labels[target_mask]
    
    unique_labels = np.unique(filtered_labels)
    if len(unique_labels) < 2:
        return 0.0
    
    n_trials_filtered, n_neurons = filtered_data.shape
    n_classes = len(unique_labels)
    
    # 检查样本数是否足够
    min_samples_per_class = min([np.sum(filtered_labels == label) for label in unique_labels])
    if min_samples_per_class < 2:
        return 0.0
    
    # 数据标准化避免数值问题
    scaler = StandardScaler()
    mean_data_scaled = scaler.fit_transform(filtered_data)
    
    # 关键改进：当神经元数量接近或超过试次数时，使用PCA降维
    if n_neurons > n_trials_filtered * 0.5:  # 当神经元数 > 试次数的50%时进行降维
        # 目标维度：试次数的1/3，但至少保留2维，最多不超过15维
        target_dim = max(2, min(15, n_trials_filtered // 3))
        
        print(f"使用PCA降维: {n_neurons}维 -> {target_dim}维 (试次数: {n_trials_filtered})")
        
        # 执行PCA降维
        pca = PCA(n_components=target_dim, random_state=42)
        mean_data_scaled = pca.fit_transform(mean_data_scaled)
        
        # 更新维度信息
        n_neurons = target_dim
        print(f"PCA解释方差比: {np.sum(pca.explained_variance_ratio_):.3f}")
    
    # 现在在降维后的数据上计算多变量Fisher信息
    # 计算总体均值
    grand_mean = np.mean(mean_data_scaled, axis=0).astype(np.float64)
    
    # 计算类别均值和样本数
    class_means = []
    class_sizes = []
    
    for label in unique_labels:
        label_mask = filtered_labels == label
        label_data = mean_data_scaled[label_mask]
        if len(label_data) > 0:
            class_means.append(np.mean(label_data, axis=0).astype(np.float64))
            class_sizes.append(len(label_data))
        else:
            class_means.append(grand_mean)
            class_sizes.append(0)
    
    class_means = np.array(class_means).astype(np.float64)
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
        label_mask = filtered_labels == label
        label_data = mean_data_scaled[label_mask]
        if len(label_data) > 1:
            class_mean = np.mean(label_data, axis=0).astype(np.float64)
            centered_data = (label_data - class_mean).astype(np.float64)
            S_w += np.dot(centered_data.T, centered_data).astype(np.float64)
    
    # 添加正则化项避免奇异矩阵
    try:
        eigenvals = eigvals(S_w).real.astype(np.float64)
        eigenvals = eigenvals[eigenvals > 0]
        if len(eigenvals) > 1:
            condition_number = float(np.max(eigenvals) / np.min(eigenvals))
            reg_strength = max(1e-6, float(np.max(eigenvals)) * 1e-10 * condition_number)
        else:
            reg_strength = 1e-3
    except:
        reg_strength = 1e-3
    
    regularization = (reg_strength * np.eye(n_neurons)).astype(np.float64)
    S_w += regularization
    
    try:
        # 使用更稳定的方法计算多变量Fisher信息
        # 方法1: 直接计算trace(S_w^(-1) * S_b)
        S_w_inv = pinv(S_w).astype(np.float64)
        fisher_matrix = np.dot(S_w_inv, S_b).astype(np.float64)
        fisher_score = float(np.trace(fisher_matrix).real)  # 确保实数
        
        # 数值稳定性检查
        if np.isnan(fisher_score) or np.isinf(fisher_score) or fisher_score < 0:
            # 方法2: 使用广义特征值问题求解
            from scipy.linalg import eigh
            try:
                eigenvals, _ = eigh(S_b, S_w)
                eigenvals_real = eigenvals.real.astype(np.float64)
                fisher_score = float(np.sum(eigenvals_real[eigenvals_real > 0]))
            except:
                # 方法3: 简化的多变量Fisher比率
                trace_s_b = float(np.trace(S_b).real)
                trace_s_w = float(np.trace(S_w).real)
                fisher_score = trace_s_b / (trace_s_w + 1e-10)
        
        # 确保返回非负有限值
        fisher_score = max(0.0, float(fisher_score))
        if not np.isfinite(fisher_score):
            fisher_score = 0.0
        
    except Exception as e:
        # 最后的备选方案：使用简化版本
        try:
            trace_s_b = float(np.trace(S_b).real)
            trace_s_w = float(np.trace(S_w).real)
            fisher_score = trace_s_b / (trace_s_w + 1e-10)
            fisher_score = max(0.0, float(fisher_score))
        except:
            fisher_score = 0.0
    
    return fisher_score


def build_networks_from_correlations(correlations, method='threshold', threshold=0.3, density=0.1):
    """
    从相关性矩阵构建网络并计算网络指标
    
    Parameters:
    -----------
    correlations : dict
        相关性矩阵字典
    method : str
        网络构建方法: 'threshold' (绝对阈值) 或 'density' (相对密度)
    threshold : float
        绝对相关性阈值（当method='threshold'时使用）
    density : float
        网络密度阈值（当method='density'时使用）
        
    Returns:
    --------
    network_metrics : dict
        每个条件的网络指标
    """
    if method == 'threshold':
        print(f"构建网络并计算指标（绝对阈值: {threshold}）...")
    else:
        print(f"构建网络并计算指标（相对密度: {density}）...")
    
    network_metrics = {}
    
    for condition, corr_matrix in correlations.items():
        print(f"  处理条件 {condition}")
        
        # 获取绝对相关性值，去除对角线
        abs_corr = np.abs(corr_matrix)
        np.fill_diagonal(abs_corr, 0)
        
        if method == 'threshold':
            # 绝对阈值方法：超过阈值的连接保留
            adj_matrix = abs_corr > threshold
        else:
            # 相对密度方法：保留前N%最强的连接
            n_nodes = abs_corr.shape[0]
            n_possible_edges = n_nodes * (n_nodes - 1) // 2  # 上三角矩阵
            n_edges_to_keep = int(n_possible_edges * density)
            
            # 获取上三角部分的相关性值
            triu_indices = np.triu_indices_from(abs_corr, k=1)
            triu_values = abs_corr[triu_indices]
            
            # 找到阈值：保留最强的density比例的连接
            if n_edges_to_keep > 0:
                actual_threshold = np.partition(triu_values, -n_edges_to_keep)[-n_edges_to_keep]
                adj_matrix = abs_corr >= actual_threshold
                # 确保对角线为0
                np.fill_diagonal(adj_matrix, 0)
            else:
                adj_matrix = np.zeros_like(abs_corr, dtype=bool)
        
        # 计算网络指标
        metrics = compute_network_metrics(adj_matrix.astype(int))
        network_metrics[condition] = metrics
        
        # 保存邻接矩阵用于后续分析
        metrics['adjacency_matrix'] = adj_matrix
        
        # 打印关键指标
        print(f"    节点数: {metrics['n_nodes']}, 边数: {metrics['n_edges']}")
        print(f"    网络密度: {metrics['density']:.4f}")
        print(f"    平均度: {metrics['avg_degree']:.2f}")
        print(f"    平均聚类系数: {metrics['avg_clustering']:.4f}")
    
    return network_metrics


def compare_network_metrics(noise_metrics, signal_metrics):
    """
    比较噪音相关性和信号相关性网络的指标差异
    
    Parameters:
    -----------
    noise_metrics : dict
        噪音相关性网络指标
    signal_metrics : dict
        信号相关性网络指标
        
    Returns:
    --------
    comparison : dict
        网络指标比较结果
    """
    print("比较噪音与信号相关性网络指标...")
    
    comparison = {}
    common_conditions = set(noise_metrics.keys()) & set(signal_metrics.keys())
    
    for condition in common_conditions:
        noise_m = noise_metrics[condition]
        signal_m = signal_metrics[condition]
        
        condition_comparison = {}
        
        # 比较关键指标
        key_metrics = ['density', 'avg_degree', 'avg_clustering']
        for metric in key_metrics:
            if metric in noise_m and metric in signal_m:
                noise_val = noise_m[metric]
                signal_val = signal_m[metric]
                difference = noise_val - signal_val
                relative_diff = difference / (signal_val + 1e-8) * 100
                
                condition_comparison[metric] = {
                    'noise': noise_val,
                    'signal': signal_val,
                    'difference': difference,
                    'relative_diff_percent': relative_diff
                }
        
        comparison[condition] = condition_comparison
        
        print(f"\n条件 {condition} 网络指标比较:")
        for metric, values in condition_comparison.items():
            print(f"  {metric}: 噪音={values['noise']:.4f}, 信号={values['signal']:.4f}, "
                  f"相对差异={values['relative_diff_percent']:.1f}%")
    
    return comparison


def analyze_hub_peripheral_noise_correlation(noise_correlations, noise_network_metrics):
    """
    分析枢纽神经元和边缘神经元的噪音相关性差异
    
    Parameters:
    -----------
    noise_correlations : dict
        噪音相关性矩阵
    noise_network_metrics : dict
        噪音相关性网络指标（包含邻接矩阵）
        
    Returns:
    --------
    hub_analysis : dict
        枢纽-边缘神经元分析结果
    """
    print("分析枢纽神经元与边缘神经元的噪音相关性差异...")
    
    hub_analysis = {}
    
    for condition in noise_correlations.keys():
        if condition not in noise_network_metrics:
            continue
            
        print(f"  分析条件 {condition}")
        
        # 获取噪音相关性矩阵和网络度
        noise_matrix = noise_correlations[condition]
        adj_matrix = noise_network_metrics[condition]['adjacency_matrix']
        
        # 计算每个节点的度（连接数）
        degrees = np.sum(adj_matrix, axis=1)
        
        # 定义枢纽神经元和边缘神经元（使用配置参数）
        degree_hub = np.percentile(degrees, ncfg.HUB_PERCENTILE)
        degree_peripheral = np.percentile(degrees, ncfg.PERIPHERAL_PERCENTILE)
        
        hub_indices = np.where(degrees >= degree_hub)[0]
        peripheral_indices = np.where(degrees <= degree_peripheral)[0]
        
        print(f"    枢纽神经元数: {len(hub_indices)} (度 >= {degree_hub})")
        print(f"    边缘神经元数: {len(peripheral_indices)} (度 <= {degree_peripheral})")
        
        if len(hub_indices) == 0 or len(peripheral_indices) == 0:
            print(f"    警告: 条件 {condition} 枢纽或边缘神经元数量为0，跳过")
            continue
        
        # 计算不同类型神经元间的噪音相关性
        # 1. 枢纽-枢纽相关性
        hub_hub_correlations = []
        for i, hub_i in enumerate(hub_indices):
            for j, hub_j in enumerate(hub_indices):
                if i < j:  # 只考虑上三角
                    hub_hub_correlations.append(abs(noise_matrix[hub_i, hub_j]))
        
        # 2. 边缘-边缘相关性
        peripheral_peripheral_correlations = []
        for i, per_i in enumerate(peripheral_indices):
            for j, per_j in enumerate(peripheral_indices):
                if i < j:  # 只考虑上三角
                    peripheral_peripheral_correlations.append(abs(noise_matrix[per_i, per_j]))
        
        # 3. 枢纽-边缘相关性
        hub_peripheral_correlations = []
        for hub_i in hub_indices:
            for per_j in peripheral_indices:
                hub_peripheral_correlations.append(abs(noise_matrix[hub_i, per_j]))
        
        # 统计分析
        hub_hub_mean = np.mean(hub_hub_correlations) if hub_hub_correlations else 0
        per_per_mean = np.mean(peripheral_peripheral_correlations) if peripheral_peripheral_correlations else 0
        hub_per_mean = np.mean(hub_peripheral_correlations) if hub_peripheral_correlations else 0
        
        # 进行显著性检验
        from scipy.stats import mannwhitneyu
        
        # 枢纽vs边缘内部相关性
        if hub_hub_correlations and peripheral_peripheral_correlations:
            try:
                hub_vs_per_stat, hub_vs_per_p = mannwhitneyu(
                    hub_hub_correlations, peripheral_peripheral_correlations, 
                    alternative='two-sided'
                )
            except:
                hub_vs_per_stat, hub_vs_per_p = np.nan, np.nan
        else:
            hub_vs_per_stat, hub_vs_per_p = np.nan, np.nan
        
        # 枢纽内部 vs 枢纽-边缘相关性
        if hub_hub_correlations and hub_peripheral_correlations:
            try:
                hub_internal_vs_cross_stat, hub_internal_vs_cross_p = mannwhitneyu(
                    hub_hub_correlations, hub_peripheral_correlations,
                    alternative='two-sided'
                )
            except:
                hub_internal_vs_cross_stat, hub_internal_vs_cross_p = np.nan, np.nan
        else:
            hub_internal_vs_cross_stat, hub_internal_vs_cross_p = np.nan, np.nan
        
        condition_result = {
            'n_hubs': len(hub_indices),
            'n_peripheral': len(peripheral_indices),
            'degree_threshold_hub': degree_hub,
            'degree_threshold_peripheral': degree_peripheral,
            'hub_hub_correlation': {
                'mean': hub_hub_mean,
                'std': np.std(hub_hub_correlations) if hub_hub_correlations else 0,
                'n_pairs': len(hub_hub_correlations)
            },
            'peripheral_peripheral_correlation': {
                'mean': per_per_mean,
                'std': np.std(peripheral_peripheral_correlations) if peripheral_peripheral_correlations else 0,
                'n_pairs': len(peripheral_peripheral_correlations)
            },
            'hub_peripheral_correlation': {
                'mean': hub_per_mean,
                'std': np.std(hub_peripheral_correlations) if hub_peripheral_correlations else 0,
                'n_pairs': len(hub_peripheral_correlations)
            },
            'statistics': {
                'hub_vs_peripheral_internal': {
                    'statistic': hub_vs_per_stat,
                    'p_value': hub_vs_per_p
                },
                'hub_internal_vs_cross': {
                    'statistic': hub_internal_vs_cross_stat,
                    'p_value': hub_internal_vs_cross_p
                }
            }
        }
        
        hub_analysis[condition] = condition_result
        
        print(f"    枢纽-枢纽噪音相关性: {hub_hub_mean:.4f} ± {np.std(hub_hub_correlations) if hub_hub_correlations else 0:.4f}")
        print(f"    边缘-边缘噪音相关性: {per_per_mean:.4f} ± {np.std(peripheral_peripheral_correlations) if peripheral_peripheral_correlations else 0:.4f}")
        print(f"    枢纽-边缘噪音相关性: {hub_per_mean:.4f} ± {np.std(hub_peripheral_correlations) if hub_peripheral_correlations else 0:.4f}")
        print(f"    枢纽内部 vs 边缘内部 p值: {hub_vs_per_p:.4f}" if not np.isnan(hub_vs_per_p) else "    枢纽内部 vs 边缘内部 p值: NaN")
    
    return hub_analysis


# %% 可视化函数

def visualize_noise_signal_comparison(comparison_results, title="Noise-Signal Correlation Comparison", save_path=None):
    """专业可视化噪音和信号相关性的比较"""
    setup_noise_plot_style()
    
    conditions = list(comparison_results.keys())
    noise_means = [np.mean(comparison_results[cond]['noise_values']) for cond in conditions]
    signal_means = [np.mean(comparison_results[cond]['signal_values']) for cond in conditions]
    noise_stds = [np.std(comparison_results[cond]['noise_values']) for cond in conditions]
    signal_stds = [np.std(comparison_results[cond]['signal_values']) for cond in conditions]
    
    fig, axes = plt.subplots(1, 2, figsize=ncfg.FIGURE_SIZE_LARGE)
    
    # 专业条形图比较
    ax1 = axes[0]
    x = np.arange(len(conditions))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, noise_means, width, yerr=noise_stds,
                   label='Noise Correlation', alpha=0.8, 
                   color=ncfg.COLORS['noise'], edgecolor='black', linewidth=1.2,
                   capsize=5, error_kw={'linewidth': 1.5, 'capthick': 1.5})
    bars2 = ax1.bar(x + width/2, signal_means, width, yerr=signal_stds,
                   label='Signal Correlation', alpha=0.8,
                   color=ncfg.COLORS['signal'], edgecolor='black', linewidth=1.2,
                   capsize=5, error_kw={'linewidth': 1.5, 'capthick': 1.5})
    
    ax1.set_xlabel('Condition')
    ax1.set_ylabel('Correlation Strength')
    ax1.set_title('Noise vs Signal Correlation')
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, rotation=45, ha='right')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 专业散点图比较
    ax2 = axes[1]
    scatter = ax2.scatter(noise_means, signal_means, 
                         s=150, alpha=0.8, 
                         color=ncfg.COLORS['primary'],
                         edgecolors='black', linewidth=1.5)
    
    # 添加对角线
    all_values = noise_means + signal_means
    min_val, max_val = min(all_values), max(all_values)
    ax2.plot([min_val, max_val], [min_val, max_val], 
             color=ncfg.COLORS['neutral'], linestyle='--', 
             linewidth=2, alpha=0.7, label='Unity Line')
    
    # 计算相关系数
    if len(noise_means) > 2:
        corr_coef, p_val = pearsonr(noise_means, signal_means)
        # 添加回归线
        z = np.polyfit(noise_means, signal_means, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(noise_means), max(noise_means), 100)
        ax2.plot(x_line, p(x_line), 
                color=ncfg.COLORS['accent'], linestyle='-', 
                linewidth=2, alpha=0.8, 
                label=f'Regression (r={corr_coef:.3f})')
    
    ax2.set_xlabel('Noise Correlation')
    ax2.set_ylabel('Signal Correlation')
    ax2.set_title('Noise vs Signal Correlation')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, y=0.98)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=ncfg.DPI, bbox_inches='tight')
    plt.show()

def visualize_neuron_pairs_scatter(noise_correlations, signal_correlations, 
                                 title="Neuron Pairs: Noise vs Signal Correlations", save_path=None):
    """
    绘制神经元对的噪音相关性与信号相关性散点图
    
    Parameters:
    -----------
    noise_correlations : dict
        噪音相关性矩阵 {condition: correlation_matrix}
    signal_correlations : dict
        信号相关性矩阵 {condition: correlation_matrix}
    title : str
        图标题
    save_path : str, optional
        保存路径
    """
    print("生成神经元对噪音-信号相关性散点图...")
    setup_noise_plot_style()
    
    # 准备数据
    conditions = list(noise_correlations.keys())
    n_conditions = len(conditions)
    
    if n_conditions == 0:
        print("没有可用的相关性数据")
        return
    
    # 创建子图布局
    if n_conditions == 1:
        fig, ax = plt.subplots(1, 1, figsize=ncfg.FIGSIZE)
        axes = [ax]
    elif n_conditions <= 3:
        fig, axes = plt.subplots(1, n_conditions, figsize=(6*n_conditions, 6))
        if n_conditions == 1:
            axes = [axes]
    else:
        # 超过3个条件使用2行布局
        cols = (n_conditions + 1) // 2
        fig, axes = plt.subplots(2, cols, figsize=(6*cols, 12))
        axes = axes.flatten() if n_conditions > 1 else [axes]
    
    # 为每个条件生成散点图
    for idx, condition in enumerate(conditions):
        ax = axes[idx]
        
        if condition not in signal_correlations:
            ax.text(0.5, 0.5, f'Condition {condition}\nNo signal data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Condition {condition}')
            continue
        
        noise_matrix = noise_correlations[condition]
        signal_matrix = signal_correlations[condition]
        
        if noise_matrix.shape != signal_matrix.shape:
            ax.text(0.5, 0.5, f'Condition {condition}\nMatrix size mismatch', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Condition {condition}')
            continue
        
        # 提取上三角矩阵的值（排除对角线）
        n_neurons = noise_matrix.shape[0]
        triu_indices = np.triu_indices(n_neurons, k=1)
        
        noise_values = noise_matrix[triu_indices]
        signal_values = signal_matrix[triu_indices]
        
        # 移除NaN和无穷值
        valid_mask = np.isfinite(noise_values) & np.isfinite(signal_values)
        noise_values = noise_values[valid_mask]
        signal_values = signal_values[valid_mask]
        
        if len(noise_values) == 0:
            ax.text(0.5, 0.5, f'Condition {condition}\nNo valid data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Condition {condition}')
            continue
        
        print(f"  条件 {condition}: {len(noise_values)} 个神经元对")
        
        # 根据数据点数量选择可视化策略
        if len(noise_values) > 5000:
            # 数据点太多，使用hexbin
            hb = ax.hexbin(noise_values, signal_values, gridsize=50, cmap='Blues', 
                          alpha=0.8, mincnt=1)
            plt.colorbar(hb, ax=ax, shrink=0.8, label='Count')
        elif len(noise_values) > 1000:
            # 中等数量，使用密度散点图
            ax.scatter(noise_values, signal_values, s=8, alpha=0.4, 
                      color=ncfg.COLORS['primary'], edgecolors='none')
        else:
            # 数据点较少，使用标准散点图
            ax.scatter(noise_values, signal_values, s=20, alpha=0.6, 
                      color=ncfg.COLORS['primary'], edgecolors='black', linewidth=0.3)
        
        # 计算相关系数
        try:
            corr_coef, p_val = pearsonr(noise_values, signal_values)
            
            # 添加回归线
            if len(noise_values) > 3:
                z = np.polyfit(noise_values, signal_values, 1)
                p = np.poly1d(z)
                x_range = np.linspace(noise_values.min(), noise_values.max(), 100)
                ax.plot(x_range, p(x_range), color='red', linestyle='--', 
                       linewidth=2, alpha=0.8)
            
            # 添加对角线
            all_values = np.concatenate([noise_values, signal_values])
            min_val, max_val = np.min(all_values), np.max(all_values)
            ax.plot([min_val, max_val], [min_val, max_val], 
                   color=ncfg.COLORS['neutral'], linestyle=':', 
                   linewidth=1.5, alpha=0.6, label='Unity Line')
            
            # 显示统计信息
            ax.text(0.05, 0.95, f'r = {corr_coef:.3f}\np = {p_val:.3e}\nn = {len(noise_values)}',
                   transform=ax.transAxes, fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                   verticalalignment='top')
                   
        except Exception as e:
            print(f"    统计计算失败: {e}")
        
        # 设置坐标轴
        ax.set_xlabel('Noise Correlation', fontsize=12)
        ax.set_ylabel('Signal Correlation', fontsize=12)
        ax.set_title(f'Condition {condition}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 设置坐标轴范围为对称
        all_values = np.concatenate([noise_values, signal_values])
        if len(all_values) > 0:
            max_abs = np.max(np.abs(all_values))
            ax.set_xlim(-max_abs*1.1, max_abs*1.1)
            ax.set_ylim(-max_abs*1.1, max_abs*1.1)
    
    # 隐藏多余的子图
    for idx in range(n_conditions, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(title, y=0.98, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=ncfg.DPI, bbox_inches='tight')
        print(f"神经元对散点图已保存: {save_path}")
    
    plt.show()

def visualize_shuffle_effects(shuffle_results, title="Neuron Shuffling Effects on Fisher Information", save_path=None):
    """专业可视化置换对Fisher信息的影响"""
    setup_noise_plot_style()
    
    fractions = shuffle_results['shuffle_fractions']
    original_fisher = shuffle_results['original_fisher_mean']
    
    # 提取Fisher信息均值和标准差
    fisher_means = []
    fisher_stds = []
    degradation_percents = []
    
    for fraction in fractions:
        result = shuffle_results['shuffle_results'][fraction]
        fisher_means.append(result['fisher_value'])
        fisher_stds.append(result['fisher_std'])
        degradation_percents.append(result['degradation_percent'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=ncfg.FIGURE_SIZE_LARGE)
    
    # Fisher信息变化图
    ax1.errorbar(fractions, fisher_means, yerr=fisher_stds, 
                marker='o', capsize=5, linewidth=3, markersize=8,
                color=ncfg.COLORS['fisher'], alpha=0.8,
                markerfacecolor='white', markeredgewidth=2)
    
    # 添加原始值参考线
    ax1.axhline(y=original_fisher, color=ncfg.COLORS['accent'], 
               linestyle='--', linewidth=2, alpha=0.8,
               label=f'Original: {original_fisher:.3f}')
    
    # 添加填充区域
    ax1.fill_between(fractions, 
                     np.array(fisher_means) - np.array(fisher_stds),
                     np.array(fisher_means) + np.array(fisher_stds),
                     alpha=0.2, color=ncfg.COLORS['fisher'])
    
    ax1.set_xlabel('Shuffling Fraction')
    ax1.set_ylabel('Fisher Information')
    ax1.set_title('Fisher Information Degradation')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 降解百分比图
    bars = ax2.bar(range(len(fractions)), degradation_percents,
                   alpha=0.8, color=ncfg.COLORS['secondary'],
                   edgecolor='black', linewidth=1.2)
    
    # 添加数值标签
    for bar, percent in zip(bars, degradation_percents):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                f'{percent:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_xlabel('Shuffling Fraction')
    ax2.set_ylabel('Information Degradation (%)')
    ax2.set_title('Information Loss Percentage')
    ax2.set_xticks(range(len(fractions)))
    ax2.set_xticklabels([f'{f:.1f}' for f in fractions])
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, y=0.98)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=ncfg.DPI, bbox_inches='tight')
    plt.show()

def visualize_network_metrics_comparison(noise_metrics, signal_metrics, title="Network Metrics Comparison", save_path=None):
    """专业可视化噪音和信号网络的指标比较"""
    setup_noise_plot_style()
    
    metrics_to_plot = ['density', 'avg_clustering', 'avg_degree']
    conditions = list(noise_metrics.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=ncfg.FIGURE_SIZE_EXTRA_LARGE)
    axes = axes.flatten()
    
    # 配色方案
    colors = [ncfg.COLORS['noise'], ncfg.COLORS['signal']]
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        noise_vals = [noise_metrics[cond].get(metric, 0) for cond in conditions]
        signal_vals = [signal_metrics[cond].get(metric, 0) for cond in conditions]
        
        x = np.arange(len(conditions))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, noise_vals, width, 
                      label='Noise Network', alpha=0.8,
                      color=colors[0], edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x + width/2, signal_vals, width, 
                      label='Signal Network', alpha=0.8,
                      color=colors[1], edgecolor='black', linewidth=1.2)
        
        # 添加数值标签
        max_val = max(max(noise_vals), max(signal_vals))
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + max_val * 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + max_val * 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Condition')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max_val * 1.15)
    
    # 雷达图比较
    ax_radar = plt.subplot(2, 2, 4, projection='polar')
    
    # 标准化指标用于雷达图
    all_noise_vals = []
    all_signal_vals = []
    for metric in metrics_to_plot:
        noise_vals = [noise_metrics[cond].get(metric, 0) for cond in conditions]
        signal_vals = [signal_metrics[cond].get(metric, 0) for cond in conditions]
        all_noise_vals.append(np.mean(noise_vals))
        all_signal_vals.append(np.mean(signal_vals))
    
    # 创建雷达图
    angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    # 标准化数据
    all_vals = all_noise_vals + all_signal_vals
    if max(all_vals) > 0:
        noise_norm = [v/max(all_vals) for v in all_noise_vals]
        signal_norm = [v/max(all_vals) for v in all_signal_vals]
    else:
        noise_norm = [0] * len(all_noise_vals)
        signal_norm = [0] * len(all_signal_vals)
    
    noise_norm += noise_norm[:1]
    signal_norm += signal_norm[:1]
    
    ax_radar.plot(angles, noise_norm, 'o-', linewidth=2.5,
                 label='Noise Network', color=colors[0], alpha=0.8)
    ax_radar.fill(angles, noise_norm, alpha=0.2, color=colors[0])
    
    ax_radar.plot(angles, signal_norm, 's-', linewidth=2.5,
                 label='Signal Network', color=colors[1], alpha=0.8)
    ax_radar.fill(angles, signal_norm, alpha=0.2, color=colors[1])
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels([m.replace('_', '\n').title() for m in metrics_to_plot])
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title('Normalized Metrics\nRadar Chart', y=1.08)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.suptitle(title, y=0.98)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=ncfg.DPI, bbox_inches='tight')
    plt.show()

def visualize_correlation_matrices(noise_correlations, signal_correlations, title="Correlation Matrices", save_path=None):
    """专业可视化噪音和信号相关性矩阵"""
    setup_noise_plot_style()
    
    conditions = list(noise_correlations.keys())
    n_conditions = len(conditions)
    
    fig, axes = plt.subplots(2, n_conditions, figsize=(5*n_conditions, 10))
    if n_conditions == 1:
        axes = axes.reshape(2, 1)
    
    for i, condition in enumerate(conditions):
        # 噪音相关性矩阵
        ax1 = axes[0, i]
        im1 = ax1.imshow(noise_correlations[condition], 
                        cmap='RdBu_r', vmin=-1, vmax=1, 
                        aspect='auto', interpolation='nearest')
        ax1.set_title(f'Noise Correlation\nCondition {condition}')
        ax1.set_xlabel('Neuron Index')
        ax1.set_ylabel('Neuron Index')
        
        # 添加颜色条
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Correlation', rotation=270, labelpad=15)
        
        # 信号相关性矩阵
        ax2 = axes[1, i]
        im2 = ax2.imshow(signal_correlations[condition], 
                        cmap='RdBu_r', vmin=-1, vmax=1,
                        aspect='auto', interpolation='nearest')
        ax2.set_title(f'Signal Correlation\nCondition {condition}')
        ax2.set_xlabel('Neuron Index')
        ax2.set_ylabel('Neuron Index')
        
        # 添加颜色条
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('Correlation', rotation=270, labelpad=15)
    
    plt.suptitle(title, y=0.98)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=ncfg.DPI, bbox_inches='tight')
    plt.show()

def visualize_hub_peripheral_analysis(hub_analysis, title="Hub-Peripheral Neuron Analysis", save_path=None):
    """专业可视化枢纽-边缘神经元分析"""
    setup_noise_plot_style()
    
    conditions = list(hub_analysis.keys())
    n_conditions = len(conditions)
    
    fig, axes = plt.subplots(2, 2, figsize=ncfg.FIGURE_SIZE_EXTRA_LARGE)
    
    # 1. 枢纽和边缘神经元数量比较
    ax1 = axes[0, 0]
    hub_counts = [hub_analysis[cond]['n_hubs'] for cond in conditions]
    peripheral_counts = [hub_analysis[cond]['n_peripheral'] for cond in conditions]
    
    x = np.arange(len(conditions))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, hub_counts, width, 
                   label='Hub Neurons', alpha=0.8,
                   color=ncfg.COLORS['hub'], edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x + width/2, peripheral_counts, width,
                   label='Peripheral Neurons', alpha=0.8,
                   color=ncfg.COLORS['peripheral'], edgecolor='black', linewidth=1.2)
    
    ax1.set_xlabel('Condition')
    ax1.set_ylabel('Number of Neurons')
    ax1.set_title('Hub vs Peripheral Neuron Count')
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 内部相关性强度比较
    ax2 = axes[0, 1]
    hub_hub_means = [hub_analysis[cond]['hub_hub_correlation']['mean'] for cond in conditions]
    per_per_means = [hub_analysis[cond]['peripheral_peripheral_correlation']['mean'] for cond in conditions]
    hub_per_means = [hub_analysis[cond]['hub_peripheral_correlation']['mean'] for cond in conditions]
    
    hub_hub_stds = [hub_analysis[cond]['hub_hub_correlation']['std'] for cond in conditions]
    per_per_stds = [hub_analysis[cond]['peripheral_peripheral_correlation']['std'] for cond in conditions]
    hub_per_stds = [hub_analysis[cond]['hub_peripheral_correlation']['std'] for cond in conditions]
    
    width = 0.25
    x = np.arange(len(conditions))
    
    bars1 = ax2.bar(x - width, hub_hub_means, width, yerr=hub_hub_stds,
                   label='Hub-Hub', alpha=0.8, color=ncfg.COLORS['hub'],
                   edgecolor='black', linewidth=1.2, capsize=5)
    bars2 = ax2.bar(x, per_per_means, width, yerr=per_per_stds,
                   label='Peripheral-Peripheral', alpha=0.8, color=ncfg.COLORS['peripheral'],
                   edgecolor='black', linewidth=1.2, capsize=5)
    bars3 = ax2.bar(x + width, hub_per_means, width, yerr=hub_per_stds,
                   label='Hub-Peripheral', alpha=0.8, color=ncfg.COLORS['neutral'],
                   edgecolor='black', linewidth=1.2, capsize=5)
    
    ax2.set_xlabel('Condition')
    ax2.set_ylabel('Noise Correlation Strength')
    ax2.set_title('Internal vs Cross-Type Correlations')
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 统计显著性测试结果
    ax3 = axes[1, 0]
    p_values = []
    significance_labels = []
    
    for cond in conditions:
        p_val = hub_analysis[cond]['statistics']['hub_vs_peripheral_internal']['p_value']
        p_values.append(p_val if not np.isnan(p_val) else 1.0)
        
        if np.isnan(p_val):
            significance_labels.append('N/A')
        elif p_val < 0.001:
            significance_labels.append('***')
        elif p_val < 0.01:
            significance_labels.append('**')
        elif p_val < 0.05:
            significance_labels.append('*')
        else:
            significance_labels.append('ns')
    
    colors = [ncfg.COLORS['accent'] if p < 0.05 else ncfg.COLORS['neutral'] 
              for p in p_values]
    
    bars = ax3.bar(range(len(conditions)), [-np.log10(p + 1e-10) for p in p_values],
                  alpha=0.8, color=colors, edgecolor='black', linewidth=1.2)
    
    # 添加显著性标记
    for i, (bar, label) in enumerate(zip(bars, significance_labels)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                label, ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax3.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7,
               label='p = 0.05')
    ax3.set_xlabel('Condition')
    ax3.set_ylabel('-log₁₀(p-value)')
    ax3.set_title('Statistical Significance\n(Hub vs Peripheral Internal)')
    ax3.set_xticks(range(len(conditions)))
    ax3.set_xticklabels(conditions)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 相关性分布箱图
    ax4 = axes[1, 1]
    
    # 准备数据用于箱图
    all_data = []
    all_labels = []
    all_colors = []
    
    for cond in conditions:
        # Hub-Hub相关性
        n_hub_pairs = hub_analysis[cond]['hub_hub_correlation']['n_pairs']
        if n_hub_pairs > 0:
            hub_hub_mean = hub_analysis[cond]['hub_hub_correlation']['mean']
            hub_hub_std = hub_analysis[cond]['hub_hub_correlation']['std']
            # 模拟分布用于箱图
            hub_hub_data = np.random.normal(hub_hub_mean, hub_hub_std, min(100, n_hub_pairs))
            all_data.append(hub_hub_data)
            all_labels.append(f'{cond}\nHub-Hub')
            all_colors.append(ncfg.COLORS['hub'])
        
        # Peripheral-Peripheral相关性
        n_per_pairs = hub_analysis[cond]['peripheral_peripheral_correlation']['n_pairs']
        if n_per_pairs > 0:
            per_per_mean = hub_analysis[cond]['peripheral_peripheral_correlation']['mean']
            per_per_std = hub_analysis[cond]['peripheral_peripheral_correlation']['std']
            per_per_data = np.random.normal(per_per_mean, per_per_std, min(100, n_per_pairs))
            all_data.append(per_per_data)
            all_labels.append(f'{cond}\nPer-Per')
            all_colors.append(ncfg.COLORS['peripheral'])
    
    if all_data:
        bp = ax4.boxplot(all_data, labels=all_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], all_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)
    
    ax4.set_ylabel('Noise Correlation Strength')
    ax4.set_title('Correlation Distribution')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, y=0.98)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=ncfg.DPI, bbox_inches='tight')
    plt.show()

# %% 结果保存函数

def save_comparison_results(comparison_results, filename):
    """保存比较结果为结构化数据"""
    ncfg.ensure_results_dir()
    
    # 保存噪音-信号相关性比较结果
    comparison_data = {}
    for condition, results in comparison_results.items():
        comparison_data[f'{condition}_correlation'] = results['correlation']
        comparison_data[f'{condition}_p_value'] = results['p_value']
        # 保存统计信息，但不保存原始数据向量（太大）
        comparison_data[f'{condition}_noise_mean'] = np.mean(results['noise_values'])
        comparison_data[f'{condition}_noise_std'] = np.std(results['noise_values'])
        comparison_data[f'{condition}_signal_mean'] = np.mean(results['signal_values'])
        comparison_data[f'{condition}_signal_std'] = np.std(results['signal_values'])
    
    np.savez_compressed(
        os.path.join(ncfg.RESULTS_DIR, filename),
        **comparison_data
    )
    print(f"比较结果已保存到 {filename}")


# %% 主函数和分析流程

def run_noise_correlation_analysis():
    """运行完整的噪音相关性分析流程"""
    print("=" * 60)
    print("噪音相关性与神经元置换分析")
    print("=" * 60)
    
    # 确保结果目录存在
    ncfg.ensure_results_dir()
    
    # 1. 加载和预处理数据
    print("\n1. 数据加载与预处理")
    print("-" * 30)
    
    # 加载数据
    if cfg.LOADER_VERSION == 'new':
        neural_data_raw, neuron_pos, start_edges, stimulus_data = load_data(cfg.DATA_PATH)
        segments, labels = segment_neuron_data(neural_data_raw, start_edges, stimulus_data)
        neural_data = np.array(segments)  # (n_trials, n_neurons, n_timepoints)
        labels = np.array(labels)
        # 重分类标签（合并噪音条件）
        labels = reclassify_labels(stimulus_data)
    elif cfg.LOADER_VERSION == 'old':
        from loaddata import load_old_version_data
        neuron_index, neural_data, labels, neuron_pos = load_old_version_data(
            cfg.OLD_VERSION_PATHS['neurons'],
            cfg.OLD_VERSION_PATHS['trials'],
            cfg.OLD_VERSION_PATHS['location']
        )
        # 对于旧版数据，neural_data和labels已经是处理好的格式
        # neural_data 已经是 (trials, neurons, timepoints) 格式
        neuron_pos = neuron_pos[0:2, :] if neuron_pos.shape[0] >= 2 else neuron_pos
        print(f"旧版数据维度: neural_data={neural_data.shape}, labels={len(labels)}, neuron_pos={neuron_pos.shape}")
        print("已切换到旧版数据加载模式")
    else:
        raise ValueError("无效的 LOADER_VERSION 配置")
    
    # 过滤掉标签为0的数据
    valid_mask = labels != 0
    neural_data = neural_data[valid_mask]
    labels = labels[valid_mask]
    
    # RR神经元选择
    rr_results = fast_rr_selection(neural_data, labels)
    rr_indices = rr_results['rr_neurons']
    neural_data_rr = neural_data[:, rr_indices, :]
    
    print(f"数据维度: {neural_data_rr.shape}")
    print(f"标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
    print(f"RR神经元数量: {len(rr_indices)}")
    
    # 2. 计算噪音相关性和信号相关性
    print("\n2. 相关性分析")
    print("-" * 30)
    
    noise_correlations = calculate_noise_correlation(neural_data_rr, labels)
    signal_correlations = calculate_signal_correlation(neural_data_rr, labels)
    
    # 3. 保存相关性矩阵
    print("\n3. 保存相关性矩阵")
    print("-" * 30)
    
    ncfg.ensure_results_dir()
    
    # 保存噪音相关性矩阵
    np.savez_compressed(
        os.path.join(ncfg.RESULTS_DIR, 'noise_correlation_matrices.npz'),
        **{f'condition_{k}': v for k, v in noise_correlations.items()}
    )
    print("噪音相关性矩阵已保存")
    
    # 保存信号相关性矩阵
    np.savez_compressed(
        os.path.join(ncfg.RESULTS_DIR, 'signal_correlation_matrices.npz'),
        **{f'condition_{k}': v for k, v in signal_correlations.items()}
    )
    print("信号相关性矩阵已保存")
    
    # 4. 比较噪音和信号相关性
    print("\n4. 噪音-信号相关性比较")
    print("-" * 30)
    
    comparison_results = compare_noise_signal_correlations(noise_correlations, signal_correlations)
    save_comparison_results(comparison_results, 'noise_signal_comparison.npz')
    
    # 5. 神经元置换效果分析（使用Fisher信息）
    print("\n5. 神经元置换效果分析（Fisher信息）")
    print("-" * 30)
    
    # 对于已经筛选的RR数据，传入连续的索引
    rr_indices_sequential = list(range(len(rr_indices)))
    shuffle_results = analyze_shuffle_effect_with_fisher(neural_data_rr, labels, rr_indices_sequential)
    
    # 保存置换结果
    np.savez_compressed(
        os.path.join(ncfg.RESULTS_DIR, 'shuffle_fisher_results.npz'),
        original_fisher_mean=shuffle_results['original_fisher_mean'],
        shuffle_fractions=shuffle_results['shuffle_fractions'],
        fisher_means=[shuffle_results['shuffle_results'][f]['fisher_value'] for f in shuffle_results['shuffle_fractions']],
        fisher_stds=[shuffle_results['shuffle_results'][f]['fisher_std'] for f in shuffle_results['shuffle_fractions']],
        degradation_percents=[shuffle_results['shuffle_results'][f]['degradation_percent'] for f in shuffle_results['shuffle_fractions']],
        n_iterations=shuffle_results['n_iterations']
    )
    print("置换Fisher信息结果已保存")
    
    # 6. 网络拓扑分析
    print("\n6. 网络拓扑分析")
    print("-" * 30)
    
    # 构建网络并计算指标
    noise_network_metrics = build_networks_from_correlations(
        noise_correlations, method=ncfg.NETWORK_METHOD, 
        threshold=ncfg.NETWORK_THRESHOLD, density=ncfg.NETWORK_DENSITY
    )
    signal_network_metrics = build_networks_from_correlations(
        signal_correlations, method=ncfg.NETWORK_METHOD,
        threshold=ncfg.NETWORK_THRESHOLD, density=ncfg.NETWORK_DENSITY
    )
    
    # 比较网络指标
    network_comparison = compare_network_metrics(noise_network_metrics, signal_network_metrics)
    
    # 保存网络指标比较结果
    network_comparison_results = {}
    for condition, metrics in network_comparison.items():
        network_comparison_results[condition] = {
            metric: {
                'noise': values['noise'],
                'signal': values['signal'], 
                'difference': values['difference'],
                'relative_diff_percent': values['relative_diff_percent']
            }
            for metric, values in metrics.items()
        }
    
    np.savez_compressed(
        os.path.join(ncfg.RESULTS_DIR, 'network_metrics_comparison.npz'),
        **{f'{condition}_{metric}_{measure}': value 
           for condition, condition_data in network_comparison_results.items()
           for metric, metric_data in condition_data.items()
           for measure, value in metric_data.items()}
    )
    print("网络指标比较结果已保存")
    
    # 7. 枢纽-边缘神经元分析
    print("\n7. 枢纽-边缘神经元分析")
    print("-" * 30)
    
    hub_analysis = analyze_hub_peripheral_noise_correlation(noise_correlations, noise_network_metrics)
    
    # 保存枢纽-边缘分析结果
    hub_analysis_results = {}
    for condition, analysis in hub_analysis.items():
        hub_analysis_results[f'{condition}_n_hubs'] = analysis['n_hubs']
        hub_analysis_results[f'{condition}_n_peripheral'] = analysis['n_peripheral']
        hub_analysis_results[f'{condition}_hub_hub_mean'] = analysis['hub_hub_correlation']['mean']
        hub_analysis_results[f'{condition}_hub_hub_std'] = analysis['hub_hub_correlation']['std']
        hub_analysis_results[f'{condition}_peripheral_peripheral_mean'] = analysis['peripheral_peripheral_correlation']['mean']
        hub_analysis_results[f'{condition}_peripheral_peripheral_std'] = analysis['peripheral_peripheral_correlation']['std']
        hub_analysis_results[f'{condition}_hub_peripheral_mean'] = analysis['hub_peripheral_correlation']['mean']
        hub_analysis_results[f'{condition}_hub_peripheral_std'] = analysis['hub_peripheral_correlation']['std']
        hub_analysis_results[f'{condition}_hub_vs_peripheral_p'] = analysis['statistics']['hub_vs_peripheral_internal']['p_value']
    
    np.savez_compressed(
        os.path.join(ncfg.RESULTS_DIR, 'hub_peripheral_analysis.npz'),
        **hub_analysis_results
    )
    print("枢纽-边缘神经元分析结果已保存")
    
    # 8. 专业可视化分析结果
    print("\n8. 专业可视化分析结果")
    print("-" * 30)
    
    # 可视化相关性矩阵
    print("生成相关性矩阵可视化...")
    visualize_correlation_matrices(
        noise_correlations, signal_correlations,
        title="Noise vs Signal Correlation Matrices",
        save_path=os.path.join(ncfg.RESULTS_DIR, 'correlation_matrices.png')
    )
    
    # 可视化噪音-信号相关性比较
    print("生成噪音-信号相关性比较图...")
    visualize_noise_signal_comparison(
        comparison_results,
        title="Noise-Signal Correlation Comparison",
        save_path=os.path.join(ncfg.RESULTS_DIR, 'noise_signal_comparison.png')
    )
    
    # 可视化神经元对的噪音-信号相关性散点图
    print("生成神经元对噪音-信号相关性散点图...")
    visualize_neuron_pairs_scatter(
        noise_correlations, signal_correlations,
        title="Neuron Pairs: Noise vs Signal Correlations",
        save_path=os.path.join(ncfg.RESULTS_DIR, 'neuron_pairs_scatter.png')
    )
    
    # 可视化置换效果
    print("生成神经元置换效果图...")
    visualize_shuffle_effects(
        shuffle_results,
        title="Neuron Shuffling Effects on Fisher Information", 
        save_path=os.path.join(ncfg.RESULTS_DIR, 'shuffle_effects.png')
    )
    
    # 可视化网络指标比较
    print("生成网络指标比较图...")
    visualize_network_metrics_comparison(
        noise_network_metrics, signal_network_metrics,
        title="Network Metrics: Noise vs Signal Networks",
        save_path=os.path.join(ncfg.RESULTS_DIR, 'network_metrics_comparison.png')
    )
    
    # 可视化枢纽-边缘神经元分析
    print("生成枢纽-边缘神经元分析图...")
    visualize_hub_peripheral_analysis(
        hub_analysis,
        title="Hub-Peripheral Neuron Noise Correlation Analysis",
        save_path=os.path.join(ncfg.RESULTS_DIR, 'hub_peripheral_analysis.png')
    )
    
    # 9. 生成分析报告
    print("\n9. 生成分析报告")
    print("-" * 30)
    
    report = generate_analysis_report(
        noise_correlations, signal_correlations, 
        comparison_results, shuffle_results, network_comparison, hub_analysis
    )
    
    # 保存报告
    report_path = os.path.join(ncfg.RESULTS_DIR, 'analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"分析完成！结果保存在: {ncfg.RESULTS_DIR}")
    return {
        'noise_correlations': noise_correlations,
        'signal_correlations': signal_correlations,
        'comparison_results': comparison_results,
        'shuffle_results': shuffle_results,
        'noise_network_metrics': noise_network_metrics,
        'signal_network_metrics': signal_network_metrics,
        'network_comparison': network_comparison,
        'hub_analysis': hub_analysis,
        'report': report
    }


def generate_analysis_report(noise_correlations, signal_correlations, comparison_results, shuffle_results, network_comparison, hub_analysis):
    """生成分析报告"""
    report = []
    report.append("=" * 60)
    report.append("噪音相关性与神经元置换分析报告")
    report.append("=" * 60)
    report.append("")
    
    # 基本信息
    report.append("## 基本信息")
    report.append(f"- 分析条件数: {len(noise_correlations)}")
    report.append(f"- 噪音相关性矩阵维度: {list(noise_correlations.values())[0].shape}")
    report.append("")
    
    # 噪音-信号相关性分析结果
    report.append("## 噪音-信号相关性分析")
    for condition, results in comparison_results.items():
        r_val = results['correlation']
        p_val = results['p_value']
        significance = "显著" if p_val < 0.05 else "不显著"
        report.append(f"- 条件 {condition}: r = {r_val:.4f} (p = {p_val:.4f}, {significance})")
    report.append("")
    
    # 置换效果分析（Fisher信息）
    report.append("## 神经元置换效果（Fisher信息）")
    original_fi = shuffle_results['original_fisher_mean']
    report.append(f"- 原始Fisher信息: {original_fi:.4f}")
    report.append("- 不同置换比例的影响:")
    
    for fraction in shuffle_results['shuffle_fractions']:
        result = shuffle_results['shuffle_results'][fraction]
        fisher_val = result['fisher_value']
        degradation = result['degradation_percent']
        report.append(f"  * {fraction:.1f}: {fisher_val:.4f} (Fisher信息下降 {degradation:.1f}%)")
    report.append("")
    
    # 网络拓扑比较
    report.append("## 网络拓扑比较")
    for condition in network_comparison.keys():
        report.append(f"- 条件 {condition}:")
        comp = network_comparison[condition]
        for metric, values in comp.items():
            report.append(f"  * {metric}: 噪音={values['noise']:.4f}, 信号={values['signal']:.4f}, "
                         f"相对差异={values['relative_diff_percent']:.1f}%")
    report.append("")
    
    # 关键发现
    report.append("## 关键发现")
    
    # 找到Fisher信息下降最快的置换比例点
    max_degradation_fraction = None
    max_degradation_rate = 0
    
    fractions = shuffle_results['shuffle_fractions']
    for i, fraction in enumerate(fractions):
        if i == 0:
            continue
        current_fi = shuffle_results['shuffle_results'][fraction]['fisher_value']
        prev_fi = shuffle_results['shuffle_results'][fractions[i-1]]['fisher_value']
        degradation_rate = (prev_fi - current_fi) / (fraction - fractions[i-1])
        
        if degradation_rate > max_degradation_rate:
            max_degradation_rate = degradation_rate
            max_degradation_fraction = fraction
    
    if max_degradation_fraction is not None:
        report.append(f"- Fisher信息下降最快的置换点: {max_degradation_fraction:.1f}")
    
    # 噪音-信号相关性的总体模式
    avg_correlation = np.mean([r['correlation'] for r in comparison_results.values()])
    if avg_correlation > 0.1:
        correlation_pattern = "正相关"
    elif avg_correlation < -0.1:
        correlation_pattern = "负相关"
    else:
        correlation_pattern = "弱相关"
    
    report.append(f"- 噪音与信号相关性总体呈现: {correlation_pattern} (平均 r = {avg_correlation:.4f})")
    
    # 网络拓扑差异总结
    density_diffs = [comp['density']['relative_diff_percent'] for comp in network_comparison.values()]
    avg_density_diff = np.mean(density_diffs)
    clustering_diffs = [comp['avg_clustering']['relative_diff_percent'] for comp in network_comparison.values()]
    avg_clustering_diff = np.mean(clustering_diffs)
    
    report.append(f"- 网络密度差异: 噪音网络相对信号网络平均{avg_density_diff:+.1f}%")
    report.append(f"- 聚类系数差异: 噪音网络相对信号网络平均{avg_clustering_diff:+.1f}%")
    
    # 枢纽-边缘分析
    report.append("")
    report.append("## 枢纽-边缘神经元分析")
    for condition in hub_analysis.keys():
        hub_result = hub_analysis[condition]
        report.append(f"- 条件 {condition}:")
        report.append(f"  * 枢纽神经元数: {hub_result['n_hubs']} (度 >= {hub_result['degree_threshold_hub']:.1f})")
        report.append(f"  * 边缘神经元数: {hub_result['n_peripheral']} (度 <= {hub_result['degree_threshold_peripheral']:.1f})")
        report.append(f"  * 枢纽-枢纽噪音相关性: {hub_result['hub_hub_correlation']['mean']:.4f} ± {hub_result['hub_hub_correlation']['std']:.4f}")
        report.append(f"  * 边缘-边缘噪音相关性: {hub_result['peripheral_peripheral_correlation']['mean']:.4f} ± {hub_result['peripheral_peripheral_correlation']['std']:.4f}")
        report.append(f"  * 枢纽-边缘噪音相关性: {hub_result['hub_peripheral_correlation']['mean']:.4f} ± {hub_result['hub_peripheral_correlation']['std']:.4f}")
        
        hub_vs_per_p = hub_result['statistics']['hub_vs_peripheral_internal']['p_value']
        if not np.isnan(hub_vs_per_p):
            significance = "显著" if hub_vs_per_p < 0.05 else "不显著"
            report.append(f"  * 枢纽内部vs边缘内部差异: p={hub_vs_per_p:.4f} ({significance})")
    report.append("")
    
    # 最终发现汇总
    if hub_analysis:
        all_hub_hub = [h['hub_hub_correlation']['mean'] for h in hub_analysis.values()]
        all_per_per = [h['peripheral_peripheral_correlation']['mean'] for h in hub_analysis.values()]
        avg_hub_correlation = np.mean(all_hub_hub)
        avg_per_correlation = np.mean(all_per_per)
        
        report.append(f"- 枢纽神经元内部噪音相关性: 平均 {avg_hub_correlation:.4f}")
        report.append(f"- 边缘神经元内部噪音相关性: 平均 {avg_per_correlation:.4f}")
        
        if avg_hub_correlation > avg_per_correlation:
            correlation_pattern = f"枢纽神经元内部噪音相关性更强 (+{(avg_hub_correlation-avg_per_correlation)/avg_per_correlation*100:.1f}%)"
        else:
            correlation_pattern = f"边缘神经元内部噪音相关性更强 (+{(avg_per_correlation-avg_hub_correlation)/avg_hub_correlation*100:.1f}%)"
        
        report.append(f"- 噪音相关性模式: {correlation_pattern}")
    
    return "\n".join(report)


# %% 主程序执行
if __name__ == "__main__":
    # 设置matplotlib中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    
    # 运行分析
    results = run_noise_correlation_analysis()
    
    print("\n专业科研风格噪音相关性分析完成！")
    print("主要结果文件:")
    print(f"- 噪音vs信号相关性矩阵: {ncfg.RESULTS_DIR}/correlation_matrices.png")
    print(f"- 噪音-信号相关性比较: {ncfg.RESULTS_DIR}/noise_signal_comparison.png")
    print(f"- 神经元置换Fisher信息效果: {ncfg.RESULTS_DIR}/shuffle_effects.png")
    print(f"- 网络指标专业对比: {ncfg.RESULTS_DIR}/network_metrics_comparison.png")
    print(f"- 枢纽-边缘神经元分析: {ncfg.RESULTS_DIR}/hub_peripheral_analysis.png")
    print(f"- 详细分析报告: {ncfg.RESULTS_DIR}/analysis_report.txt")
    print(f"- 数据保存目录: {ncfg.RESULTS_DIR}/")
    print("\n所有图像均采用高分辨率科研绘图标准生成，适合论文发表使用。")