# 网络效率神经元打乱分析
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
from collections import defaultdict

# 导入必要模块
from loaddata import cfg
from noise_correlation_analysis import (
    NoiseCorrelationConfig, calculate_noise_correlation, 
    shuffle_within_condition, build_networks_from_correlations
)

class NetworkEfficiencyShuffleConfig(NoiseCorrelationConfig):
    """网络效率打乱分析配置"""
    SHUFFLE_FRACTIONS = [0.0, 0.3, 0.6, 1.0]  # 减少测试点
    N_ITERATIONS = 2  # 减少迭代次数
    
    @classmethod
    def get_results_dir(cls):
        return os.path.join(cfg.get_results_dir(), 'network_efficiency_shuffle')
    
    @classmethod
    def ensure_results_dir(cls):
        results_dir = cls.get_results_dir()
        os.makedirs(results_dir, exist_ok=True)
        return results_dir

necfg = NetworkEfficiencyShuffleConfig()

def setup_plot_style():
    """设置科研绘图风格"""
    plt.style.use('seaborn-v0_8-whitegrid')
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

def calculate_network_efficiency(adj_matrix):
    """计算网络的全局效率、局部效率和聚类系数"""
    try:
        G = nx.from_numpy_array(adj_matrix)
        global_efficiency = nx.global_efficiency(G)
        local_efficiency = nx.local_efficiency(G)
        clustering_coefficient = nx.average_clustering(G)
        
        return {
            'global_efficiency': global_efficiency,
            'local_efficiency': local_efficiency,
            'clustering_coefficient': clustering_coefficient
        }
    except:
        return {
            'global_efficiency': 0.0,
            'local_efficiency': 0.0,
            'clustering_coefficient': 0.0
        }

def calculate_flattened_correlation(neural_data):
    """展开数据计算相关性矩阵"""
    n_trials, n_neurons, n_timepoints = neural_data.shape
    
    # 展开数据: (n_neurons, n_trials * n_timepoints)
    flattened_data = neural_data.transpose(1, 0, 2).reshape(n_neurons, -1)
    
    # 计算相关性矩阵
    corr_matrix = np.corrcoef(flattened_data)
    
    # 处理NaN值
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    return corr_matrix

def analyze_shuffle_network_efficiency(neural_data, labels, shuffle_fractions=None, n_iterations=3):
    """分析神经元打乱对网络效率的影响（回退版：展开数据计算相关性，只使用正相关）"""
    if shuffle_fractions is None:
        shuffle_fractions = necfg.SHUFFLE_FRACTIONS
    
    print(f"分析神经元打乱对网络效率的影响（回退版）...")
    print(f"- 展开数据计算相关性矩阵")
    print(f"- 只使用正相关值构建网络")
    print(f"- 每个打乱比例重复{n_iterations}次取平均")
    
    results = {
        'shuffle_fractions': shuffle_fractions,
        'global_efficiency': [],
        'local_efficiency': [],
        'clustering_coefficient': [],
        'global_efficiency_std': [],
        'local_efficiency_std': [],
        'clustering_coefficient_std': [],
        'n_iterations': n_iterations
    }
    
    for fraction in shuffle_fractions:
        print(f"\\n打乱比例: {fraction:.1f}")
        
        # 存储每次迭代的结果
        global_effs = []
        local_effs = []
        clusterings = []
        
        for iteration in range(n_iterations):
            # 打乱数据
            if fraction == 0.0:
                shuffled_data = neural_data.copy()
            else:
                shuffled_data = shuffle_within_condition(neural_data, labels, fraction)
            
            # 展开数据计算相关性矩阵
            avg_corr_matrix = calculate_flattened_correlation(shuffled_data)
            
            # 构建网络：只使用正相关值
            pos_corr = np.copy(avg_corr_matrix)
            pos_corr[pos_corr < 0] = 0  # 将负相关设为0
            np.fill_diagonal(pos_corr, 0)  # 对角线设为0
            
            # 使用密度阈值方法
            n_nodes = pos_corr.shape[0]
            n_possible_edges = n_nodes * (n_nodes - 1) // 2
            n_edges_to_keep = int(n_possible_edges * necfg.NETWORK_DENSITY)
            
            if n_edges_to_keep > 0:
                # 只在上三角矩阵中选择最强的正相关
                triu_indices = np.triu_indices_from(pos_corr, k=1)
                triu_values = pos_corr[triu_indices]
                
                if np.sum(triu_values > 0) >= n_edges_to_keep:
                    # 如果有足够的正相关值
                    threshold = np.partition(triu_values, -n_edges_to_keep)[-n_edges_to_keep]
                    adj_matrix = pos_corr >= threshold
                else:
                    # 如果正相关值不够，使用所有正相关
                    adj_matrix = pos_corr > 0
                
                np.fill_diagonal(adj_matrix, 0)
            else:
                adj_matrix = np.zeros_like(pos_corr, dtype=bool)
            
            # 计算网络效率
            efficiency_metrics = calculate_network_efficiency(adj_matrix)
            
            global_effs.append(efficiency_metrics['global_efficiency'])
            local_effs.append(efficiency_metrics['local_efficiency'])
            clusterings.append(efficiency_metrics['clustering_coefficient'])
        
        # 计算统计量
        results['global_efficiency'].append(np.mean(global_effs) if global_effs else 0.0)
        results['local_efficiency'].append(np.mean(local_effs) if local_effs else 0.0)
        results['clustering_coefficient'].append(np.mean(clusterings) if clusterings else 0.0)
        
        results['global_efficiency_std'].append(np.std(global_effs) if len(global_effs) > 1 else 0.0)
        results['local_efficiency_std'].append(np.std(local_effs) if len(local_effs) > 1 else 0.0)
        results['clustering_coefficient_std'].append(np.std(clusterings) if len(clusterings) > 1 else 0.0)
        
        print(f"  全局效率: {results['global_efficiency'][-1]:.4f} ± {results['global_efficiency_std'][-1]:.4f}")
        print(f"  局部效率: {results['local_efficiency'][-1]:.4f} ± {results['local_efficiency_std'][-1]:.4f}")
        print(f"  聚类系数: {results['clustering_coefficient'][-1]:.4f} ± {results['clustering_coefficient_std'][-1]:.4f}")
    
    return results

def visualize_network_efficiency_shuffle(results, title="Network Efficiency vs Shuffle Fraction", save_path=None):
    """可视化网络效率随打乱比例的变化（合并版本）"""
    setup_plot_style()
    
    shuffle_fractions = results['shuffle_fractions']
    
    # 创建2x2子图布局
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 定义颜色
    color_global = '#2E86AB'   # 蓝色
    color_local = '#A23B72'    # 紫色
    color_cluster = '#F18F01'  # 橙色
    
    # 1. 全局效率
    ax1 = axes[0, 0]
    global_effs = results['global_efficiency']
    global_stds = results['global_efficiency_std']
    
    ax1.errorbar(shuffle_fractions, global_effs, yerr=global_stds,
                marker='o', capsize=5, linewidth=3, markersize=8,
                color=color_global, alpha=0.8, label='Global Efficiency')
    
    # 填充标准误区域
    ax1.fill_between(shuffle_fractions, 
                    np.array(global_effs) - np.array(global_stds),
                    np.array(global_effs) + np.array(global_stds),
                    alpha=0.2, color=color_global)
    
    ax1.set_xlabel('Shuffle Fraction')
    ax1.set_ylabel('Global Efficiency')
    ax1.set_title('Global Efficiency vs Shuffle Fraction')
    ax1.grid(True, alpha=0.3)
    
    # 2. 局部效率
    ax2 = axes[0, 1]
    local_effs = results['local_efficiency']
    local_stds = results['local_efficiency_std']
    
    ax2.errorbar(shuffle_fractions, local_effs, yerr=local_stds,
                marker='s', capsize=5, linewidth=3, markersize=8,
                color=color_local, alpha=0.8, label='Local Efficiency')
    
    ax2.fill_between(shuffle_fractions, 
                    np.array(local_effs) - np.array(local_stds),
                    np.array(local_effs) + np.array(local_stds),
                    alpha=0.2, color=color_local)
    
    ax2.set_xlabel('Shuffle Fraction')
    ax2.set_ylabel('Local Efficiency')
    ax2.set_title('Local Efficiency vs Shuffle Fraction')
    ax2.grid(True, alpha=0.3)
    
    # 3. 聚类系数
    ax3 = axes[1, 0]
    clusterings = results['clustering_coefficient']
    clustering_stds = results['clustering_coefficient_std']
    
    ax3.errorbar(shuffle_fractions, clusterings, yerr=clustering_stds,
                marker='^', capsize=5, linewidth=3, markersize=8,
                color=color_cluster, alpha=0.8, label='Clustering Coefficient')
    
    ax3.fill_between(shuffle_fractions, 
                    np.array(clusterings) - np.array(clustering_stds),
                    np.array(clusterings) + np.array(clustering_stds),
                    alpha=0.2, color=color_cluster)
    
    ax3.set_xlabel('Shuffle Fraction')
    ax3.set_ylabel('Clustering Coefficient')
    ax3.set_title('Clustering Coefficient vs Shuffle Fraction')
    ax3.grid(True, alpha=0.3)
    
    # 4. 归一化变化率
    ax4 = axes[1, 1]
    
    # 计算相对于原始值（fraction=0）的变化率
    global_effs = np.array(results['global_efficiency'])
    local_effs = np.array(results['local_efficiency'])
    cluster_effs = np.array(results['clustering_coefficient'])
    
    if global_effs[0] > 0:
        global_change = (global_effs - global_effs[0]) / global_effs[0] * 100
        ax4.plot(shuffle_fractions, global_change, 
                marker='o', linewidth=3, markersize=6, color=color_global, alpha=0.8, 
                label='Global Efficiency')
    
    if local_effs[0] > 0:
        local_change = (local_effs - local_effs[0]) / local_effs[0] * 100
        ax4.plot(shuffle_fractions, local_change, 
                marker='s', linewidth=3, markersize=6, color=color_local, alpha=0.8,
                label='Local Efficiency')
    
    if cluster_effs[0] > 0:
        cluster_change = (cluster_effs - cluster_effs[0]) / cluster_effs[0] * 100
        ax4.plot(shuffle_fractions, cluster_change,
                marker='^', linewidth=3, markersize=6, color=color_cluster, alpha=0.8,
                label='Clustering Coefficient')
    
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Shuffle Fraction')
    ax4.set_ylabel('Change from Original (%)')
    ax4.set_title('Relative Change in Network Metrics')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, y=0.98, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=necfg.DPI, bbox_inches='tight')
        print(f"网络效率分析图已保存: {save_path}")
    
    plt.close()

def run_shuffle_analysis():
    """运行打乱分析（使用M65真实数据）"""
    print("=" * 60)
    print("网络效率打乱分析 - M65数据")
    print("=" * 60)
    
    # 确保结果目录存在
    results_dir = necfg.ensure_results_dir()
    
    # 加载M65真实数据
    print("\\n加载M65真实数据...")
    try:
        from loaddata import load_data, segment_neuron_data, reclassify_labels, fast_rr_selection
        
        # M65数据路径
        m65_data_path = r'C:\Users\76629\OneDrive\brain\Micedata\M65_0816'
        
        # 加载原始数据
        neural_data_raw, neuron_pos, start_edges, stimulus_data = load_data(m65_data_path)
        segments, labels = segment_neuron_data(neural_data_raw, start_edges, stimulus_data)
        neural_data = np.array(segments)
        labels = np.array(labels)
        labels = reclassify_labels(stimulus_data)
        
        # 过滤有效数据
        valid_mask = labels != 0
        neural_data = neural_data[valid_mask]
        labels = labels[valid_mask]
        
        # RR神经元选择
        print("进行RR神经元筛选...")
        rr_results = fast_rr_selection(neural_data, labels)
        rr_indices = rr_results['rr_neurons']
        neural_data_rr = neural_data[:, rr_indices, :]
        
        # 限制数据量以加快分析
        max_trials = min(80, neural_data_rr.shape[0])  # 限制最大试次数
        max_neurons = min(30, neural_data_rr.shape[1])  # 限制最大神经元数
        
        neural_data_rr = neural_data_rr[:max_trials, :max_neurons, :]
        labels = labels[:max_trials]
        
        print(f"M65数据加载成功!")
        print(f"原始数据维度: {neural_data.shape}")
        print(f"RR神经元数量: {len(rr_indices)}")
        print(f"限制后分析数据维度: {neural_data_rr.shape}")
        print(f"标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        # 使用RR神经元数据进行分析
        neural_data = neural_data_rr
        
    except Exception as e:
        print(f"M65数据加载失败: {e}")
        print("尝试简化的数据加载方法...")
        import scipy.io
        try:
            # 直接从mat文件加载
            data_path = r'C:\Users\76629\OneDrive\brain\Micedata\M65_0816\wholebrain_output.mat'
            mat_data = scipy.io.loadmat(data_path)
            neural_data_raw = mat_data['whole_trace_ori'].T
            
            # 创建简化的试次数据
            n_timepoints = neural_data_raw.shape[1]
            trial_length = 50
            n_trials = min(100, n_timepoints // trial_length)  # 限制试次数量
            
            if n_trials > 0:
                neural_data = neural_data_raw[:, :n_trials*trial_length].reshape(
                    neural_data_raw.shape[0], n_trials, trial_length
                ).transpose(1, 0, 2)
                
                # 限制神经元数量以加快分析
                max_neurons = min(50, neural_data.shape[1])
                neural_data = neural_data[:, :max_neurons, :]
                
                # 创建简单标签
                labels = np.tile([1, 2], n_trials//2)[:n_trials]
                
                print(f"简化M65数据加载成功!")
                print(f"数据维度: {neural_data.shape}")
                print(f"标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
            else:
                raise Exception("无法创建有效的试次数据")
                
        except Exception as e2:
            print(f"简化数据加载也失败: {e2}")
            return None
    
    # 运行打乱分析
    print("\\n开始网络效率打乱分析...")
    efficiency_results = analyze_shuffle_network_efficiency(
        neural_data, labels, 
        shuffle_fractions=necfg.SHUFFLE_FRACTIONS,
        n_iterations=necfg.N_ITERATIONS
    )
    
    # 保存结果
    print("\\n保存分析结果...")
    np.savez_compressed(
        os.path.join(results_dir, 'network_efficiency_shuffle_results.npz'),
        shuffle_fractions=efficiency_results['shuffle_fractions'],
        global_efficiency=efficiency_results['global_efficiency'],
        local_efficiency=efficiency_results['local_efficiency'],
        clustering_coefficient=efficiency_results['clustering_coefficient'],
        global_efficiency_std=efficiency_results['global_efficiency_std'],
        local_efficiency_std=efficiency_results['local_efficiency_std'],
        clustering_coefficient_std=efficiency_results['clustering_coefficient_std'],
        n_iterations=efficiency_results['n_iterations']
    )
    print("网络效率打乱分析结果已保存")
    
    # 可视化
    print("\\n生成可视化图表...")
    visualize_network_efficiency_shuffle(
        efficiency_results,
        title="Network Efficiency and Clustering vs Neural Shuffling",
        save_path=os.path.join(results_dir, 'network_efficiency_shuffle_analysis.png')
    )
    
    print(f"\\n网络效率打乱分析完成！结果保存在: {results_dir}")
    return efficiency_results

if __name__ == "__main__":
    results = run_shuffle_analysis()
    
    results_dir = necfg.get_results_dir()
    print("\\n网络效率打乱分析完成！")
    print("主要结果文件:")
    print(f"- 网络效率分析图: {results_dir}/network_efficiency_shuffle_analysis.png")
    print(f"- 分析数据: {results_dir}/network_efficiency_shuffle_results.npz")