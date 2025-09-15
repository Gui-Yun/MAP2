# 网络效率与聚类系数分析
# 分析神经元置换对网络效率和聚类系数的影响
# guiy24@mails.tsinghua.edu.cn

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 导入项目模块
from loaddata import (
    load_data, segment_neuron_data, reclassify_labels, 
    fast_rr_selection, cfg
)
from network import NetworkConfig, compute_network_metrics
from noise_correlation_analysis import (
    NoiseCorrelationConfig, calculate_noise_correlation, 
    shuffle_within_condition, build_networks_from_correlations
)

# 配置参数
class NetworkEfficiencyConfig(NoiseCorrelationConfig):
    """网络效率分析配置，继承噪音相关性分析配置"""
    
    # 网络效率分析特定参数
    SHUFFLE_FRACTIONS = [0.0, 0.5, 1.0]  # 最小化测试点
    N_ITERATIONS = 1  # 最小化迭代次数
    
    # 结果保存路径
    @classmethod
    def get_results_dir(cls):
        """获取网络效率分析结果保存目录"""
        from loaddata import cfg
        return os.path.join(cfg.get_results_dir(), 'network_efficiency')
    
    @classmethod
    def ensure_results_dir(cls):
        """确保结果目录存在"""
        results_dir = cls.get_results_dir()
        os.makedirs(results_dir, exist_ok=True)
        return results_dir

necfg = NetworkEfficiencyConfig()

def setup_efficiency_plot_style():
    """设置网络效率分析科研绘图风格"""
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
    """
    计算网络的全局效率和局部效率
    
    Parameters:
    -----------
    adj_matrix : ndarray
        邻接矩阵
        
    Returns:
    --------
    dict: 包含全局效率和局部效率的字典
    """
    import networkx as nx
    
    # 创建NetworkX图
    G = nx.from_numpy_array(adj_matrix)
    
    # 计算全局效率
    try:
        global_efficiency = nx.global_efficiency(G)
    except:
        global_efficiency = 0.0
    
    # 计算局部效率
    try:
        local_efficiency = nx.local_efficiency(G)
    except:
        local_efficiency = 0.0
    
    return {
        'global_efficiency': global_efficiency,
        'local_efficiency': local_efficiency,
        'clustering_coefficient': nx.average_clustering(G)
    }

def analyze_shuffle_network_efficiency(neural_data, labels, shuffle_fractions=None, n_iterations=5):
    """
    分析不同程度神经元置换对网络效率和聚类系数的影响
    
    Parameters:
    -----------
    neural_data : ndarray, shape (n_trials, n_neurons, n_timepoints)
        神经活动数据
    labels : ndarray, shape (n_trials,)
        试次标签
    shuffle_fractions : list, optional
        置换比例列表
    n_iterations : int, optional
        每个置换比例的重复次数
        
    Returns:
    --------
    results : dict
        分析结果
    """
    if shuffle_fractions is None:
        shuffle_fractions = necfg.SHUFFLE_FRACTIONS
    
    print(f"分析神经元置换对网络效率的影响...")
    print(f"每个置换比例重复{n_iterations}次取平均")
    
    results = {
        'shuffle_fractions': shuffle_fractions,
        'conditions': {},
        'n_iterations': n_iterations
    }
    
    unique_conditions = np.unique(labels[labels != 0])
    
    for condition in unique_conditions:
        print(f"\n分析条件 {condition}")
        
        condition_results = {
            'global_efficiency': [],
            'local_efficiency': [],
            'clustering_coefficient': [],
            'global_efficiency_std': [],
            'local_efficiency_std': [],
            'clustering_coefficient_std': []
        }
        
        for fraction in shuffle_fractions:
            print(f"  置换比例: {fraction:.1f}")
            
            # 存储每次迭代的结果
            global_effs = []
            local_effs = []
            clusterings = []
            
            for iteration in range(n_iterations):
                # 打乱数据
                if fraction == 0.0:
                    # 不打乱，使用原始数据
                    shuffled_data = neural_data.copy()
                else:
                    shuffled_data = shuffle_within_condition(neural_data, labels, fraction)
                
                # 计算噪音相关性
                noise_corr = calculate_noise_correlation(shuffled_data, labels)
                
                if condition in noise_corr:
                    # 构建网络
                    networks = build_networks_from_correlations(
                        {condition: noise_corr[condition]}, 
                        method=necfg.NETWORK_METHOD,
                        threshold=necfg.NETWORK_THRESHOLD, 
                        density=necfg.NETWORK_DENSITY
                    )
                    
                    if condition in networks:
                        adj_matrix = networks[condition]['adjacency_matrix']
                        
                        # 计算网络效率
                        efficiency_metrics = calculate_network_efficiency(adj_matrix)
                        
                        global_effs.append(efficiency_metrics['global_efficiency'])
                        local_effs.append(efficiency_metrics['local_efficiency'])
                        clusterings.append(efficiency_metrics['clustering_coefficient'])
            
            # 计算统计量
            condition_results['global_efficiency'].append(np.mean(global_effs) if global_effs else 0.0)
            condition_results['local_efficiency'].append(np.mean(local_effs) if local_effs else 0.0)
            condition_results['clustering_coefficient'].append(np.mean(clusterings) if clusterings else 0.0)
            
            condition_results['global_efficiency_std'].append(np.std(global_effs) if len(global_effs) > 1 else 0.0)
            condition_results['local_efficiency_std'].append(np.std(local_effs) if len(local_effs) > 1 else 0.0)
            condition_results['clustering_coefficient_std'].append(np.std(clusterings) if len(clusterings) > 1 else 0.0)
            
            print(f"    全局效率: {condition_results['global_efficiency'][-1]:.4f} ± {condition_results['global_efficiency_std'][-1]:.4f}")
            print(f"    局部效率: {condition_results['local_efficiency'][-1]:.4f} ± {condition_results['local_efficiency_std'][-1]:.4f}")
            print(f"    聚类系数: {condition_results['clustering_coefficient'][-1]:.4f} ± {condition_results['clustering_coefficient_std'][-1]:.4f}")
        
        results['conditions'][condition] = condition_results
    
    return results

def visualize_network_efficiency_shuffle(results, title="Network Efficiency vs Shuffle Fraction", save_path=None):
    """
    可视化网络效率随置换比例的变化
    
    Parameters:
    -----------
    results : dict
        analyze_shuffle_network_efficiency的结果
    title : str
        图标题
    save_path : str, optional
        保存路径
    """
    setup_efficiency_plot_style()
    
    shuffle_fractions = results['shuffle_fractions']
    conditions = list(results['conditions'].keys())
    
    # 创建2x2子图布局
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(conditions)))
    
    # 1. 全局效率
    ax1 = axes[0, 0]
    for i, condition in enumerate(conditions):
        data = results['conditions'][condition]
        global_effs = data['global_efficiency']
        global_stds = data['global_efficiency_std']
        
        ax1.errorbar(shuffle_fractions, global_effs, yerr=global_stds,
                    marker='o', capsize=5, linewidth=2.5, markersize=6,
                    color=colors[i], alpha=0.8, label=f'Condition {condition}')
        
        # 填充标准误区域
        ax1.fill_between(shuffle_fractions, 
                        np.array(global_effs) - np.array(global_stds),
                        np.array(global_effs) + np.array(global_stds),
                        alpha=0.2, color=colors[i])
    
    ax1.set_xlabel('Shuffle Fraction')
    ax1.set_ylabel('Global Efficiency')
    ax1.set_title('Global Efficiency vs Shuffle Fraction')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 局部效率
    ax2 = axes[0, 1]
    for i, condition in enumerate(conditions):
        data = results['conditions'][condition]
        local_effs = data['local_efficiency']
        local_stds = data['local_efficiency_std']
        
        ax2.errorbar(shuffle_fractions, local_effs, yerr=local_stds,
                    marker='s', capsize=5, linewidth=2.5, markersize=6,
                    color=colors[i], alpha=0.8, label=f'Condition {condition}')
        
        ax2.fill_between(shuffle_fractions, 
                        np.array(local_effs) - np.array(local_stds),
                        np.array(local_effs) + np.array(local_stds),
                        alpha=0.2, color=colors[i])
    
    ax2.set_xlabel('Shuffle Fraction')
    ax2.set_ylabel('Local Efficiency')
    ax2.set_title('Local Efficiency vs Shuffle Fraction')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 聚类系数
    ax3 = axes[1, 0]
    for i, condition in enumerate(conditions):
        data = results['conditions'][condition]
        clusterings = data['clustering_coefficient']
        clustering_stds = data['clustering_coefficient_std']
        
        ax3.errorbar(shuffle_fractions, clusterings, yerr=clustering_stds,
                    marker='^', capsize=5, linewidth=2.5, markersize=6,
                    color=colors[i], alpha=0.8, label=f'Condition {condition}')
        
        ax3.fill_between(shuffle_fractions, 
                        np.array(clusterings) - np.array(clustering_stds),
                        np.array(clusterings) + np.array(clustering_stds),
                        alpha=0.2, color=colors[i])
    
    ax3.set_xlabel('Shuffle Fraction')
    ax3.set_ylabel('Clustering Coefficient')
    ax3.set_title('Clustering Coefficient vs Shuffle Fraction')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 归一化变化率
    ax4 = axes[1, 1]
    for i, condition in enumerate(conditions):
        data = results['conditions'][condition]
        
        # 计算相对于原始值（fraction=0）的变化率
        global_effs = np.array(data['global_efficiency'])
        local_effs = np.array(data['local_efficiency'])
        
        if global_effs[0] > 0:
            global_change = (global_effs - global_effs[0]) / global_effs[0] * 100
            ax4.plot(shuffle_fractions, global_change, 
                    marker='o', linewidth=2, color=colors[i], alpha=0.8, 
                    label=f'Global Efficiency (Cond {condition})')
        
        if local_effs[0] > 0:
            local_change = (local_effs - local_effs[0]) / local_effs[0] * 100
            ax4.plot(shuffle_fractions, local_change, 
                    marker='s', linewidth=2, color=colors[i], alpha=0.6, linestyle='--',
                    label=f'Local Efficiency (Cond {condition})')
    
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Shuffle Fraction')
    ax4.set_ylabel('Change from Original (%)')
    ax4.set_title('Relative Change in Network Efficiency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, y=0.98, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=necfg.DPI, bbox_inches='tight')
        print(f"网络效率分析图已保存: {save_path}")
    
    plt.show()

def run_network_efficiency_analysis():
    """运行完整的网络效率分析流程（最小化实现）"""
    print("=" * 60)
    print("网络效率与聚类系数分析")
    print("=" * 60)
    
    # 确保结果目录存在
    results_dir = necfg.ensure_results_dir()
    
    # 1. 加载和预处理数据
    print("\n1. 数据加载与预处理")
    print("-" * 30)
    
    try:
        # 加载数据（复用noise_correlation_analysis.py的逻辑）
        if cfg.LOADER_VERSION == 'new':
            print("使用新版数据加载器...")
            neural_data_raw, neuron_pos, start_edges, stimulus_data = load_data(cfg.DATA_PATH)
            segments, labels = segment_neuron_data(neural_data_raw, start_edges, stimulus_data)
            neural_data = np.array(segments)
            labels = np.array(labels)
            labels = reclassify_labels(stimulus_data)
            print(f"新版数据加载成功: {neural_data.shape}")
        elif cfg.LOADER_VERSION == 'old':
            print("使用旧版数据加载器...")
            from loaddata import load_old_version_data
            neuron_index, neural_data, labels, neuron_pos = load_old_version_data(
                cfg.OLD_VERSION_PATHS['neurons'],
                cfg.OLD_VERSION_PATHS['trials'],
                cfg.OLD_VERSION_PATHS['location']
            )
            print(f"旧版数据加载成功: {neural_data.shape}")
        else:
            raise ValueError("无效的 LOADER_VERSION 配置")
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("尝试使用简化的数据加载方法...")
        # 简化的数据加载方法
        import scipy.io
        try:
            data_path = r'C:\Users\76629\OneDrive\brain\Micedata\M65_0816\wholebrain_output.mat'
            mat_data = scipy.io.loadmat(data_path)
            neural_data_raw = mat_data['whole_trace_ori'].T  # (时间点, 神经元) -> (神经元, 时间点)
            # 创建虚拟的试次数据用于测试
            n_timepoints = neural_data_raw.shape[1]
            trial_length = 50  # 假设每个试次50个时间点
            n_trials = n_timepoints // trial_length
            neural_data = neural_data_raw[:, :n_trials*trial_length].reshape(neural_data_raw.shape[0], n_trials, trial_length)
            neural_data = neural_data.transpose(1, 0, 2)  # (试次, 神经元, 时间点)
            labels = np.repeat([1, 2], n_trials//2)[:n_trials]  # 简单的标签
            print(f"简化数据加载成功: {neural_data.shape}, 标签: {len(labels)}")
        except Exception as e2:
            print(f"简化数据加载也失败: {e2}")
            return None
    
    # 过滤和RR神经元选择
    valid_mask = labels != 0
    neural_data = neural_data[valid_mask]
    labels = labels[valid_mask]
    
    rr_results = fast_rr_selection(neural_data, labels)
    rr_indices = rr_results['rr_neurons']
    neural_data_rr = neural_data[:, rr_indices, :]
    
    print(f"数据维度: {neural_data_rr.shape}")
    print(f"标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
    print(f"RR神经元数量: {len(rr_indices)}")
    
    # 2. 网络效率置换分析
    print("\n2. 网络效率置换分析")
    print("-" * 30)
    
    efficiency_results = analyze_shuffle_network_efficiency(
        neural_data_rr, labels, 
        shuffle_fractions=necfg.SHUFFLE_FRACTIONS,
        n_iterations=necfg.N_ITERATIONS
    )
    
    # 3. 保存结果
    print("\n3. 保存分析结果")
    print("-" * 30)
    
    # 保存分析结果
    np.savez_compressed(
        os.path.join(results_dir, 'network_efficiency_results.npz'),
        shuffle_fractions=efficiency_results['shuffle_fractions'],
        conditions=efficiency_results['conditions'],
        n_iterations=efficiency_results['n_iterations']
    )
    print("网络效率分析结果已保存")
    
    # 4. 可视化
    print("\n4. 生成可视化图表")
    print("-" * 30)
    
    visualize_network_efficiency_shuffle(
        efficiency_results,
        title="Network Efficiency and Clustering vs Neural Shuffling",
        save_path=os.path.join(results_dir, 'network_efficiency_analysis.png')
    )
    
    # 5. 生成简要报告
    print("\n5. 生成分析报告")
    print("-" * 30)
    
    report = generate_efficiency_report(efficiency_results)
    report_path = os.path.join(results_dir, 'efficiency_analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n网络效率分析完成！结果保存在: {results_dir}")
    return efficiency_results

def generate_efficiency_report(results):
    """生成网络效率分析报告"""
    report = []
    report.append("=" * 60)
    report.append("网络效率与聚类系数分析报告")
    report.append("=" * 60)
    report.append("")
    
    # 基本信息
    report.append("## 基本信息")
    report.append(f"- 分析条件数: {len(results['conditions'])}")
    report.append(f"- 置换比例: {results['shuffle_fractions']}")
    report.append(f"- 每个比例重复次数: {results['n_iterations']}")
    report.append("")
    
    # 各条件的结果
    report.append("## 各条件网络效率分析结果")
    for condition, data in results['conditions'].items():
        report.append(f"\n### 条件 {condition}")
        
        original_global = data['global_efficiency'][0]
        original_local = data['local_efficiency'][0]
        original_clustering = data['clustering_coefficient'][0]
        
        final_global = data['global_efficiency'][-1]
        final_local = data['local_efficiency'][-1]
        final_clustering = data['clustering_coefficient'][-1]
        
        global_change = ((final_global - original_global) / original_global * 100) if original_global > 0 else 0
        local_change = ((final_local - original_local) / original_local * 100) if original_local > 0 else 0
        clustering_change = ((final_clustering - original_clustering) / original_clustering * 100) if original_clustering > 0 else 0
        
        report.append(f"- 原始全局效率: {original_global:.4f}")
        report.append(f"- 最终全局效率: {final_global:.4f} (变化 {global_change:+.1f}%)")
        report.append(f"- 原始局部效率: {original_local:.4f}")
        report.append(f"- 最终局部效率: {final_local:.4f} (变化 {local_change:+.1f}%)")
        report.append(f"- 原始聚类系数: {original_clustering:.4f}")
        report.append(f"- 最终聚类系数: {final_clustering:.4f} (变化 {clustering_change:+.1f}%)")
    
    report.append("\n## 主要发现")
    
    # 找出效率变化最大的条件
    max_global_change = 0
    max_global_condition = None
    
    for condition, data in results['conditions'].items():
        original = data['global_efficiency'][0]
        final = data['global_efficiency'][-1]
        if original > 0:
            change = abs((final - original) / original)
            if change > max_global_change:
                max_global_change = change
                max_global_condition = condition
    
    if max_global_condition:
        report.append(f"- 全局效率变化最大的条件: {max_global_condition} ({max_global_change*100:.1f}%)")
    
    return "\n".join(report)

# 主程序执行
if __name__ == "__main__":
    results = run_network_efficiency_analysis()
    
    results_dir = necfg.get_results_dir()
    print("\n网络效率分析完成！")
    print("主要结果文件:")
    print(f"- 网络效率分析图: {results_dir}/network_efficiency_analysis.png")
    print(f"- 分析数据: {results_dir}/network_efficiency_results.npz")
    print(f"- 分析报告: {results_dir}/efficiency_analysis_report.txt")