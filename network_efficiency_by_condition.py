# 按条件的网络效率神经元打乱分析
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
from collections import defaultdict

# 导入必要模块
from loaddata import cfg, load_data, segment_neuron_data, reclassify_labels, fast_rr_selection
from noise_correlation_analysis import (
    NoiseCorrelationConfig, shuffle_within_condition
)

class ConditionNetworkEfficiencyConfig(NoiseCorrelationConfig):
    """按条件网络效率分析配置"""
    SHUFFLE_FRACTIONS = [0.0, 0.3, 0.6, 1.0]  # 测试点
    N_ITERATIONS = 3  # 迭代次数
    MAX_TRIALS_PER_CONDITION = 40  # 每个条件最大试次数
    MAX_NEURONS = 50  # 限制最大神经元数
    
    @classmethod
    def get_results_dir(cls):
        return os.path.join(cfg.get_results_dir(), 'network_efficiency_by_condition')
    
    @classmethod
    def ensure_results_dir(cls):
        results_dir = cls.get_results_dir()
        os.makedirs(results_dir, exist_ok=True)
        return results_dir

config = ConditionNetworkEfficiencyConfig()

def setup_plot_style():
    """设置科研绘图风格"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
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

def analyze_condition_network_efficiency(neural_data, labels, condition, shuffle_fractions=None, n_iterations=3):
    """分析特定条件下的网络效率变化"""
    if shuffle_fractions is None:
        shuffle_fractions = config.SHUFFLE_FRACTIONS
    
    print(f"分析条件{condition}的网络效率打乱分析...")
    print(f"- 数据维度: {neural_data.shape}")
    print(f"- 展开数据计算相关性")
    print(f"- 只使用正相关值构建网络")
    print(f"- 每个打乱比例重复{n_iterations}次取平均")
    
    results = {
        'condition': condition,
        'shuffle_fractions': shuffle_fractions,
        'global_efficiency': [],
        'local_efficiency': [],
        'clustering_coefficient': [],
        'global_efficiency_std': [],
        'local_efficiency_std': [],
        'clustering_coefficient_std': [],
        'n_iterations': n_iterations,
        'n_trials': neural_data.shape[0],
        'n_neurons': neural_data.shape[1]
    }
    
    for fraction in shuffle_fractions:
        print(f"  打乱比例: {fraction:.1f}")
        
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
            corr_matrix = calculate_flattened_correlation(shuffled_data)
            
            # 构建网络：只使用正相关值
            pos_corr = np.copy(corr_matrix)
            pos_corr[pos_corr < 0] = 0  # 将负相关设为0
            np.fill_diagonal(pos_corr, 0)  # 对角线设为0
            
            # 使用密度阈值方法
            n_nodes = pos_corr.shape[0]
            n_possible_edges = n_nodes * (n_nodes - 1) // 2
            n_edges_to_keep = int(n_possible_edges * config.NETWORK_DENSITY)
            
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
        
        print(f"    全局效率: {results['global_efficiency'][-1]:.4f} ± {results['global_efficiency_std'][-1]:.4f}")
        print(f"    局部效率: {results['local_efficiency'][-1]:.4f} ± {results['local_efficiency_std'][-1]:.4f}")
        print(f"    聚类系数: {results['clustering_coefficient'][-1]:.4f} ± {results['clustering_coefficient_std'][-1]:.4f}")
    
    return results

def visualize_condition_comparison(condition_results, title="Network Efficiency by Condition", save_path=None):
    """可视化不同条件的网络效率对比"""
    setup_plot_style()
    
    # 创建2x2子图布局
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 定义颜色
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    shuffle_fractions = condition_results[0]['shuffle_fractions']
    
    # 1. 全局效率对比
    ax1 = axes[0, 0]
    for i, result in enumerate(condition_results):
        condition = result['condition']
        global_effs = result['global_efficiency']
        global_stds = result['global_efficiency_std']
        
        ax1.errorbar(shuffle_fractions, global_effs, yerr=global_stds,
                    marker='o', capsize=3, linewidth=2, markersize=6,
                    color=colors[i % len(colors)], alpha=0.8, label=f'Condition {condition}')
        
        # 填充标准误区域
        ax1.fill_between(shuffle_fractions, 
                        np.array(global_effs) - np.array(global_stds),
                        np.array(global_effs) + np.array(global_stds),
                        alpha=0.2, color=colors[i % len(colors)])
    
    ax1.set_xlabel('Shuffle Fraction')
    ax1.set_ylabel('Global Efficiency')
    ax1.set_title('Global Efficiency vs Shuffle Fraction')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 局部效率对比
    ax2 = axes[0, 1]
    for i, result in enumerate(condition_results):
        condition = result['condition']
        local_effs = result['local_efficiency']
        local_stds = result['local_efficiency_std']
        
        ax2.errorbar(shuffle_fractions, local_effs, yerr=local_stds,
                    marker='s', capsize=3, linewidth=2, markersize=6,
                    color=colors[i % len(colors)], alpha=0.8, label=f'Condition {condition}')
        
        ax2.fill_between(shuffle_fractions, 
                        np.array(local_effs) - np.array(local_stds),
                        np.array(local_effs) + np.array(local_stds),
                        alpha=0.2, color=colors[i % len(colors)])
    
    ax2.set_xlabel('Shuffle Fraction')
    ax2.set_ylabel('Local Efficiency')
    ax2.set_title('Local Efficiency vs Shuffle Fraction')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 聚类系数对比
    ax3 = axes[1, 0]
    for i, result in enumerate(condition_results):
        condition = result['condition']
        clusterings = result['clustering_coefficient']
        clustering_stds = result['clustering_coefficient_std']
        
        ax3.errorbar(shuffle_fractions, clusterings, yerr=clustering_stds,
                    marker='^', capsize=3, linewidth=2, markersize=6,
                    color=colors[i % len(colors)], alpha=0.8, label=f'Condition {condition}')
        
        ax3.fill_between(shuffle_fractions, 
                        np.array(clusterings) - np.array(clustering_stds),
                        np.array(clusterings) + np.array(clustering_stds),
                        alpha=0.2, color=colors[i % len(colors)])
    
    ax3.set_xlabel('Shuffle Fraction')
    ax3.set_ylabel('Clustering Coefficient')
    ax3.set_title('Clustering Coefficient vs Shuffle Fraction')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 归一化变化率对比
    ax4 = axes[1, 1]
    
    for i, result in enumerate(condition_results):
        condition = result['condition']
        # 计算相对于原始值（fraction=0）的变化率
        global_effs = np.array(result['global_efficiency'])
        if global_effs[0] > 0:
            global_change = (global_effs - global_effs[0]) / global_effs[0] * 100
            ax4.plot(shuffle_fractions, global_change, 
                    marker='o', linewidth=2, markersize=5, color=colors[i % len(colors)], alpha=0.8, 
                    label=f'Condition {condition} (Global)')
        
        local_effs = np.array(result['local_efficiency'])
        if local_effs[0] > 0:
            local_change = (local_effs - local_effs[0]) / local_effs[0] * 100
            ax4.plot(shuffle_fractions, local_change, 
                    marker='s', linewidth=2, markersize=5, color=colors[i % len(colors)], alpha=0.6, 
                    linestyle='--', label=f'Condition {condition} (Local)')
    
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Shuffle Fraction')
    ax4.set_ylabel('Change from Original (%)')
    ax4.set_title('Relative Change in Network Metrics')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, y=0.98, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"按条件网络效率分析图已保存: {save_path}")
    
    plt.close()

def run_condition_analysis(data_path=None, mouse_name="M65"):
    """运行按条件的网络效率分析"""
    print("=" * 70)
    print(f"按条件网络效率打乱分析 - {mouse_name}数据")
    print("=" * 70)
    
    # 确保结果目录存在
    results_dir = config.ensure_results_dir()
    
    # 加载数据
    print(f"\n加载{mouse_name}数据...")
    if data_path is None:
        data_path = f'C:\\Users\\76629\\OneDrive\\brain\\Micedata\\{mouse_name}_0816'
    
    try:
        # 加载原始数据
        neural_data_raw, neuron_pos, start_edges, stimulus_data = load_data(data_path)
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
        
        # 限制神经元数量
        max_neurons = min(config.MAX_NEURONS, neural_data_rr.shape[1])
        neural_data_rr = neural_data_rr[:, :max_neurons, :]
        
        print(f"数据加载成功!")
        print(f"原始数据维度: {neural_data.shape}")
        print(f"RR神经元数量: {len(rr_indices)}")
        print(f"分析数据维度: {neural_data_rr.shape}")
        print(f"标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        # 按条件分组数据
        unique_conditions = np.unique(labels)
        condition_results = []
        
        print(f"\n开始按条件分析，共{len(unique_conditions)}个条件...")
        
        for condition in unique_conditions:
            print(f"\n{'='*50}")
            print(f"分析条件 {condition}")
            print(f"{'='*50}")
            
            # 提取该条件的数据
            condition_mask = labels == condition
            condition_data = neural_data_rr[condition_mask]
            condition_labels = labels[condition_mask]
            
            # 限制每个条件的试次数
            max_trials = min(config.MAX_TRIALS_PER_CONDITION, condition_data.shape[0])
            condition_data = condition_data[:max_trials]
            condition_labels = condition_labels[:max_trials]
            
            print(f"条件{condition}数据维度: {condition_data.shape}")
            
            if condition_data.shape[0] >= 10:  # 至少需要10个试次
                # 分析该条件的网络效率
                result = analyze_condition_network_efficiency(
                    condition_data, condition_labels, condition,
                    shuffle_fractions=config.SHUFFLE_FRACTIONS,
                    n_iterations=config.N_ITERATIONS
                )
                condition_results.append(result)
                
                # 保存该条件的结果
                np.savez_compressed(
                    os.path.join(results_dir, f'{mouse_name.lower()}_condition_{condition}_results.npz'),
                    **result
                )
                print(f"条件{condition}分析完成并保存")
            else:
                print(f"条件{condition}试次数量不足({condition_data.shape[0]} < 10)，跳过分析")
        
        if len(condition_results) > 0:
            # 保存汇总结果
            print(f"\n保存汇总结果...")
            summary_data = {
                'mouse_name': mouse_name,
                'n_conditions': len(condition_results),
                'conditions': [r['condition'] for r in condition_results],
                'condition_results': condition_results
            }
            
            np.savez_compressed(
                os.path.join(results_dir, f'{mouse_name.lower()}_all_conditions_summary.npz'),
                **summary_data
            )
            
            # 可视化对比
            print(f"\n生成条件对比可视化...")
            visualize_condition_comparison(
                condition_results,
                title=f"{mouse_name} Network Efficiency by Condition",
                save_path=os.path.join(results_dir, f'{mouse_name.lower()}_condition_comparison.png')
            )
            
            # 打印汇总信息
            print(f"\n{'='*70}")
            print(f"按条件网络效率分析完成！")
            print(f"{'='*70}")
            print(f"分析了 {len(condition_results)} 个条件:")
            
            for result in condition_results:
                condition = result['condition']
                baseline_global = result['global_efficiency'][0]
                final_global = result['global_efficiency'][-1]
                global_change = (final_global - baseline_global) / baseline_global * 100 if baseline_global > 0 else 0
                
                baseline_local = result['local_efficiency'][0]
                final_local = result['local_efficiency'][-1]
                local_change = (final_local - baseline_local) / baseline_local * 100 if baseline_local > 0 else 0
                
                print(f"  条件{condition}: 试次数={result['n_trials']}, 神经元数={result['n_neurons']}")
                print(f"    基线全局效率: {baseline_global:.4f}, 完全打乱后变化: {global_change:+.1f}%")
                print(f"    基线局部效率: {baseline_local:.4f}, 完全打乱后变化: {local_change:+.1f}%")
            
            print(f"\n结果保存在: {results_dir}")
            return condition_results
        else:
            print("没有足够的数据进行条件分析")
            return None
            
    except Exception as e:
        print(f"数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # 运行M65的按条件分析
    results = run_condition_analysis(mouse_name="M65")
    
    if results:
        print("\n按条件网络效率分析完成！")
        results_dir = config.get_results_dir()
        print("主要结果文件:")
        print(f"- 条件对比图: {results_dir}/m65_condition_comparison.png")
        print(f"- 汇总数据: {results_dir}/m65_all_conditions_summary.npz")
        print(f"- 各条件详细结果: {results_dir}/m65_condition_*_results.npz")