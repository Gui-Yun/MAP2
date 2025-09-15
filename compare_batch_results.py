# 批量网络效率分析结果对比
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from loaddata import cfg

def setup_plot_style():
    """设置科研绘图风格"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
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

def load_batch_results():
    """加载批量分析结果"""
    results_dir = os.path.join(cfg.get_results_dir(), 'batch_network_efficiency')
    
    mice_data = {}
    mice_names = ['M27', 'M30', 'M65', 'M74']  # 所有分析的小鼠
    
    for mouse in mice_names:
        result_file = os.path.join(results_dir, f'{mouse.lower()}_network_efficiency_results.npz')
        
        if os.path.exists(result_file):
            print(f"加载{mouse}数据: {result_file}")
            data = np.load(result_file)
            
            mice_data[mouse] = {
                'shuffle_fractions': data['shuffle_fractions'],
                'global_efficiency': data['global_efficiency'],
                'local_efficiency': data['local_efficiency'],
                'clustering_coefficient': data['clustering_coefficient'],
                'global_efficiency_std': data['global_efficiency_std'],
                'local_efficiency_std': data['local_efficiency_std'],
                'clustering_coefficient_std': data['clustering_coefficient_std'],
                'n_iterations': data['n_iterations']
            }
            print(f"  {mouse}数据加载成功")
        else:
            print(f"未找到{mouse}数据文件: {result_file}")
    
    return mice_data

def compare_network_efficiency(mice_data, save_path=None):
    """对比不同小鼠的网络效率分析结果"""
    setup_plot_style()
    
    if not mice_data:
        print("没有找到有效的分析数据")
        return
    
    # 创建2x2子图布局
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 颜色方案
    colors = {'M65': '#2E86AB', 'M74': '#A23B72', 'M30': '#F18F01', 'M27': '#C73E1D'}
    markers = {'M65': 'o', 'M74': 's', 'M30': '^', 'M27': 'D'}
    
    # 1. 全局效率对比
    ax1 = axes[0, 0]
    for mouse, data in mice_data.items():
        ax1.errorbar(data['shuffle_fractions'], data['global_efficiency'], 
                    yerr=data['global_efficiency_std'],
                    marker=markers[mouse], capsize=5, linewidth=2.5, markersize=8,
                    color=colors[mouse], alpha=0.8, label=f'{mouse} Global Efficiency')
    
    ax1.set_xlabel('Shuffle Fraction')
    ax1.set_ylabel('Global Efficiency')
    ax1.set_title('Global Efficiency Comparison Across Mice')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 局部效率对比
    ax2 = axes[0, 1]
    for mouse, data in mice_data.items():
        ax2.errorbar(data['shuffle_fractions'], data['local_efficiency'], 
                    yerr=data['local_efficiency_std'],
                    marker=markers[mouse], capsize=5, linewidth=2.5, markersize=8,
                    color=colors[mouse], alpha=0.8, label=f'{mouse} Local Efficiency')
    
    ax2.set_xlabel('Shuffle Fraction')
    ax2.set_ylabel('Local Efficiency')
    ax2.set_title('Local Efficiency Comparison Across Mice')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 聚类系数对比
    ax3 = axes[1, 0]
    for mouse, data in mice_data.items():
        ax3.errorbar(data['shuffle_fractions'], data['clustering_coefficient'], 
                    yerr=data['clustering_coefficient_std'],
                    marker=markers[mouse], capsize=5, linewidth=2.5, markersize=8,
                    color=colors[mouse], alpha=0.8, label=f'{mouse} Clustering Coefficient')
    
    ax3.set_xlabel('Shuffle Fraction')
    ax3.set_ylabel('Clustering Coefficient')
    ax3.set_title('Clustering Coefficient Comparison Across Mice')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 归一化变化率对比
    ax4 = axes[1, 1]
    
    for mouse, data in mice_data.items():
        # 计算相对变化率
        global_effs = np.array(data['global_efficiency'])
        local_effs = np.array(data['local_efficiency'])
        
        if global_effs[0] > 0:
            global_change = (global_effs - global_effs[0]) / global_effs[0] * 100
            ax4.plot(data['shuffle_fractions'], global_change, 
                    marker=markers[mouse], linewidth=2, markersize=6, 
                    color=colors[mouse], alpha=0.8, linestyle='-',
                    label=f'{mouse} Global')
        
        if local_effs[0] > 0:
            local_change = (local_effs - local_effs[0]) / local_effs[0] * 100
            ax4.plot(data['shuffle_fractions'], local_change, 
                    marker=markers[mouse], linewidth=2, markersize=6, 
                    color=colors[mouse], alpha=0.6, linestyle='--',
                    label=f'{mouse} Local')
    
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Shuffle Fraction')
    ax4.set_ylabel('Change from Original (%)')
    ax4.set_title('Relative Change in Network Efficiency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Network Efficiency Analysis: Cross-Mouse Comparison', y=0.98, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比分析图已保存: {save_path}")
    
    plt.close()

def generate_summary_statistics(mice_data, save_path=None):
    """生成统计摘要"""
    setup_plot_style()
    
    if not mice_data:
        print("没有数据可生成摘要")
        return
    
    # 计算关键统计量
    summary_data = []
    
    for mouse, data in mice_data.items():
        baseline_global = data['global_efficiency'][0]  # 无打乱时的全局效率
        baseline_local = data['local_efficiency'][0]    # 无打乱时的局部效率
        baseline_cluster = data['clustering_coefficient'][0]  # 无打乱时的聚类系数
        
        full_shuffle_global = data['global_efficiency'][-1]  # 完全打乱时的全局效率
        full_shuffle_local = data['local_efficiency'][-1]    # 完全打乱时的局部效率
        full_shuffle_cluster = data['clustering_coefficient'][-1]  # 完全打乱时的聚类系数
        
        # 计算变化率
        global_change = ((full_shuffle_global - baseline_global) / baseline_global * 100) if baseline_global > 0 else 0
        local_change = ((full_shuffle_local - baseline_local) / baseline_local * 100) if baseline_local > 0 else 0
        cluster_change = ((full_shuffle_cluster - baseline_cluster) / baseline_cluster * 100) if baseline_cluster > 0 else 0
        
        summary_data.append({
            'mouse': mouse,
            'baseline_global': baseline_global,
            'baseline_local': baseline_local,
            'baseline_cluster': baseline_cluster,
            'global_change_percent': global_change,
            'local_change_percent': local_change,
            'cluster_change_percent': cluster_change
        })
    
    # 创建摘要可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    mice_names = list(mice_data.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(mice_names)]
    
    # 1. 基线网络指标对比
    ax1 = axes[0, 0]
    baseline_globals = [s['baseline_global'] for s in summary_data]
    baseline_locals = [s['baseline_local'] for s in summary_data]
    baseline_clusters = [s['baseline_cluster'] for s in summary_data]
    
    x = np.arange(len(mice_names))
    width = 0.25
    
    ax1.bar(x - width, baseline_globals, width, label='Global Efficiency', 
            color=colors[0] if len(colors) > 0 else '#2E86AB', alpha=0.8)
    ax1.bar(x, baseline_locals, width, label='Local Efficiency', 
            color=colors[1] if len(colors) > 1 else '#A23B72', alpha=0.8)
    ax1.bar(x + width, baseline_clusters, width, label='Clustering Coefficient', 
            color=colors[2] if len(colors) > 2 else '#F18F01', alpha=0.8)
    
    ax1.set_xlabel('Mouse')
    ax1.set_ylabel('Network Metric Value')
    ax1.set_title('Baseline Network Metrics Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(mice_names)
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    
    # 2. 打乱后变化率对比
    ax2 = axes[0, 1]
    global_changes = [s['global_change_percent'] for s in summary_data]
    local_changes = [s['local_change_percent'] for s in summary_data]
    cluster_changes = [s['cluster_change_percent'] for s in summary_data]
    
    ax2.bar(x - width, global_changes, width, label='Global Efficiency Change', 
            color=colors[0] if len(colors) > 0 else '#2E86AB', alpha=0.8)
    ax2.bar(x, local_changes, width, label='Local Efficiency Change', 
            color=colors[1] if len(colors) > 1 else '#A23B72', alpha=0.8)
    ax2.bar(x + width, cluster_changes, width, label='Clustering Coefficient Change', 
            color=colors[2] if len(colors) > 2 else '#F18F01', alpha=0.8)
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Mouse')
    ax2.set_ylabel('Change (%)')
    ax2.set_title('Network Metrics Change After Full Shuffling')
    ax2.set_xticks(x)
    ax2.set_xticklabels(mice_names)
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    
    # 3. 网络效率衰减对比（显示所有打乱点）
    ax3 = axes[1, 0]
    for i, (mouse, data) in enumerate(mice_data.items()):
        global_effs = np.array(data['global_efficiency'])
        normalized_global = global_effs / global_effs[0] if global_effs[0] > 0 else global_effs
        ax3.plot(data['shuffle_fractions'], normalized_global, 
                marker='o', linewidth=2.5, markersize=6,
                color=colors[i % len(colors)], alpha=0.8, label=f'{mouse}')
    
    ax3.set_xlabel('Shuffle Fraction')
    ax3.set_ylabel('Normalized Global Efficiency')
    ax3.set_title('Global Efficiency Degradation Pattern')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 聚类系数衰减对比
    ax4 = axes[1, 1]
    for i, (mouse, data) in enumerate(mice_data.items()):
        cluster_coeffs = np.array(data['clustering_coefficient'])
        normalized_cluster = cluster_coeffs / cluster_coeffs[0] if cluster_coeffs[0] > 0 else cluster_coeffs
        ax4.plot(data['shuffle_fractions'], normalized_cluster, 
                marker='s', linewidth=2.5, markersize=6,
                color=colors[i % len(colors)], alpha=0.8, label=f'{mouse}')
    
    ax4.set_xlabel('Shuffle Fraction')
    ax4.set_ylabel('Normalized Clustering Coefficient')
    ax4.set_title('Clustering Coefficient Degradation Pattern')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Network Analysis Summary: Cross-Mouse Statistics', y=0.98, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"摘要统计图已保存: {save_path}")
    
    plt.close()
    
    # 打印文本摘要
    print("\n=== 网络效率分析摘要 ===")
    for data in summary_data:
        print(f"\n{data['mouse']}小鼠:")
        print(f"  基线全局效率: {data['baseline_global']:.4f}")
        print(f"  基线局部效率: {data['baseline_local']:.4f}")
        print(f"  基线聚类系数: {data['baseline_cluster']:.4f}")
        print(f"  完全打乱后变化:")
        print(f"    全局效率变化: {data['global_change_percent']:+.1f}%")
        print(f"    局部效率变化: {data['local_change_percent']:+.1f}%")
        print(f"    聚类系数变化: {data['cluster_change_percent']:+.1f}%")

def run_comparison():
    """运行对比分析"""
    print("="*60)
    print("批量网络效率分析结果对比")
    print("="*60)
    
    # 加载数据
    print("\n加载批量分析结果...")
    mice_data = load_batch_results()
    
    if not mice_data:
        print("没有找到有效的分析数据")
        return
    
    print(f"成功加载{len(mice_data)}只小鼠的数据: {list(mice_data.keys())}")
    
    # 确保结果目录存在
    results_dir = os.path.join(cfg.get_results_dir(), 'batch_network_efficiency')
    os.makedirs(results_dir, exist_ok=True)
    
    # 生成对比分析
    print("\n生成网络效率对比分析...")
    compare_network_efficiency(
        mice_data,
        save_path=os.path.join(results_dir, 'cross_mouse_comparison.png')
    )
    
    # 生成摘要统计
    print("\n生成摘要统计...")
    generate_summary_statistics(
        mice_data,
        save_path=os.path.join(results_dir, 'summary_statistics.png')
    )
    
    print(f"\n对比分析完成！结果保存在: {results_dir}")
    print("生成的文件:")
    print("- cross_mouse_comparison.png: 跨小鼠网络效率对比")
    print("- summary_statistics.png: 摘要统计图表")

if __name__ == "__main__":
    run_comparison()