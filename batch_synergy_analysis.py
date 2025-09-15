# 批量协同信息打乱分析 - 所有小鼠（使用枢纽神经元）
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from synergy_shuffle_analysis import (
    run_synergy_shuffle_analysis, 
    SynergyShuffleConfig, 
    setup_plot_style,
    visualize_synergy_shuffle_results
)
from network import (
    compute_correlation_matrix, threshold_correlation_matrix
)

class BatchSynergyConfig(SynergyShuffleConfig):
    """批量协同信息分析配置（枢纽神经元版本）"""
    MAX_TRIALS = 50  # 减少试次数以加快批量分析
    MAX_NEURONS = 20  # 减少神经元数以加快批量分析
    N_NEURON_PAIRS = 30  # 减少神经元对数量
    N_ITERATIONS = 2  # 减少迭代次数
    
    # 枢纽神经元识别参数
    HUB_PERCENTILE = 90  # 枢纽神经元百分位阈值
    STIMULUS_START = 10  # 刺激开始时间点
    STIMULUS_DURATION = 20  # 刺激持续时间
    NETWORK_DENSITY = 0.1  # 网络密度阈值
    
    @classmethod
    def get_batch_results_dir(cls):
        from loaddata import cfg
        return os.path.join(cfg.get_results_dir(), 'batch_synergy_hub_analysis')
    
    @classmethod
    def ensure_batch_results_dir(cls):
        results_dir = cls.get_batch_results_dir()
        os.makedirs(results_dir, exist_ok=True)
        return results_dir

batch_config = BatchSynergyConfig()

def identify_hub_neurons(neural_data, labels):
    """
    识别枢纽神经元基于功能连接度中心性
    
    参数:
    neural_data: 神经数据 (trials, neurons, timepoints)
    labels: 标签数组
    
    返回:
    hub_indices: 枢纽神经元索引列表
    degree_values: 所有神经元的度值
    """
    print("基于功能连接度中心性识别枢纽神经元...")
    
    # 过滤有效数据
    valid_mask = labels != 0
    valid_data = neural_data[valid_mask]
    
    # 提取刺激期数据
    stimulus_window = np.arange(batch_config.STIMULUS_START, 
                               min(batch_config.STIMULUS_START + batch_config.STIMULUS_DURATION, 
                                   valid_data.shape[2]))
    
    # 计算刺激期平均活动
    neural_activity = np.mean(valid_data[:, :, stimulus_window], axis=2)
    
    print(f"神经活动数据维度: {neural_activity.shape}")
    
    # 计算相关性矩阵
    try:
        corr_matrix, p_matrix = compute_correlation_matrix(neural_activity, method='pearson')
        
        # 构建邻接矩阵
        adj_matrix = threshold_correlation_matrix(
            corr_matrix, p_matrix, 
            method='density', 
            network_density=batch_config.NETWORK_DENSITY
        )
        
        # 创建网络并计算度中心性
        G = nx.from_numpy_array(adj_matrix)
        degrees = dict(G.degree())
        degree_values = list(degrees.values())
        
        # 识别枢纽神经元
        hub_threshold = np.percentile(degree_values, batch_config.HUB_PERCENTILE)
        hub_indices = [i for i in range(len(degree_values)) if degree_values[i] >= hub_threshold]
        
        print(f"枢纽神经元阈值: {hub_threshold} (第{batch_config.HUB_PERCENTILE}百分位)")
        print(f"识别枢纽神经元: {len(hub_indices)} 个")
        print(f"平均度值: {np.mean(degree_values):.2f}, 最大度值: {np.max(degree_values)}")
        
        if len(hub_indices) == 0:
            print("警告: 未识别到枢纽神经元，使用度值最高的前10%神经元")
            n_top = max(1, len(degree_values) // 10)
            sorted_indices = np.argsort(degree_values)[::-1]
            hub_indices = sorted_indices[:n_top].tolist()
            print(f"使用备选方案，选择前{len(hub_indices)}个高度值神经元")
        
        return hub_indices, degree_values
        
    except Exception as e:
        print(f"枢纽神经元识别失败: {e}")
        print("使用备选方案: 选择前20%神经元作为枢纽神经元")
        
        # 备选方案：随机选择前20%的神经元
        n_neurons = neural_activity.shape[1]
        n_hub = max(1, n_neurons // 5)  # 20%
        hub_indices = list(range(n_hub))
        degree_values = [0] * n_neurons
        
        return hub_indices, degree_values

def run_mouse_synergy_analysis(mouse_data_path, mouse_name):
    """为单只小鼠运行协同信息分析"""
    print(f"\n{'='*50}")
    print(f"分析{mouse_name}小鼠协同信息")
    print(f"{'='*50}")
    
    try:
        # 检测数据格式并加载数据
        print("加载数据...")
        
        # 检查是否为旧版数据格式
        neurons_mat = os.path.join(mouse_data_path, 'Neurons.mat')
        trials_mat = os.path.join(mouse_data_path, 'Trial_data.mat')
        location_mat = os.path.join(mouse_data_path, 'wholebrain_output.mat')
        
        if os.path.exists(neurons_mat) and os.path.exists(trials_mat):
            # 旧版数据处理
            print(f"检测到旧版数据格式，使用简化协同信息分析")
            return analyze_old_format_synergy(mouse_data_path, mouse_name)
        else:
            # 新版数据处理
            print(f"使用新版数据加载方法")
            
            from loaddata import load_data, segment_neuron_data, reclassify_labels, fast_rr_selection
            from noise_correlation_analysis import shuffle_within_condition
            from synergy_shuffle_analysis import analyze_synergy_shuffle_effects
            
            # 加载数据
            neural_data_raw, neuron_pos, start_edges, stimulus_data = load_data(mouse_data_path)
            segments, labels = segment_neuron_data(neural_data_raw, start_edges, stimulus_data)
            neural_data = np.array(segments)
            labels = np.array(labels)
            labels = reclassify_labels(stimulus_data)
            
            # 过滤有效数据
            valid_mask = labels != 0
            neural_data = neural_data[valid_mask]
            labels = labels[valid_mask]
            
            # 枢纽神经元选择
            print("进行枢纽神经元识别...")
            hub_indices, degree_values = identify_hub_neurons(neural_data, labels)
            neural_data_hub = neural_data[:, hub_indices, :]
            
            # 限制数据量
            max_trials = min(batch_config.MAX_TRIALS, neural_data_hub.shape[0])
            max_neurons = min(batch_config.MAX_NEURONS, neural_data_hub.shape[1])
            
            neural_data_hub = neural_data_hub[:max_trials, :max_neurons, :]
            labels = labels[:max_trials]
            
            print(f"数据加载成功!")
            print(f"原始数据维度: {neural_data.shape}")
            print(f"枢纽神经元数量: {len(hub_indices)}")
            print(f"分析数据维度: {neural_data_hub.shape}")
            print(f"标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
            
            # 运行协同信息分析
            print(f"开始协同信息打乱分析...")
            synergy_results = analyze_synergy_shuffle_effects(
                neural_data_hub, labels,
                shuffle_fractions=batch_config.SHUFFLE_FRACTIONS,
                n_iterations=batch_config.N_ITERATIONS
            )
            
            # 添加小鼠信息
            synergy_results['mouse_name'] = mouse_name
            synergy_results['data_info'] = {
                'original_shape': neural_data.shape,
                'hub_neurons': len(hub_indices),
                'analysis_shape': neural_data_hub.shape,
                'label_distribution': dict(zip(*np.unique(labels, return_counts=True))),
                'hub_indices': hub_indices,
                'degree_values': degree_values
            }
            
            print(f"{mouse_name}协同信息分析完成")
            return synergy_results
            
    except Exception as e:
        print(f"{mouse_name}分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_old_format_synergy(mouse_data_path, mouse_name):
    """分析旧版数据格式的协同信息"""
    try:
        from loaddata import load_old_version_data, fast_rr_selection
        from synergy_shuffle_analysis import analyze_synergy_shuffle_effects
        
        # 文件路径
        neurons_mat = os.path.join(mouse_data_path, 'Neurons.mat')
        trials_mat = os.path.join(mouse_data_path, 'Trial_data.mat')
        location_mat = os.path.join(mouse_data_path, 'wholebrain_output.mat')
        
        # 加载旧版数据
        neuron_index, segments_raw, labels_raw, location = load_old_version_data(
            neurons_mat, trials_mat, location_mat
        )
        
        # 转换为新版格式
        neural_data = np.array(segments_raw)
        labels = np.array(labels_raw)
        
        print(f"旧版数据加载完成！")
        print(f"数据维度: {neural_data.shape}")
        
        # 枢纽神经元选择
        print("进行枢纽神经元识别...")
        hub_indices, degree_values = identify_hub_neurons(neural_data, labels)
        neural_data_hub = neural_data[:, hub_indices, :]
        
        # 限制数据量
        max_trials = min(batch_config.MAX_TRIALS, neural_data_hub.shape[0])
        max_neurons = min(batch_config.MAX_NEURONS, neural_data_hub.shape[1])
        
        neural_data_hub = neural_data_hub[:max_trials, :max_neurons, :]
        labels = labels[:max_trials]
        
        print(f"分析数据维度: {neural_data_hub.shape}")
        print(f"标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        # 运行协同信息分析
        print(f"开始协同信息打乱分析...")
        synergy_results = analyze_synergy_shuffle_effects(
            neural_data_hub, labels,
            shuffle_fractions=batch_config.SHUFFLE_FRACTIONS,
            n_iterations=batch_config.N_ITERATIONS
        )
        
        # 添加小鼠信息
        synergy_results['mouse_name'] = mouse_name
        synergy_results['data_info'] = {
            'original_shape': neural_data.shape,
            'hub_neurons': len(hub_indices),
            'analysis_shape': neural_data_hub.shape,
            'label_distribution': dict(zip(*np.unique(labels, return_counts=True))),
            'hub_indices': hub_indices,
            'degree_values': degree_values
        }
        
        print(f"{mouse_name}协同信息分析完成")
        return synergy_results
        
    except Exception as e:
        print(f"{mouse_name}旧版数据分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_all_mice_synergy(all_results, save_path=None):
    """对比所有小鼠的协同信息结果"""
    setup_plot_style()
    
    # 创建2x2子图布局
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 定义颜色
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
    shuffle_fractions = all_results[0]['shuffle_fractions']
    
    mice_names = [result['mouse_name'] for result in all_results]
    
    # 1. 协同信息对比
    ax1 = axes[0, 0]
    for i, result in enumerate(all_results):
        mouse_name = result['mouse_name']
        synergies = result['mean_synergy']
        synergy_stds = result['std_synergy']
        
        ax1.errorbar(shuffle_fractions, synergies, yerr=synergy_stds,
                    marker='o', capsize=3, linewidth=2, markersize=6,
                    color=colors[i], alpha=0.8, label=f'{mouse_name}')
    
    ax1.set_xlabel('Shuffle Fraction')
    ax1.set_ylabel('Mean Synergy (bits)')
    ax1.set_title('Synergy Comparison Across Mice')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 冗余信息对比
    ax2 = axes[0, 1]
    for i, result in enumerate(all_results):
        mouse_name = result['mouse_name']
        redundancies = result['mean_redundancy']
        redundancy_stds = result['std_redundancy']
        
        ax2.errorbar(shuffle_fractions, redundancies, yerr=redundancy_stds,
                    marker='s', capsize=3, linewidth=2, markersize=6,
                    color=colors[i], alpha=0.8, label=f'{mouse_name}')
    
    ax2.set_xlabel('Shuffle Fraction')
    ax2.set_ylabel('Mean Redundancy (bits)')
    ax2.set_title('Redundancy Comparison Across Mice')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 唯一信息对比
    ax3 = axes[1, 0]
    for i, result in enumerate(all_results):
        mouse_name = result['mouse_name']
        uniques = result['mean_unique']
        unique_stds = result['std_unique']
        
        ax3.errorbar(shuffle_fractions, uniques, yerr=unique_stds,
                    marker='^', capsize=3, linewidth=2, markersize=6,
                    color=colors[i], alpha=0.8, label=f'{mouse_name}')
    
    ax3.set_xlabel('Shuffle Fraction')
    ax3.set_ylabel('Mean Unique Information (bits)')
    ax3.set_title('Unique Information Comparison Across Mice')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 归一化变化率对比
    ax4 = axes[1, 1]
    
    for i, result in enumerate(all_results):
        mouse_name = result['mouse_name']
        # 计算协同信息的相对变化
        synergies = np.array(result['mean_synergy'])
        if synergies[0] > 0:
            synergy_change = (synergies - synergies[0]) / synergies[0] * 100
            ax4.plot(shuffle_fractions, synergy_change, 
                    marker='o', linewidth=2, markersize=5, color=colors[i], alpha=0.8, 
                    label=f'{mouse_name} (Synergy)')
        
        # 计算冗余信息的相对变化
        redundancies = np.array(result['mean_redundancy'])
        if redundancies[0] > 0:
            redundancy_change = (redundancies - redundancies[0]) / redundancies[0] * 100
            ax4.plot(shuffle_fractions, redundancy_change, 
                    marker='s', linewidth=2, markersize=5, color=colors[i], alpha=0.6, 
                    linestyle='--', label=f'{mouse_name} (Redundancy)')
    
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Shuffle Fraction')
    ax4.set_ylabel('Change from Original (%)')
    ax4.set_title('Relative Change in Information')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Cross-Mouse Synergy Analysis Comparison', y=0.98, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=batch_config.DPI, bbox_inches='tight')
        print(f"跨鼠协同信息对比图已保存: {save_path}")
    
    plt.close()

def create_synergy_summary_statistics(all_results, save_path=None):
    """创建协同信息汇总统计图"""
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    mice_names = [result['mouse_name'] for result in all_results]
    
    # 提取基线值和完全打乱后的值
    baseline_synergy = [result['mean_synergy'][0] for result in all_results]
    final_synergy = [result['mean_synergy'][-1] for result in all_results]
    
    baseline_redundancy = [result['mean_redundancy'][0] for result in all_results]
    final_redundancy = [result['mean_redundancy'][-1] for result in all_results]
    
    baseline_unique = [result['mean_unique'][0] for result in all_results]
    final_unique = [result['mean_unique'][-1] for result in all_results]
    
    # 计算变化率
    synergy_changes = [(final - baseline) / baseline * 100 if baseline > 0 else 0 
                      for baseline, final in zip(baseline_synergy, final_synergy)]
    redundancy_changes = [(final - baseline) / baseline * 100 if baseline > 0 else 0 
                         for baseline, final in zip(baseline_redundancy, final_redundancy)]
    unique_changes = [(final - baseline) / baseline * 100 if baseline > 0 else 0 
                     for baseline, final in zip(baseline_unique, final_unique)]
    
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
    
    # 1. 基线协同信息对比
    ax1 = axes[0, 0]
    bars1 = ax1.bar(mice_names, baseline_synergy, color=colors, alpha=0.7)
    ax1.set_ylabel('Baseline Synergy (bits)')
    ax1.set_title('Baseline Synergy Comparison')
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars1, baseline_synergy):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 2. 基线冗余信息对比
    ax2 = axes[0, 1]
    bars2 = ax2.bar(mice_names, baseline_redundancy, color=colors, alpha=0.7)
    ax2.set_ylabel('Baseline Redundancy (bits)')
    ax2.set_title('Baseline Redundancy Comparison')
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars2, baseline_redundancy):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 3. 协同信息变化率对比
    ax3 = axes[1, 0]
    bars3 = ax3.bar(mice_names, synergy_changes, color=colors, alpha=0.7)
    ax3.set_ylabel('Synergy Change (%)')
    ax3.set_title('Synergy Change After Full Shuffling')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars3, synergy_changes):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, 
                height + (0.5 if height >= 0 else -1),
                f'{value:+.1f}%', ha='center', 
                va='bottom' if height >= 0 else 'top', fontsize=10)
    
    # 4. 冗余信息变化率对比
    ax4 = axes[1, 1]
    bars4 = ax4.bar(mice_names, redundancy_changes, color=colors, alpha=0.7)
    ax4.set_ylabel('Redundancy Change (%)')
    ax4.set_title('Redundancy Change After Full Shuffling')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars4, redundancy_changes):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, 
                height + (0.5 if height >= 0 else -1),
                f'{value:+.1f}%', ha='center', 
                va='bottom' if height >= 0 else 'top', fontsize=10)
    
    plt.suptitle('Synergy Analysis Summary: Cross-Mouse Statistics', y=0.98, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=batch_config.DPI, bbox_inches='tight')
        print(f"协同信息汇总统计图已保存: {save_path}")
    
    plt.close()

def run_batch_synergy_analysis():
    """运行所有小鼠的协同信息分析（基于枢纽神经元）"""
    print("=" * 80)
    print("批量协同信息打乱分析 - 所有小鼠（基于枢纽神经元）")
    print("=" * 80)
    
    # 小鼠数据路径
    mice_paths = {
        'M27': r'C:\Users\76629\OneDrive\brain\Micedata\M27_1008',
        'M30': r'C:\Users\76629\OneDrive\brain\Micedata\M30_0420', 
        'M65': r'C:\Users\76629\OneDrive\brain\Micedata\M65_0816',
        'M74': r'C:\Users\76629\OneDrive\brain\Micedata\M74_0816'
    }
    
    # 确保结果目录存在
    results_dir = batch_config.ensure_batch_results_dir()
    
    successful_results = []
    failed_analyses = []
    
    for mouse_name, data_path in mice_paths.items():
        print(f"\n开始分析{mouse_name}...")
        
        if os.path.exists(data_path):
            result = run_mouse_synergy_analysis(data_path, mouse_name)
            if result:
                successful_results.append(result)
                
                # 保存个体结果
                np.savez_compressed(
                    os.path.join(results_dir, f'{mouse_name.lower()}_synergy_results.npz'),
                    **result
                )
                
                # 生成个体可视化
                individual_save_path = os.path.join(results_dir, f'{mouse_name.lower()}_synergy_analysis.png')
                visualize_synergy_shuffle_results(
                    result,
                    title=f"{mouse_name} Synergy Analysis: Neural Shuffling Effects",
                    save_path=individual_save_path
                )
                
                print(f"[成功] {mouse_name}分析成功完成")
            else:
                failed_analyses.append(mouse_name)
                print(f"[失败] {mouse_name}分析失败")
        else:
            failed_analyses.append(f"{mouse_name} (路径不存在)")
            print(f"[失败] {mouse_name}数据路径不存在: {data_path}")
    
    # 生成跨鼠对比分析
    if len(successful_results) > 1:
        print(f"\n生成跨鼠对比分析...")
        
        # 跨鼠对比图
        compare_all_mice_synergy(
            successful_results,
            save_path=os.path.join(results_dir, 'cross_mouse_synergy_comparison.png')
        )
        
        # 汇总统计图
        create_synergy_summary_statistics(
            successful_results,
            save_path=os.path.join(results_dir, 'synergy_summary_statistics.png')
        )
        
        # 保存汇总结果
        summary_data = {
            'n_mice': len(successful_results),
            'mice_names': [r['mouse_name'] for r in successful_results],
            'all_results': successful_results
        }
        
        np.savez_compressed(
            os.path.join(results_dir, 'all_mice_synergy_summary.npz'),
            **summary_data
        )
    
    # 打印汇总报告
    print("\n" + "=" * 80)
    print("批量协同信息分析汇总结果（基于枢纽神经元）")
    print("=" * 80)
    print(f"成功分析: {len(successful_results)}只小鼠")
    for result in successful_results:
        mouse_name = result['mouse_name']
        baseline_synergy = result['mean_synergy'][0]
        final_synergy = result['mean_synergy'][-1]
        synergy_change = (final_synergy - baseline_synergy) / baseline_synergy * 100 if baseline_synergy > 0 else 0
        
        baseline_redundancy = result['mean_redundancy'][0]
        final_redundancy = result['mean_redundancy'][-1]
        redundancy_change = (final_redundancy - baseline_redundancy) / baseline_redundancy * 100 if baseline_redundancy > 0 else 0
        
        baseline_unique = result['mean_unique'][0]
        final_unique = result['mean_unique'][-1]
        unique_change = (final_unique - baseline_unique) / baseline_unique * 100 if baseline_unique > 0 else 0
        
        data_info = result['data_info']
        hub_count = data_info['hub_neurons']
        analysis_shape = data_info['analysis_shape']
        
        print(f"  [成功] {mouse_name}:")
        print(f"    数据维度: {analysis_shape}, 枢纽神经元: {hub_count}个")
        print(f"    基线协同信息: {baseline_synergy:.4f} bits, 打乱后变化: {synergy_change:+.1f}%")
        print(f"    基线冗余信息: {baseline_redundancy:.4f} bits, 打乱后变化: {redundancy_change:+.1f}%")
        print(f"    基线唯一信息: {baseline_unique:.4f} bits, 打乱后变化: {unique_change:+.1f}%")
    
    if failed_analyses:
        print(f"\n失败分析: {len(failed_analyses)}只小鼠")
        for failure in failed_analyses:
            print(f"  [失败] {failure}")
    
    print(f"\n批量协同信息分析完成！（基于枢纽神经元）")
    print(f"所有结果保存在: {results_dir}")
    
    return successful_results, failed_analyses

if __name__ == "__main__":
    successful, failed = run_batch_synergy_analysis()
    
    print("\n主要输出文件:")
    print("- cross_mouse_synergy_comparison.png: 跨鼠协同信息对比（枢纽神经元）")
    print("- synergy_summary_statistics.png: 协同信息汇总统计（枢纽神经元）")
    print("- *_synergy_analysis.png: 各小鼠个体分析图（枢纽神经元）")
    print("- *_synergy_results.npz: 各小鼠详细结果数据（包含枢纽神经元信息）")