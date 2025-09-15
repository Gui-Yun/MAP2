# 协同信息打乱分析 - 使用PID方法分析打乱对协同信息的影响
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict
import random
from scipy.stats import entropy

# 导入必要模块
from loaddata import cfg, load_data, segment_neuron_data, reclassify_labels, fast_rr_selection
from noise_correlation_analysis import shuffle_within_condition, NoiseCorrelationConfig

class SynergyShuffleConfig(NoiseCorrelationConfig):
    """协同信息打乱分析配置"""
    SHUFFLE_FRACTIONS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # 打乱比例
    N_ITERATIONS = 3  # 迭代次数
    MAX_TRIALS = 60  # 限制试次数
    MAX_NEURONS = 25  # 限制神经元数（PID计算较慢）
    N_NEURON_PAIRS = 50  # 随机选择的神经元对数量
    N_BINS = 4  # 离散化bin数量
    
    @classmethod
    def get_results_dir(cls):
        return os.path.join(cfg.get_results_dir(), 'synergy_shuffle_analysis')
    
    @classmethod
    def ensure_results_dir(cls):
        results_dir = cls.get_results_dir()
        os.makedirs(results_dir, exist_ok=True)
        return results_dir

config = SynergyShuffleConfig()

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

def discretize_data(data, n_bins=4):
    """
    将连续数据离散化为指定bin数量
    
    参数:
    data: 输入数据数组
    n_bins: 离散化的bin数量
    
    返回:
    discrete_data: 离散化后的数据
    """
    if len(data) == 0:
        return np.array([])
    
    # 使用等频率分箱
    try:
        percentiles = np.linspace(0, 100, n_bins + 1)
        thresholds = np.percentile(data, percentiles)
        # 处理重复值
        thresholds = np.unique(thresholds)
        if len(thresholds) < n_bins + 1:
            # 如果唯一值太少，使用等宽分箱
            thresholds = np.linspace(np.min(data), np.max(data), n_bins + 1)
        
        discrete_data = np.digitize(data, thresholds[:-1]) - 1
        discrete_data = np.clip(discrete_data, 0, n_bins - 1)
        return discrete_data.astype(int)
    except:
        # 如果分箱失败，返回全零数组
        return np.zeros(len(data), dtype=int)

def calculate_mutual_information_discrete(X, Y):
    """
    计算两个离散变量的互信息
    
    参数:
    X, Y: 离散化的数据数组
    
    返回:
    mi: 互信息值
    """
    # 检查输入有效性
    if len(X) == 0 or len(Y) == 0 or len(X) != len(Y):
        return 0.0
    
    # 确保数据为整数类型
    X = X.astype(int)
    Y = Y.astype(int)
    
    try:
        # 创建联合分布
        xy_joint = np.column_stack([X.ravel(), Y.ravel()])
        
        # 计算联合熵和边际熵
        H_X = entropy(np.bincount(X.ravel()) + 1e-10, base=2)
        H_Y = entropy(np.bincount(Y.ravel()) + 1e-10, base=2)
        
        # 计算联合熵
        unique_pairs, counts = np.unique(xy_joint, axis=0, return_counts=True)
        H_XY = entropy(counts + 1e-10, base=2)
        
        # 互信息 = H(X) + H(Y) - H(X,Y)
        mi = H_X + H_Y - H_XY
        return max(0, mi)  # 互信息不能为负
    except:
        return 0.0

def partial_information_decomposition(X1, X2, Y):
    """
    对三个变量进行部分信息分解（PID）
    
    参数:
    X1, X2: 源变量（神经元）
    Y: 目标变量（标签）
    
    返回:
    结果字典包含：synergy, redundancy, unique_1, unique_2
    """
    # 计算各种互信息
    I_X1_Y = calculate_mutual_information_discrete(X1, Y)
    I_X2_Y = calculate_mutual_information_discrete(X2, Y)
    
    # 创建X1X2的联合变量
    X1_max = int(np.max(X1)) + 1
    X1X2_joint = X1.astype(int) * X1_max + X2.astype(int)
    I_X1X2_Y = calculate_mutual_information_discrete(X1X2_joint, Y)
    
    # 简化的PID分解
    # 协同信息: 联合信息减去单独信息之和的正数部分
    synergy = max(0, I_X1X2_Y - I_X1_Y - I_X2_Y)
    
    # 冗余信息: 两个单独信息的最小值减去联合优势的一半
    joint_advantage = max(0, I_X1_Y + I_X2_Y - I_X1X2_Y)
    redundancy = min(I_X1_Y, I_X2_Y) - joint_advantage / 2
    redundancy = max(0, redundancy)
    
    # 唯一信息
    unique_1 = max(0, I_X1_Y - redundancy - synergy)
    unique_2 = max(0, I_X2_Y - redundancy - synergy)
    
    return {
        'synergy': synergy,
        'redundancy': redundancy, 
        'unique_1': unique_1,
        'unique_2': unique_2,
        'total_info': I_X1X2_Y,
        'individual_1': I_X1_Y,
        'individual_2': I_X2_Y
    }

def calculate_population_synergy(neural_data, labels, n_pairs=50, n_bins=4):
    """
    计算群体神经元的协同信息
    
    参数:
    neural_data: 神经数据 (n_trials, n_neurons, n_timepoints)
    labels: 标签数组
    n_pairs: 随机选择的神经元对数量
    n_bins: 离散化bin数量
    
    返回:
    synergy_stats: 协同信息统计结果
    """
    n_trials, n_neurons, n_timepoints = neural_data.shape
    
    # 将神经数据平均到试次-神经元矩阵
    neural_responses = np.mean(neural_data, axis=2)  # (n_trials, n_neurons)
    
    # 离散化标签
    discrete_labels = discretize_data(labels, n_bins)
    
    # 随机选择神经元对
    neuron_pairs = []
    if n_neurons >= 2:
        # 生成所有可能的神经元对
        all_pairs = [(i, j) for i in range(n_neurons) for j in range(i+1, n_neurons)]
        
        # 随机选择n_pairs个对
        if len(all_pairs) > n_pairs:
            selected_pairs = random.sample(all_pairs, n_pairs)
        else:
            selected_pairs = all_pairs
        
        neuron_pairs = selected_pairs
    
    if len(neuron_pairs) == 0:
        return {
            'mean_synergy': 0.0,
            'mean_redundancy': 0.0,
            'mean_unique': 0.0,
            'std_synergy': 0.0,
            'std_redundancy': 0.0,
            'std_unique': 0.0,
            'n_pairs': 0
        }
    
    # 计算每对神经元的PID
    synergies = []
    redundancies = []
    uniques = []
    
    for i, j in neuron_pairs:
        # 提取神经元响应并离散化
        neuron1_response = discretize_data(neural_responses[:, i], n_bins)
        neuron2_response = discretize_data(neural_responses[:, j], n_bins)
        
        # 计算PID
        pid_result = partial_information_decomposition(
            neuron1_response, neuron2_response, discrete_labels
        )
        
        synergies.append(pid_result['synergy'])
        redundancies.append(pid_result['redundancy'])
        uniques.append((pid_result['unique_1'] + pid_result['unique_2']) / 2)
    
    # 计算统计量
    synergies = np.array(synergies)
    redundancies = np.array(redundancies)
    uniques = np.array(uniques)
    
    return {
        'mean_synergy': np.mean(synergies) if len(synergies) > 0 else 0.0,
        'mean_redundancy': np.mean(redundancies) if len(redundancies) > 0 else 0.0,
        'mean_unique': np.mean(uniques) if len(uniques) > 0 else 0.0,
        'std_synergy': np.std(synergies) if len(synergies) > 1 else 0.0,
        'std_redundancy': np.std(redundancies) if len(redundancies) > 1 else 0.0,
        'std_unique': np.std(uniques) if len(uniques) > 1 else 0.0,
        'n_pairs': len(neuron_pairs),
        'raw_synergies': synergies,
        'raw_redundancies': redundancies,
        'raw_uniques': uniques
    }

def analyze_synergy_shuffle_effects(neural_data, labels, shuffle_fractions=None, n_iterations=3):
    """
    分析打乱对协同信息的影响
    
    参数:
    neural_data: 神经数据 (n_trials, n_neurons, n_timepoints)
    labels: 标签数组
    shuffle_fractions: 打乱比例列表
    n_iterations: 迭代次数
    
    返回:
    results: 分析结果字典
    """
    if shuffle_fractions is None:
        shuffle_fractions = config.SHUFFLE_FRACTIONS
    
    print(f"分析打乱对协同信息的影响...")
    print(f"- 数据维度: {neural_data.shape}")
    print(f"- 随机选择神经元对数量: {config.N_NEURON_PAIRS}")
    print(f"- 离散化bin数量: {config.N_BINS}")
    print(f"- 每个打乱比例重复{n_iterations}次取平均")
    
    results = {
        'shuffle_fractions': shuffle_fractions,
        'mean_synergy': [],
        'mean_redundancy': [],
        'mean_unique': [],
        'std_synergy': [],
        'std_redundancy': [], 
        'std_unique': [],
        'synergy_std': [],
        'redundancy_std': [],
        'unique_std': [],
        'n_iterations': n_iterations
    }
    
    for fraction in shuffle_fractions:
        print(f"\n打乱比例: {fraction:.1f}")
        
        # 存储每次迭代的结果
        iteration_synergies = []
        iteration_redundancies = []
        iteration_uniques = []
        
        for iteration in range(n_iterations):
            print(f"  迭代 {iteration + 1}/{n_iterations}")
            
            # 打乱数据
            if fraction == 0.0:
                shuffled_data = neural_data.copy()
            else:
                shuffled_data = shuffle_within_condition(neural_data, labels, fraction)
            
            # 计算协同信息
            synergy_stats = calculate_population_synergy(
                shuffled_data, labels, 
                n_pairs=config.N_NEURON_PAIRS,
                n_bins=config.N_BINS
            )
            
            iteration_synergies.append(synergy_stats['mean_synergy'])
            iteration_redundancies.append(synergy_stats['mean_redundancy'])
            iteration_uniques.append(synergy_stats['mean_unique'])
        
        # 计算跨迭代的统计量
        results['mean_synergy'].append(np.mean(iteration_synergies))
        results['mean_redundancy'].append(np.mean(iteration_redundancies))
        results['mean_unique'].append(np.mean(iteration_uniques))
        
        results['std_synergy'].append(np.std(iteration_synergies) if len(iteration_synergies) > 1 else 0.0)
        results['std_redundancy'].append(np.std(iteration_redundancies) if len(iteration_redundancies) > 1 else 0.0)
        results['std_unique'].append(np.std(iteration_uniques) if len(iteration_uniques) > 1 else 0.0)
        
        # 为了兼容性，重复存储标准差
        results['synergy_std'].append(results['std_synergy'][-1])
        results['redundancy_std'].append(results['std_redundancy'][-1])
        results['unique_std'].append(results['std_unique'][-1])
        
        print(f"  协同信息: {results['mean_synergy'][-1]:.4f} ± {results['std_synergy'][-1]:.4f}")
        print(f"  冗余信息: {results['mean_redundancy'][-1]:.4f} ± {results['std_redundancy'][-1]:.4f}")
        print(f"  唯一信息: {results['mean_unique'][-1]:.4f} ± {results['std_unique'][-1]:.4f}")
    
    return results

def visualize_synergy_shuffle_results(results, title="Synergy Changes vs Neural Shuffling", save_path=None):
    """
    可视化协同信息打乱分析结果
    
    参数:
    results: 分析结果字典
    title: 图表标题
    save_path: 保存路径
    """
    setup_plot_style()
    
    shuffle_fractions = results['shuffle_fractions']
    
    # 创建2x2子图布局
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 定义颜色
    color_synergy = '#E74C3C'    # 红色 - 协同信息
    color_redundancy = '#3498DB' # 蓝色 - 冗余信息
    color_unique = '#2ECC71'     # 绿色 - 唯一信息
    
    # 1. 协同信息变化
    ax1 = axes[0, 0]
    synergies = results['mean_synergy']
    synergy_stds = results['std_synergy']
    
    ax1.errorbar(shuffle_fractions, synergies, yerr=synergy_stds,
                marker='o', capsize=5, linewidth=3, markersize=8,
                color=color_synergy, alpha=0.8, label='Synergy')
    
    ax1.fill_between(shuffle_fractions, 
                    np.array(synergies) - np.array(synergy_stds),
                    np.array(synergies) + np.array(synergy_stds),
                    alpha=0.2, color=color_synergy)
    
    ax1.set_xlabel('Shuffle Fraction')
    ax1.set_ylabel('Mean Synergy (bits)')
    ax1.set_title('Synergy vs Shuffle Fraction')
    ax1.grid(True, alpha=0.3)
    
    # 2. 冗余信息变化
    ax2 = axes[0, 1]
    redundancies = results['mean_redundancy']
    redundancy_stds = results['std_redundancy']
    
    ax2.errorbar(shuffle_fractions, redundancies, yerr=redundancy_stds,
                marker='s', capsize=5, linewidth=3, markersize=8,
                color=color_redundancy, alpha=0.8, label='Redundancy')
    
    ax2.fill_between(shuffle_fractions, 
                    np.array(redundancies) - np.array(redundancy_stds),
                    np.array(redundancies) + np.array(redundancy_stds),
                    alpha=0.2, color=color_redundancy)
    
    ax2.set_xlabel('Shuffle Fraction')
    ax2.set_ylabel('Mean Redundancy (bits)')
    ax2.set_title('Redundancy vs Shuffle Fraction')
    ax2.grid(True, alpha=0.3)
    
    # 3. 唯一信息变化
    ax3 = axes[1, 0]
    uniques = results['mean_unique']
    unique_stds = results['std_unique']
    
    ax3.errorbar(shuffle_fractions, uniques, yerr=unique_stds,
                marker='^', capsize=5, linewidth=3, markersize=8,
                color=color_unique, alpha=0.8, label='Unique Information')
    
    ax3.fill_between(shuffle_fractions, 
                    np.array(uniques) - np.array(unique_stds),
                    np.array(uniques) + np.array(unique_stds),
                    alpha=0.2, color=color_unique)
    
    ax3.set_xlabel('Shuffle Fraction')
    ax3.set_ylabel('Mean Unique Information (bits)')
    ax3.set_title('Unique Information vs Shuffle Fraction')
    ax3.grid(True, alpha=0.3)
    
    # 4. 综合对比和归一化变化率
    ax4 = axes[1, 1]
    
    # 绘制所有信息类型
    ax4.plot(shuffle_fractions, synergies, 
            marker='o', linewidth=3, markersize=6, color=color_synergy, alpha=0.8, 
            label='Synergy')
    ax4.plot(shuffle_fractions, redundancies, 
            marker='s', linewidth=3, markersize=6, color=color_redundancy, alpha=0.8,
            label='Redundancy')
    ax4.plot(shuffle_fractions, uniques,
            marker='^', linewidth=3, markersize=6, color=color_unique, alpha=0.8,
            label='Unique Information')
    
    ax4.set_xlabel('Shuffle Fraction')
    ax4.set_ylabel('Information (bits)')
    ax4.set_title('Information Components Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, y=0.98, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"协同信息分析图已保存: {save_path}")
    
    plt.close()

def run_synergy_shuffle_analysis(data_path=None, mouse_name="M65"):
    """
    运行协同信息打乱分析
    
    参数:
    data_path: 数据路径
    mouse_name: 小鼠名称
    
    返回:
    results: 分析结果
    """
    print("=" * 70)
    print(f"协同信息打乱分析 - {mouse_name}数据")
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
        
        # 限制数据量以加快分析
        max_trials = min(config.MAX_TRIALS, neural_data_rr.shape[0])
        max_neurons = min(config.MAX_NEURONS, neural_data_rr.shape[1])
        
        neural_data_rr = neural_data_rr[:max_trials, :max_neurons, :]
        labels = labels[:max_trials]
        
        print(f"数据加载成功!")
        print(f"原始数据维度: {neural_data.shape}")
        print(f"RR神经元数量: {len(rr_indices)}")
        print(f"分析数据维度: {neural_data_rr.shape}")
        print(f"标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        # 运行协同信息打乱分析
        print(f"\n开始协同信息打乱分析...")
        synergy_results = analyze_synergy_shuffle_effects(
            neural_data_rr, labels,
            shuffle_fractions=config.SHUFFLE_FRACTIONS,
            n_iterations=config.N_ITERATIONS
        )
        
        # 保存结果
        print(f"\n保存分析结果...")
        save_data = {
            'mouse_name': mouse_name,
            **synergy_results
        }
        
        np.savez_compressed(
            os.path.join(results_dir, f'{mouse_name.lower()}_synergy_shuffle_results.npz'),
            **save_data
        )
        print("协同信息打乱分析结果已保存")
        
        # 可视化
        print(f"\n生成可视化图表...")
        visualize_synergy_shuffle_results(
            synergy_results,
            title=f"{mouse_name} Synergy Analysis: Neural Shuffling Effects",
            save_path=os.path.join(results_dir, f'{mouse_name.lower()}_synergy_shuffle_analysis.png')
        )
        
        # 打印汇总结果
        print(f"\n{'='*70}")
        print(f"协同信息打乱分析完成！")
        print(f"{'='*70}")
        
        baseline_synergy = synergy_results['mean_synergy'][0]
        final_synergy = synergy_results['mean_synergy'][-1]
        synergy_change = (final_synergy - baseline_synergy) / baseline_synergy * 100 if baseline_synergy > 0 else 0
        
        baseline_redundancy = synergy_results['mean_redundancy'][0]
        final_redundancy = synergy_results['mean_redundancy'][-1]
        redundancy_change = (final_redundancy - baseline_redundancy) / baseline_redundancy * 100 if baseline_redundancy > 0 else 0
        
        baseline_unique = synergy_results['mean_unique'][0]
        final_unique = synergy_results['mean_unique'][-1]
        unique_change = (final_unique - baseline_unique) / baseline_unique * 100 if baseline_unique > 0 else 0
        
        print(f"基线协同信息: {baseline_synergy:.4f} bits")
        print(f"完全打乱后协同信息: {final_synergy:.4f} bits (变化: {synergy_change:+.1f}%)")
        print(f"基线冗余信息: {baseline_redundancy:.4f} bits")
        print(f"完全打乱后冗余信息: {final_redundancy:.4f} bits (变化: {redundancy_change:+.1f}%)")
        print(f"基线唯一信息: {baseline_unique:.4f} bits")
        print(f"完全打乱后唯一信息: {final_unique:.4f} bits (变化: {unique_change:+.1f}%)")
        
        print(f"\n结果保存在: {results_dir}")
        return synergy_results
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # 运行M65的协同信息打乱分析
    results = run_synergy_shuffle_analysis(mouse_name="M65")
    
    if results:
        print("\n协同信息打乱分析完成！")
        results_dir = config.get_results_dir()
        print("主要结果文件:")
        print(f"- 协同信息分析图: {results_dir}/m65_synergy_shuffle_analysis.png")
        print(f"- 分析数据: {results_dir}/m65_synergy_shuffle_results.npz")