# 批量网络效率分析脚本 - 所有小鼠数据
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
import json
from collections import defaultdict

# 导入必要模块
from loaddata import load_data, segment_neuron_data, reclassify_labels, fast_rr_selection
from noise_correlation_analysis import (
    NoiseCorrelationConfig, calculate_noise_correlation, 
    shuffle_within_condition, build_networks_from_correlations
)

class BatchNetworkEfficiencyConfig(NoiseCorrelationConfig):
    """批量网络效率分析配置"""
    SHUFFLE_FRACTIONS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # 简化的测试点
    N_ITERATIONS = 2  # 减少迭代次数以加快分析
    MAX_TRIALS = 80  # 限制最大试次数
    MAX_NEURONS = 30  # 限制最大神经元数
    
    @classmethod
    def get_batch_results_dir(cls):
        from loaddata import cfg
        return os.path.join(cfg.get_results_dir(), 'batch_network_efficiency')
    
    @classmethod
    def ensure_batch_results_dir(cls):
        results_dir = cls.get_batch_results_dir()
        os.makedirs(results_dir, exist_ok=True)
        return results_dir

config = BatchNetworkEfficiencyConfig()

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

def analyze_mouse_network_efficiency(mouse_data_path, mouse_name):
    """分析单只小鼠的网络效率"""
    print(f"\\n{'='*50}")
    print(f"分析{mouse_name}小鼠数据")
    print(f"{'='*50}")
    
    try:
        # 检测数据格式并加载数据
        print("加载数据...")
        
        # 检查是否为旧版数据格式
        neurons_mat = os.path.join(mouse_data_path, 'Neurons.mat')
        trials_mat = os.path.join(mouse_data_path, 'Trial_data.mat')
        location_mat = os.path.join(mouse_data_path, 'wholebrain_output.mat')
        
        if os.path.exists(neurons_mat) and os.path.exists(trials_mat):
            # 使用旧版数据加载方法
            print(f"检测到旧版数据格式，使用旧版加载方法")
            from loaddata import load_old_version_data
            
            neuron_index, segments_raw, labels_raw, location = load_old_version_data(
                neurons_mat, trials_mat, location_mat
            )
            
            # 转换为新版格式
            neural_data = np.array(segments_raw)
            labels = np.array(labels_raw)
            neuron_pos = location[0:2, :]  # 提取前两维坐标
            
            print(f"旧版数据加载完成！")
            print(f"数据维度: {neural_data.shape}")
            
        else:
            # 使用新版数据加载方法
            print(f"使用新版数据加载方法")
            neural_data_raw, neuron_pos, start_edges, stimulus_data = load_data(mouse_data_path)
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
        
        # 限制数据量
        max_trials = min(config.MAX_TRIALS, neural_data_rr.shape[0])
        max_neurons = min(config.MAX_NEURONS, neural_data_rr.shape[1])
        
        neural_data_rr = neural_data_rr[:max_trials, :max_neurons, :]
        labels = labels[:max_trials]
        
        print(f"数据加载成功!")
        print(f"原始数据维度: {neural_data.shape}")
        print(f"RR神经元数量: {len(rr_indices)}")
        print(f"分析数据维度: {neural_data_rr.shape}")
        print(f"标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        # 分析网络效率
        results = {
            'mouse_name': mouse_name,
            'shuffle_fractions': config.SHUFFLE_FRACTIONS,
            'global_efficiency': [],
            'local_efficiency': [],
            'clustering_coefficient': [],
            'global_efficiency_std': [],
            'local_efficiency_std': [],
            'clustering_coefficient_std': [],
            'n_iterations': config.N_ITERATIONS,
            'data_info': {
                'original_shape': neural_data.shape,
                'rr_neurons': len(rr_indices),
                'analysis_shape': neural_data_rr.shape,
                'label_distribution': dict(zip(*np.unique(labels, return_counts=True)))
            }
        }
        
        print("\\n开始网络效率打乱分析...")
        for fraction in config.SHUFFLE_FRACTIONS:
            print(f"  打乱比例: {fraction:.1f}")
            
            # 存储每次迭代的结果
            global_effs = []
            local_effs = []
            clusterings = []
            
            for iteration in range(config.N_ITERATIONS):
                # 打乱数据
                if fraction == 0.0:
                    shuffled_data = neural_data_rr.copy()
                else:
                    shuffled_data = shuffle_within_condition(neural_data_rr, labels, fraction)
                
                # 展开数据计算相关性矩阵（回退方法）
                n_trials, n_neurons, n_timepoints = shuffled_data.shape
                
                # 展开数据: (n_neurons, n_trials * n_timepoints)
                flattened_data = shuffled_data.transpose(1, 0, 2).reshape(n_neurons, -1)
                
                # 计算相关性矩阵
                avg_corr_matrix = np.corrcoef(flattened_data)
                
                # 处理NaN值
                avg_corr_matrix = np.nan_to_num(avg_corr_matrix, nan=0.0)
                
                # 构建网络：只使用正相关值
                pos_corr = np.copy(avg_corr_matrix)
                pos_corr[pos_corr < 0] = 0  # 将负相关设为0
                np.fill_diagonal(pos_corr, 0)  # 对角线设为0
                
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
        
    except Exception as e:
        print(f"{mouse_name}数据分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_mouse_data_paths():
    """获取所有小鼠的数据路径"""
    base_data_path = r'C:\Users\76629\OneDrive\brain\Micedata'
    config_dir = 'config'
    
    mouse_paths = {}
    
    # 读取配置文件获取路径
    config_files = ['m27.json', 'm30.json', 'm65.json', 'm74.json']
    
    for config_file in config_files:
        config_path = os.path.join(config_dir, config_file)
        mouse_name = config_file.replace('.json', '').upper()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 从配置文件获取数据路径，或使用默认路径
            if 'DATA_PATH' in config_data:
                data_path = config_data['DATA_PATH']
            else:
                # 默认路径模式
                data_path = os.path.join(base_data_path, f'{mouse_name}_*')
                # 查找实际存在的路径
                import glob
                possible_paths = glob.glob(data_path)
                if possible_paths:
                    data_path = possible_paths[0]  # 取第一个匹配的路径
                else:
                    continue  # 跳过不存在的路径
            
            mouse_paths[mouse_name] = data_path
            print(f"找到{mouse_name}数据路径: {data_path}")
            
        except Exception as e:
            print(f"读取{mouse_name}配置失败: {e}")
            continue
    
    return mouse_paths

def batch_analyze_all_mice():
    """批量分析所有小鼠数据"""
    print("="*60)
    print("批量网络效率分析 - 所有小鼠数据")
    print("="*60)
    
    # 确保结果目录存在
    results_dir = config.ensure_batch_results_dir()
    
    # 获取所有小鼠数据路径
    mouse_paths = get_mouse_data_paths()
    print(f"\\n找到{len(mouse_paths)}只小鼠的数据")
    
    all_results = {}
    successful_analyses = []
    
    # 分析每只小鼠
    for mouse_name, data_path in mouse_paths.items():
        print(f"\\n开始分析{mouse_name}...")
        
        try:
            results = analyze_mouse_network_efficiency(data_path, mouse_name)
            if results is not None:
                all_results[mouse_name] = results
                successful_analyses.append(mouse_name)
                
                # 保存单个小鼠的结果
                mouse_results_file = os.path.join(results_dir, f'{mouse_name.lower()}_network_efficiency_results.npz')
                np.savez_compressed(
                    mouse_results_file,
                    **{k: v for k, v in results.items() if k != 'data_info'}
                )
                print(f"{mouse_name}结果已保存")
            else:
                print(f"{mouse_name}分析失败")
                
        except Exception as e:
            print(f"{mouse_name}分析出现异常: {e}")
            continue
    
    print(f"\\n{'='*60}")
    print(f"批量分析完成！成功分析{len(successful_analyses)}只小鼠: {successful_analyses}")
    print(f"结果保存在: {results_dir}")
    
    return all_results, results_dir

if __name__ == "__main__":
    all_results, results_dir = batch_analyze_all_mice()
    
    print("\\n批量网络效率分析完成！")
    print(f"结果目录: {results_dir}")
    if all_results:
        print("成功分析的小鼠:")
        for mouse_name in all_results.keys():
            print(f"  - {mouse_name}")