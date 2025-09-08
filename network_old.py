# 旧版数据的功能连接网络分析
# guiy24@mails.tsinghua.edu.cn
# date: 25-09-04
# 专门处理旧版数据格式，标签映射：1->类别1, 2->类别2, 3->类别1强度1

# %% 导入必要的库
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置为非交互式后端，防止弹出窗口
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
from scipy.stats import pearsonr, spearmanr
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
from collections import Counter
import os

# 从loaddata.py导入函数
from loaddata import (
    load_data, segment_neuron_data, reclassify_labels, 
    fast_rr_selection, cfg
)

# 从network.py导入配置和函数
from network import (
    NetworkConfig, net_cfg, print_network_config, get_method_recommendations,
    set_fast_mode, compute_correlation_matrix, threshold_correlation_matrix,
    compute_network_metrics, setup_network_plot_style,
    visualize_correlation_matrix, visualize_adjacency_matrix,
    visualize_connectivity_distribution, visualize_network_topology,
    visualize_network_metrics_comparison, visualize_degree_distribution
)

# %% 旧版数据专用配置
class OldDataNetworkConfig(NetworkConfig):
    """旧版数据网络分析配置"""
    
    # 旧版数据标签映射
    OLD_LABEL_MAPPING = {
        1: {'category': 1, 'intensity': 0, 'description': '类别1'},
        2: {'category': 2, 'intensity': 0, 'description': '类别2'}, 
        3: {'category': 1, 'intensity': 1, 'description': '类别1强度1'}
    }
    
    # 分析的标签（旧版只有1,2,3）
    TARGET_LABELS = [1, 2, 3]
    
    # 类别映射后的分析目标
    TARGET_CATEGORIES_MAPPED = [1, 2]  # 映射后的类别1和2
    
    def get_label_description(self, label):
        """获取标签描述"""
        if label in self.OLD_LABEL_MAPPING:
            return self.OLD_LABEL_MAPPING[label]['description']
        return f"未知标签{label}"
    
    def map_old_to_new_format(self, old_labels):
        """将旧版标签映射到新版格式"""
        new_categories = []
        new_intensities = []
        
        for label in old_labels:
            if label in self.OLD_LABEL_MAPPING:
                mapping = self.OLD_LABEL_MAPPING[label]
                new_categories.append(mapping['category'])
                new_intensities.append(mapping['intensity'])
            else:
                # 未知标签默认处理
                new_categories.append(0)
                new_intensities.append(0)
        
        return np.array(new_categories), np.array(new_intensities)

# 旧版数据配置实例
old_net_cfg = OldDataNetworkConfig()

def print_old_data_config():
    """打印旧版数据配置信息"""
    print("=" * 60)
    print("旧版数据网络分析配置:")
    print("=" * 60)
    print("标签映射关系:")
    for old_label, mapping in old_net_cfg.OLD_LABEL_MAPPING.items():
        print(f"  标签 {old_label} -> {mapping['description']} (类别{mapping['category']}, 强度{mapping['intensity']})")
    
    print(f"\n分析目标标签: {old_net_cfg.TARGET_LABELS}")
    print(f"映射后目标类别: {old_net_cfg.TARGET_CATEGORIES_MAPPED}")
    print("=" * 60)
    
    # 继承原有配置显示
    print_network_config()

# %% 旧版数据处理函数

def filter_old_data_by_labels(segments, labels, target_labels=[1, 2, 3]):
    """
    根据旧版标签过滤数据
    
    参数:
    segments: 神经数据片段 (trials, neurons, timepoints)
    labels: 旧版标签数组 
    target_labels: 目标标签列表
    
    返回:
    filtered_segments: 过滤后的数据
    filtered_labels: 过滤后的标签
    """
    print(f"过滤旧版数据标签 {target_labels}...")
    
    # 创建掩码
    mask = np.isin(labels, target_labels)
    
    filtered_segments = segments[mask]
    filtered_labels = labels[mask]
    
    print(f"原始数据: {len(labels)} 个试次")
    print(f"过滤后数据: {len(filtered_labels)} 个试次")
    print(f"标签分布: {Counter(filtered_labels)}")
    
    # 显示映射信息
    for label in np.unique(filtered_labels):
        if label in old_net_cfg.OLD_LABEL_MAPPING:
            desc = old_net_cfg.get_label_description(label)
            count = np.sum(filtered_labels == label)
            print(f"  标签 {label}: {count} 个试次 ({desc})")
    
    return filtered_segments, filtered_labels

def analyze_old_data_by_original_labels(network_segments, labels, neuron_pos=None):
    """
    按原始旧版标签分析网络（不进行映射）
    
    参数:
    network_segments: 神经数据片段 (trials, neurons, timepoints)
    labels: 旧版标签数组
    neuron_pos: 神经元位置坐标
    
    返回:
    label_networks: 各标签的网络数据
    label_metrics: 各标签的网络指标
    """
    print("=== 按旧版原始标签分析网络 ===")
    
    label_networks = {}
    label_metrics = {}
    
    # 过滤目标标签的数据
    filtered_segments, filtered_labels = filter_old_data_by_labels(
        network_segments, labels, old_net_cfg.TARGET_LABELS)
    
    # 分析每个标签
    for label in old_net_cfg.TARGET_LABELS:
        print(f"\n--- 处理标签 {label} ({old_net_cfg.get_label_description(label)}) ---")
        
        # 提取该标签的数据
        label_mask = filtered_labels == label
        label_segments = filtered_segments[label_mask]
        
        if len(label_segments) == 0:
            print(f"标签 {label} 无数据，跳过")
            continue
        
        print(f"标签 {label} 数据: {len(label_segments)} 个试次")
        
        # 提取神经活动（使用刺激期）
        neural_activity = extract_neural_activity_old(label_segments, use_stimulus_period=True)
        
        # 计算相关系数矩阵
        corr_matrix, p_matrix = compute_correlation_matrix(
            neural_activity, method=old_net_cfg.CORRELATION_METHOD)
        
        # 阈值化处理
        adj_matrix = threshold_correlation_matrix(corr_matrix, p_matrix)
        
        # 计算网络指标
        metrics = compute_network_metrics(adj_matrix)
        
        # 保存结果
        label_networks[label] = {
            'correlation_matrix': corr_matrix,
            'p_matrix': p_matrix,
            'adjacency_matrix': adj_matrix,
            'neural_activity': neural_activity,
            'description': old_net_cfg.get_label_description(label)
        }
        label_metrics[label] = metrics
        
        # 打印关键指标
        print(f"标签 {label} 网络指标:")
        print(f"  节点数: {metrics['n_nodes']}")
        print(f"  连接数: {metrics['n_edges']}")
        print(f"  密度: {metrics['density']:.3f}")
        print(f"  平均度: {metrics['avg_degree']:.2f}")
        print(f"  平均聚类系数: {metrics['avg_clustering']:.3f}")
        print(f"  模块化: {metrics['modularity']:.3f}")
        if metrics['is_connected']:
            print(f"  平均路径长度: {metrics['avg_path_length']:.3f}")
        print(f"  小世界系数: {metrics['small_world_sigma']:.3f}")
    
    return label_networks, label_metrics

def analyze_old_data_by_mapped_categories(network_segments, labels, neuron_pos=None):
    """
    按映射后的类别强度分析网络
    
    参数:
    network_segments: 神经数据片段 (trials, neurons, timepoints)
    labels: 旧版标签数组
    neuron_pos: 神经元位置坐标
    
    返回:
    condition_networks: 各条件的网络数据
    condition_metrics: 各条件的网络指标
    """
    print("=== 按映射后类别强度分析网络 ===")
    
    # 映射标签到新格式
    mapped_categories, mapped_intensities = old_net_cfg.map_old_to_new_format(labels)
    
    # 创建条件数据
    condition_networks = {}
    condition_metrics = {}
    
    # 获取唯一的类别强度组合
    unique_conditions = []
    for i, label in enumerate(labels):
        if label in old_net_cfg.OLD_LABEL_MAPPING:
            mapping = old_net_cfg.OLD_LABEL_MAPPING[label]
            condition_key = f"Cat{mapping['category']}_Int{mapping['intensity']}"
            unique_conditions.append((label, condition_key, mapping))
    
    # 去重并按原始标签排序
    seen_conditions = {}
    for original_label, condition_key, mapping in unique_conditions:
        if condition_key not in seen_conditions:
            seen_conditions[condition_key] = (original_label, mapping)
    
    # 分析每个映射条件
    for condition_key, (original_label, mapping) in seen_conditions.items():
        print(f"\n--- 处理条件: {condition_key} (原标签{original_label}: {mapping['description']}) ---")
        
        # 提取该条件的数据
        condition_mask = labels == original_label
        condition_segments = network_segments[condition_mask]
        
        if len(condition_segments) == 0:
            print(f"条件 {condition_key} 无数据，跳过")
            continue
        
        print(f"条件 {condition_key} 数据: {len(condition_segments)} 个试次")
        
        # 提取神经活动（使用刺激期）
        neural_activity = extract_neural_activity_old(condition_segments, use_stimulus_period=True)
        
        # 计算相关系数矩阵
        corr_matrix, p_matrix = compute_correlation_matrix(
            neural_activity, method=old_net_cfg.CORRELATION_METHOD)
        
        # 阈值化处理
        adj_matrix = threshold_correlation_matrix(corr_matrix, p_matrix)
        
        # 计算网络指标
        metrics = compute_network_metrics(adj_matrix)
        
        # 保存结果
        condition_networks[condition_key] = {
            'correlation_matrix': corr_matrix,
            'p_matrix': p_matrix,
            'adjacency_matrix': adj_matrix,
            'neural_activity': neural_activity,
            'original_label': original_label,
            'category': mapping['category'],
            'intensity': mapping['intensity'],
            'description': mapping['description']
        }
        condition_metrics[condition_key] = metrics
        
        # 打印关键指标
        print(f"条件 {condition_key} 网络指标:")
        print(f"  节点数: {metrics['n_nodes']}")
        print(f"  连接数: {metrics['n_edges']}")
        print(f"  密度: {metrics['density']:.3f}")
        print(f"  平均度: {metrics['avg_degree']:.2f}")
        print(f"  平均聚类系数: {metrics['avg_clustering']:.3f}")
        print(f"  模块化: {metrics['modularity']:.3f}")
        if metrics['is_connected']:
            print(f"  平均路径长度: {metrics['avg_path_length']:.3f}")
        print(f"  小世界系数: {metrics['small_world_sigma']:.3f}")
    
    return condition_networks, condition_metrics

def extract_neural_activity_old(segments, use_stimulus_period=True):
    """
    从旧版数据提取神经活动
    
    参数:
    segments: 神经数据片段 (trials, neurons, timepoints)
    use_stimulus_period: 是否只使用刺激期数据
    
    返回:
    neural_activity: (trials, neurons) 每个试次每个神经元的平均活动
    """
    print("提取神经活动（旧版数据）...")
    
    if use_stimulus_period:
        # 使用刺激期数据
        stimulus_start = cfg.PRE_FRAMES
        stimulus_end = cfg.PRE_FRAMES + cfg.STIMULUS_DURATION
        time_window = np.arange(stimulus_start, min(stimulus_end, segments.shape[2]))
        segments_subset = segments[:, :, time_window]
        print(f"使用刺激期数据，时间窗口: {stimulus_start}-{stimulus_end}")
    else:
        segments_subset = segments
        print("使用完整时间序列数据")
    
    # 计算每个试次每个神经元的平均活动
    neural_activity = np.mean(segments_subset, axis=2)  # (trials, neurons)
    
    print(f"神经活动矩阵形状: {neural_activity.shape}")
    return neural_activity

def analyze_old_data_differences(label_networks, condition_networks):
    """
    分析原始标签与映射条件的差异
    
    参数:
    label_networks: 按原始标签分析的网络数据
    condition_networks: 按映射条件分析的网络数据
    
    返回:
    comparison_results: 比较结果
    """
    print("=== 分析原始标签与映射条件的差异 ===")
    
    comparison_results = {}
    
    # 比较指标
    metrics_to_compare = ['density', 'avg_clustering', 'modularity', 'avg_degree']
    
    # 创建比较表
    comparison_data = []
    
    print("\n原始标签网络指标:")
    for label in sorted(label_networks.keys()):
        metrics = compute_network_metrics(label_networks[label]['adjacency_matrix'])
        row = {'type': 'original', 'identifier': f'Label_{label}', 
               'description': old_net_cfg.get_label_description(label)}
        for metric in metrics_to_compare:
            row[metric] = metrics[metric]
        comparison_data.append(row)
        
        print(f"标签 {label} ({old_net_cfg.get_label_description(label)}):")
        for metric in metrics_to_compare:
            print(f"  {metric}: {metrics[metric]:.3f}")
    
    print("\n映射条件网络指标:")
    for condition in sorted(condition_networks.keys()):
        metrics = compute_network_metrics(condition_networks[condition]['adjacency_matrix'])
        row = {'type': 'mapped', 'identifier': condition,
               'description': condition_networks[condition]['description']}
        for metric in metrics_to_compare:
            row[metric] = metrics[metric]
        comparison_data.append(row)
        
        print(f"{condition} ({condition_networks[condition]['description']}):")
        for metric in metrics_to_compare:
            print(f"  {metric}: {metrics[metric]:.3f}")
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_results['comparison_table'] = comparison_df
    comparison_results['metrics_compared'] = metrics_to_compare
    
    return comparison_results

def save_old_data_analysis_results(label_networks, condition_networks, comparison_results, 
                                 results_dir, save_visualizations=True):
    """
    保存旧版数据分析结果
    
    参数:
    label_networks: 按原始标签的网络数据
    condition_networks: 按映射条件的网络数据  
    comparison_results: 比较结果
    results_dir: 保存目录
    save_visualizations: 是否保存可视化图片
    """
    print("=== 保存旧版数据分析结果 ===")
    
    os.makedirs(results_dir, exist_ok=True)
    old_data_dir = os.path.join(results_dir, 'old_data_analysis')
    os.makedirs(old_data_dir, exist_ok=True)
    
    # 1. 保存网络矩阵数据
    print("保存网络矩阵数据...")
    save_dict = {}
    
    # 原始标签网络
    for label, network_data in label_networks.items():
        prefix = f'label_{label}'
        save_dict[f'{prefix}_correlation'] = network_data['correlation_matrix']
        save_dict[f'{prefix}_adjacency'] = network_data['adjacency_matrix']
        save_dict[f'{prefix}_pvalues'] = network_data['p_matrix']
    
    # 映射条件网络
    for condition, network_data in condition_networks.items():
        prefix = condition.lower()
        save_dict[f'{prefix}_correlation'] = network_data['correlation_matrix']
        save_dict[f'{prefix}_adjacency'] = network_data['adjacency_matrix']
        save_dict[f'{prefix}_pvalues'] = network_data['p_matrix']
    
    networks_path = os.path.join(old_data_dir, 'old_data_network_matrices.npz')
    np.savez_compressed(networks_path, **save_dict)
    print(f"网络矩阵已保存: {networks_path}")
    
    # 2. 保存比较结果
    print("保存比较结果...")
    comparison_path = os.path.join(old_data_dir, 'label_condition_comparison.csv')
    comparison_results['comparison_table'].to_csv(comparison_path, index=False)
    print(f"比较结果已保存: {comparison_path}")
    
    # 3. 保存详细指标
    print("保存详细指标...")
    detailed_metrics = {}
    
    # 原始标签指标
    for label, network_data in label_networks.items():
        metrics = compute_network_metrics(network_data['adjacency_matrix'])
        detailed_metrics[f'label_{label}'] = metrics
    
    # 映射条件指标
    for condition, network_data in condition_networks.items():
        metrics = compute_network_metrics(network_data['adjacency_matrix'])
        detailed_metrics[condition.lower()] = metrics
    
    # 转换为JSON兼容格式
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        else:
            return obj
    
    detailed_metrics_json = convert_numpy_types(detailed_metrics)
    
    import json
    metrics_path = os.path.join(old_data_dir, 'detailed_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(detailed_metrics_json, f, indent=2)
    print(f"详细指标已保存: {metrics_path}")
    
    # 4. 保存可视化（如果启用）
    if save_visualizations:
        print("生成可视化图片...")
        
        # 为每个原始标签生成可视化
        for label, network_data in label_networks.items():
            label_desc = old_net_cfg.get_label_description(label)
            
            # 相关性矩阵
            visualize_correlation_matrix(
                network_data['correlation_matrix'],
                title=f"Label {label} ({label_desc}) Correlation Matrix",
                save_path=os.path.join(old_data_dir, f'label_{label}_correlation_matrix.png')
            )
            
            # 邻接矩阵
            visualize_adjacency_matrix(
                network_data['adjacency_matrix'],
                title=f"Label {label} ({label_desc}) Adjacency Matrix",
                save_path=os.path.join(old_data_dir, f'label_{label}_adjacency_matrix.png')
            )
            
            # 连接分布
            visualize_connectivity_distribution(
                network_data['correlation_matrix'],
                network_data['adjacency_matrix'],
                title=f"Label {label} ({label_desc}) Connectivity Distribution",
                save_path=os.path.join(old_data_dir, f'label_{label}_connectivity_distribution.png')
            )
        
        # 为每个映射条件生成可视化
        for condition, network_data in condition_networks.items():
            condition_desc = network_data['description']
            
            # 相关性矩阵
            visualize_correlation_matrix(
                network_data['correlation_matrix'],
                title=f"{condition} ({condition_desc}) Correlation Matrix",
                save_path=os.path.join(old_data_dir, f'{condition.lower()}_correlation_matrix.png')
            )
            
            # 邻接矩阵  
            visualize_adjacency_matrix(
                network_data['adjacency_matrix'],
                title=f"{condition} ({condition_desc}) Adjacency Matrix", 
                save_path=os.path.join(old_data_dir, f'{condition.lower()}_adjacency_matrix.png')
            )
            
            # 连接分布
            visualize_connectivity_distribution(
                network_data['correlation_matrix'],
                network_data['adjacency_matrix'],
                title=f"{condition} ({condition_desc}) Connectivity Distribution",
                save_path=os.path.join(old_data_dir, f'{condition.lower()}_connectivity_distribution.png')
            )
        
        print("可视化图片生成完成")
    
    print(f"所有结果已保存到: {old_data_dir}")
    return old_data_dir

# %% 主脚本
if __name__ == "__main__":
    print("=== 旧版数据功能连接网络分析 ===")
    warnings.filterwarnings('ignore')
    
    # 显示旧版数据配置
    print_old_data_config()
    
    # %% 数据加载
    print("\n=== 旧版数据加载 ===")
    
    if cfg.LOADER_VERSION != 'old':
        print("错误: 此脚本仅适用于旧版数据，请设置 cfg.LOADER_VERSION = 'old'")
        print("当前设置:", cfg.LOADER_VERSION)
        exit(1)
    
    # 加载旧版数据
    from loaddata import load_old_version_data
    neuron_index, segments, labels, neuron_pos = load_old_version_data(
        cfg.OLD_VERSION_PATHS['neurons'],
        cfg.OLD_VERSION_PATHS['trials'],
        cfg.OLD_VERSION_PATHS['location']
    )
    
    # 处理位置数据
    neuron_pos = neuron_pos[0:2, :] if neuron_pos.shape[0] >= 2 else neuron_pos
    
    print(f"旧版数据加载完成:")
    print(f"  数据维度: {segments.shape}")
    print(f"  标签数量: {len(labels)}")
    print(f"  神经元位置: {neuron_pos.shape}")
    print(f"  标签分布: {Counter(labels)}")
    
    # 验证标签
    unique_labels = np.unique(labels)
    expected_labels = old_net_cfg.TARGET_LABELS
    print(f"  期望标签: {expected_labels}")
    print(f"  实际标签: {unique_labels.tolist()}")
    
    unexpected_labels = set(unique_labels) - set(expected_labels)
    if unexpected_labels:
        print(f"  警告: 发现未预期的标签: {unexpected_labels}")
    
    # RR神经元筛选
    print("\n=== RR神经元筛选 ===")
    rr_results = fast_rr_selection(segments, labels)
    
    if len(rr_results['rr_neurons']) > 0:
        print(f"使用 {len(rr_results['rr_neurons'])} 个RR神经元")
        network_segments = segments[:, rr_results['rr_neurons'], :]
        network_neuron_pos = neuron_pos[:, rr_results['rr_neurons']] if neuron_pos.shape[1] > 0 else None
    else:
        print("未找到RR神经元，使用所有神经元")
        network_segments = segments
        network_neuron_pos = neuron_pos
    
    # %% 网络分析
    print("\n=== 开始网络分析 ===")
    
    # 1. 按原始标签分析
    print("\n1. 按原始标签分析网络...")
    label_networks, label_metrics = analyze_old_data_by_original_labels(
        network_segments, labels, network_neuron_pos)
    
    # 2. 按映射条件分析
    print("\n2. 按映射后类别强度分析网络...")
    condition_networks, condition_metrics = analyze_old_data_by_mapped_categories(
        network_segments, labels, network_neuron_pos)
    
    # 3. 差异分析
    print("\n3. 分析原始标签与映射条件的差异...")
    comparison_results = analyze_old_data_differences(label_networks, condition_networks)
    
    # %% 结果保存
    print("\n=== 保存分析结果 ===")
    results_dir = cfg.get_results_dir()
    save_dir = save_old_data_analysis_results(
        label_networks, condition_networks, comparison_results, 
        results_dir, save_visualizations=True)
    
    # %% 分析总结
    print("\n=== 分析总结 ===")
    print(f"旧版数据网络分析完成")
    print(f"分析的原始标签: {list(label_networks.keys())}")
    print(f"分析的映射条件: {list(condition_networks.keys())}")
    print(f"结果保存位置: {save_dir}")
    
    print("\n标签映射总结:")
    for label in old_net_cfg.TARGET_LABELS:
        if label in old_net_cfg.OLD_LABEL_MAPPING:
            mapping = old_net_cfg.OLD_LABEL_MAPPING[label]
            print(f"  标签 {label} -> {mapping['description']} (类别{mapping['category']}, 强度{mapping['intensity']})")
    
    print("\n网络指标对比:")
    if comparison_results and 'comparison_table' in comparison_results:
        df = comparison_results['comparison_table']
        print(df[['type', 'identifier', 'description', 'density', 'avg_clustering', 'avg_degree']].to_string(index=False))
    
    print("\n=== 旧版数据分析完成 ===")