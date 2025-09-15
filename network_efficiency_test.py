# 网络效率分析测试脚本
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import os
from collections import defaultdict

# 导入必要模块
try:
    from loaddata import cfg, load_data, segment_neuron_data, reclassify_labels, fast_rr_selection
    from noise_correlation_analysis import NoiseCorrelationConfig, calculate_noise_correlation, shuffle_within_condition, build_networks_from_correlations
    print("成功导入必要模块")
except Exception as e:
    print(f"模块导入错误: {e}")
    import sys
    sys.exit(1)

def calculate_network_efficiency(adj_matrix):
    """计算网络的全局效率和局部效率"""
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

def simple_test():
    """简单测试网络效率计算"""
    print("开始简单测试...")
    
    # 创建测试数据
    np.random.seed(42)
    n_trials, n_neurons, n_timepoints = 20, 10, 50
    neural_data = np.random.randn(n_trials, n_neurons, n_timepoints)
    labels = np.array([1] * 10 + [2] * 10)
    
    print(f"测试数据维度: {neural_data.shape}")
    print(f"标签分布: {np.bincount(labels)}")
    
    # 测试网络效率计算
    test_adj = np.random.rand(n_neurons, n_neurons) > 0.7
    np.fill_diagonal(test_adj, 0)
    
    metrics = calculate_network_efficiency(test_adj)
    print(f"测试网络效率结果: {metrics}")
    
    # 测试噪音相关性计算
    try:
        noise_correlations = calculate_noise_correlation(neural_data, labels)
        print(f"噪音相关性矩阵维度: {[str(k) + ': ' + str(v.shape) for k, v in noise_correlations.items()]}")
        
        # 测试网络构建
        networks = build_networks_from_correlations(
            noise_correlations, 
            method='density',
            threshold=0.3, 
            density=0.2
        )
        print(f"网络构建成功，条件数: {len(networks)}")
        
        # 测试打乱
        shuffled_data = shuffle_within_condition(neural_data, labels, 0.5)
        print(f"打乱后数据维度: {shuffled_data.shape}")
        
        # 分析不同条件的网络效率
        results = {}
        for condition in [1, 2]:
            if condition in networks:
                adj_matrix = networks[condition]['adjacency_matrix']
                efficiency_metrics = calculate_network_efficiency(adj_matrix)
                results[condition] = efficiency_metrics
                print(f"条件 {condition} 网络效率: {efficiency_metrics}")
        
        # 简单可视化
        plt.figure(figsize=(10, 6))
        
        conditions = list(results.keys())
        global_effs = [results[c]['global_efficiency'] for c in conditions]
        local_effs = [results[c]['local_efficiency'] for c in conditions]
        
        plt.subplot(1, 2, 1)
        plt.bar(range(len(conditions)), global_effs)
        plt.title('Global Efficiency')
        plt.xlabel('Condition')
        plt.ylabel('Global Efficiency')
        plt.xticks(range(len(conditions)), [f'Cond {c}' for c in conditions])
        
        plt.subplot(1, 2, 2)
        plt.bar(range(len(conditions)), local_effs)
        plt.title('Local Efficiency')
        plt.xlabel('Condition')
        plt.ylabel('Local Efficiency')
        plt.xticks(range(len(conditions)), [f'Cond {c}' for c in conditions])
        
        plt.tight_layout()
        
        # 保存测试图
        os.makedirs('results/network_efficiency', exist_ok=True)
        plt.savefig('results/network_efficiency/test_result.png', dpi=300, bbox_inches='tight')
        print("测试图已保存到 results/network_efficiency/test_result.png")
        
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = simple_test()
    if success:
        print("简单测试成功完成！")
    else:
        print("测试失败")