"""
RR神经元筛选模块

该模块实现了从MATLAB脚本转换而来的RR神经元筛选功能，包括：
1. 响应性神经元筛选（区分增强和抑制响应）
2. 可靠性神经元筛选（基于试次可靠性）
3. FDR校正进行多重比较校正
4. RR神经元的最终筛选

作者: 转换自MATLAB脚本
日期: 2025-08-26
"""

import numpy as np
from scipy import stats
from scipy.io import loadmat, savemat
import os
from typing import Tuple, List, Union


class RRNeuronSelector:
    """RR神经元筛选器类"""
    
    def __init__(self, 
                 ipd: int = 6,
                 isi: int = 4, 
                 sr: int = 4,
                 start_edge: int = 8,
                 t_stimulus: int = 8,
                 l: int = 24,
                 whole_length: int = 45,
                 alpha_fdr: float = 0.005,
                 alpha_level: float = 0.05,
                 reliability_ratio_threshold: float = 0.8):
        """
        初始化RR神经元筛选器
        
        Parameters:
        -----------
        ipd : int, default=6
            刺激间隔，用于设定滤波器参数
        isi : int, default=4
            刺激持续时间
        sr : int, default=4
            采样率，默认为4Hz
        start_edge : int, default=8
            刺激开始的时间
        t_stimulus : int, default=8
            刺激时段开始时间
        l : int, default=24
            刺激时段持续时间
        whole_length : int, default=45
            整个试次的时间长度
        alpha_fdr : float, default=0.005
            FDR校正的p值阈值
        alpha_level : float, default=0.05
            显著性水平
        reliability_ratio_threshold : float, default=0.8
            可靠性阈值（80%）
        """
        self.ipd = ipd
        self.isi = isi
        self.sr = sr
        self.start_edge = start_edge
        self.t_stimulus = t_stimulus
        self.l = l
        self.whole_length = whole_length
        self.alpha_fdr = alpha_fdr
        self.alpha_level = alpha_level
        self.reliability_ratio_threshold = reliability_ratio_threshold
        
        # 计算时间窗口
        self.baseline_window_pre = np.arange(0, t_stimulus)  # 刺激前的时间点（转为0索引）
        self.baseline_window_post = np.arange(t_stimulus + l + 4, whole_length)  # 刺激后的时间点
        self.post_stimulus_window = np.arange(t_stimulus, t_stimulus + l)  # 刺激后期窗口
    
    def fdr_bh(self, pvals: np.ndarray, q: float = 0.05, method: str = 'pdep') -> Tuple[np.ndarray, float, float, np.ndarray]:
        """
        Benjamini & Hochberg (1995) FDR控制
        
        Parameters:
        -----------
        pvals : np.ndarray
            p值数组
        q : float, default=0.05
            FDR水平
        method : str, default='pdep'
            校正方法 ('pdep' 或 'dep')
            
        Returns:
        --------
        h : np.ndarray
            显著性检验结果（布尔数组）
        crit_p : float
            临界p值
        adj_ci_cvrg : float
            调整后的置信区间覆盖率
        adj_p : np.ndarray
            调整后的p值
        """
        pvals_flat = pvals.flatten()
        m = len(pvals_flat)
        
        # 排序p值
        sort_ids = np.argsort(pvals_flat)
        p_sorted = pvals_flat[sort_ids]
        
        # 计算乘法因子
        if method == 'pdep':
            mult_factor = m
        elif method == 'dep':
            mult_factor = np.sum(1.0 / np.arange(1, m + 1))
        else:
            mult_factor = m
        
        # 临界p值
        crit_p = np.arange(1, m + 1) / m * q
        
        # 调整后的p值
        adj_p = p_sorted * mult_factor
        adj_p = np.minimum(adj_p, 1.0)
        
        # 恢复原始顺序
        adj_p_unsorted = np.zeros_like(adj_p)
        adj_p_unsorted[sort_ids] = adj_p
        
        # 找到显著的p值
        rej = p_sorted <= crit_p
        max_id = np.where(rej)[0]
        
        if len(max_id) == 0:
            h = np.zeros_like(pvals_flat, dtype=bool)
            crit_p_final = 0
        else:
            max_id = max_id[-1]
            h = pvals_flat <= crit_p[max_id]
            crit_p_final = crit_p[max_id]
        
        adj_ci_cvrg = (1 - q) ** (1 / m)
        
        # 恢复原始形状
        h = h.reshape(pvals.shape)
        adj_p_unsorted = adj_p_unsorted.reshape(pvals.shape)
        
        return h, crit_p_final, adj_ci_cvrg, adj_p_unsorted
    
    def select_responsive_neurons(self, trials: np.ndarray, indices: np.ndarray) -> Tuple[List[int], List[int], np.ndarray]:
        """
        筛选响应性神经元：区分增强和抑制
        
        Parameters:
        -----------
        trials : np.ndarray
            试次数据，形状为 (trials, neurons, timepoints)
        indices : np.ndarray
            条件索引
            
        Returns:
        --------
        enhanced_neurons : List[int]
            增强响应神经元索引
        suppressed_neurons : List[int]
            抑制响应神经元索引
        p_values : np.ndarray
            p值数组
        """
        N = trials.shape[1]  # 神经元数量
        p_values = np.zeros(N)
        enhanced_neurons = []
        suppressed_neurons = []
        
        for neuron_idx in range(N):
            neuron_responses = trials[indices, neuron_idx, :]
            
            # 获取基线响应（刺激前+刺激后）
            baseline_responses_pre = neuron_responses[:, self.baseline_window_pre]
            baseline_responses_post = neuron_responses[:, self.baseline_window_post]
            baseline_responses = np.concatenate([baseline_responses_pre.flatten(), 
                                               baseline_responses_post.flatten()])
            
            # 获取刺激期响应
            post_stimulus_responses = neuron_responses[:, self.post_stimulus_window].flatten()
            
            # Mann-Whitney U检验
            try:
                statistic, p = stats.mannwhitneyu(baseline_responses, post_stimulus_responses, 
                                                alternative='two-sided')
                p_values[neuron_idx] = p
                
                # 比较刺激期和基线期的平均值
                mean_baseline = np.mean(baseline_responses)
                mean_stimulus = np.mean(post_stimulus_responses)
                
                if p < self.alpha_fdr:
                    if mean_stimulus > mean_baseline:
                        enhanced_neurons.append(neuron_idx)
                    elif mean_stimulus < mean_baseline:
                        suppressed_neurons.append(neuron_idx)
            except ValueError:
                # 处理数据不足的情况
                p_values[neuron_idx] = 1.0
        
        # FDR校正
        h, crit_p, adj_ci_cvrg, adj_p_values = self.fdr_bh(p_values, self.alpha_fdr)
        
        # 根据校正后的p值筛选
        enhanced_neurons = [idx for idx in enhanced_neurons if adj_p_values[idx] < self.alpha_fdr]
        suppressed_neurons = [idx for idx in suppressed_neurons if adj_p_values[idx] < self.alpha_fdr]
        
        return enhanced_neurons, suppressed_neurons, p_values
    
    def select_reliable_neurons_by_trials(self, trials: np.ndarray, indices: np.ndarray) -> List[int]:
        """
        筛选可靠性神经元：80%的试次在刺激期显著响应
        
        Parameters:
        -----------
        trials : np.ndarray
            试次数据，形状为 (trials, neurons, timepoints)
        indices : np.ndarray
            条件索引
            
        Returns:
        --------
        reliable_neurons : List[int]
            可靠神经元索引列表
        """
        N = trials.shape[1]  # 神经元数量
        reliable_neurons = []
        
        for neuron_idx in range(N):
            neuron_responses = trials[indices, neuron_idx, :]
            num_trials = neuron_responses.shape[0]
            num_significant_trials = 0
            
            for trial_idx in range(num_trials):
                # 获取该试次的基线和刺激期数据
                baseline_responses_pre = neuron_responses[trial_idx, self.baseline_window_pre]
                baseline_responses_post = neuron_responses[trial_idx, self.baseline_window_post]
                baseline_responses = np.concatenate([baseline_responses_pre, baseline_responses_post])
                post_stimulus_responses = neuron_responses[trial_idx, self.post_stimulus_window]
                
                # Mann-Whitney U检验
                try:
                    statistic, p = stats.mannwhitneyu(baseline_responses, post_stimulus_responses, 
                                                    alternative='two-sided')
                    if p < self.alpha_level:
                        num_significant_trials += 1
                except ValueError:
                    # 处理数据不足的情况
                    continue
            
            # 计算满足显著性条件的试次比例
            if num_trials > 0:
                significant_ratio = num_significant_trials / num_trials
                if significant_ratio >= self.reliability_ratio_threshold:
                    reliable_neurons.append(neuron_idx)
        
        return reliable_neurons
    
    def select_rr_neurons(self, trials: np.ndarray, labels: np.ndarray, 
                         conditions: List[int] = None) -> dict:
        """
        主函数：筛选RR神经元
        
        Parameters:
        -----------
        trials : np.ndarray
            试次数据，形状为 (trials, neurons, timepoints)
        labels : np.ndarray
            试次标签
        conditions : List[int], optional
            需要分析的条件列表，默认为[1, 2]
            
        Returns:
        --------
        results : dict
            包含各种神经元筛选结果的字典
        """
        if conditions is None:
            conditions = [1, 2]
        
        # 分条件索引
        condition_indices = {}
        for condition in conditions:
            condition_indices[condition] = np.where(labels == condition)[0]
        
        # 筛选响应性神经元
        enhanced_neurons_all = []
        suppressed_neurons_all = []
        p_values_all = {}
        
        for condition in conditions:
            enhanced, suppressed, p_vals = self.select_responsive_neurons(
                trials, condition_indices[condition])
            enhanced_neurons_all.extend(enhanced)
            suppressed_neurons_all.extend(suppressed)
            p_values_all[condition] = p_vals
        
        # 合并增强和抑制神经元
        enhanced_neurons_union = list(set(enhanced_neurons_all))
        suppressed_neurons_union = list(set(suppressed_neurons_all))
        response_union = enhanced_neurons_union  # 只考虑增强响应
        
        # 筛选可靠性神经元
        reliable_neurons_all = []
        for condition in conditions:
            reliable = self.select_reliable_neurons_by_trials(
                trials, condition_indices[condition])
            reliable_neurons_all.extend(reliable)
        
        reliable_union = list(set(reliable_neurons_all))
        
        # 筛选RR神经元（响应性且可靠的神经元）
        rr_neurons = list(set(reliable_union) & set(response_union))
        
        results = {
            'rr_neurons': rr_neurons,
            'response_neurons': response_union,
            'reliable_neurons': reliable_union,
            'enhanced_neurons_union': enhanced_neurons_union,
            'suppressed_neurons_union': suppressed_neurons_union,
            'p_values': p_values_all
        }
        
        return results
    
    def process_data_file(self, folder_path: str, output_path: str = None) -> dict:
        """
        处理数据文件并保存结果
        
        Parameters:
        -----------
        folder_path : str
            数据文件夹路径
        output_path : str, optional
            输出路径，如果为None则保存到输入文件夹
            
        Returns:
        --------
        results : dict
            筛选结果
        """
        # 加载数据
        data_file = os.path.join(folder_path, 'Trial_data.mat')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"数据文件不存在: {data_file}")
        
        data = loadmat(data_file)
        trials = data['trials']
        labels = data['labels'].flatten()
        
        # 筛选RR神经元
        results = self.select_rr_neurons(trials, labels)
        
        # 保存结果
        if output_path is None:
            output_path = folder_path
        
        output_file = os.path.join(output_path, 'Neurons.mat')
        savemat(output_file, {
            'rr_neurons': results['rr_neurons'],
            'response_neurons': results['response_neurons'],
            'reliable_neurons': results['reliable_neurons'],
            'enhanced_neurons_union': results['enhanced_neurons_union'],
            'suppressed_neurons_union': results['suppressed_neurons_union']
        })
        
        print(f"结果已保存到: {output_file}")
        print(f"RR神经元数量: {len(results['rr_neurons'])}")
        print(f"响应性神经元数量: {len(results['response_neurons'])}")
        print(f"可靠性神经元数量: {len(results['reliable_neurons'])}")
        
        return results


def main():
    """主函数示例"""
    # 使用示例
    folder_path = r'D:\GuiYun\project\Mice Brain Data Analysis Pipeline\Data\M27_1008'
    
    # 创建筛选器实例
    selector = RRNeuronSelector()
    
    try:
        # 处理数据
        results = selector.process_data_file(folder_path)
        
        # 打印结果概要
        print("\n=== RR神经元筛选结果 ===")
        print(f"增强响应神经元: {len(results['enhanced_neurons_union'])} 个")
        print(f"抑制响应神经元: {len(results['suppressed_neurons_union'])} 个") 
        print(f"响应性神经元总数: {len(results['response_neurons'])} 个")
        print(f"可靠性神经元总数: {len(results['reliable_neurons'])} 个")
        print(f"最终RR神经元: {len(results['rr_neurons'])} 个")
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")


if __name__ == "__main__":
    main()