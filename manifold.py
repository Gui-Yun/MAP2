# 神经数据降维分析
# guiy24@mails.tsinghua.edu.cn
# 2025-08-27
# PCA, t-SNE数据分析，并且准备数据用CEBRA运行

# %% 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

# 从loaddata.py导入函数
from loaddata import (
    load_data, segment_neuron_data, reclassify_labels, 
    fast_rr_selection, preprocess_neural_data, cfg
)

# %% 配置部分
class ManifoldConfig:
    """降维分析配置"""
    
    # 降维参数
    PCA_COMPONENTS = 3           # PCA降维目标维度
    TSNE_COMPONENTS = 2          # t-SNE降维目标维度
    TSNE_PERPLEXITY = 30         # t-SNE perplexity参数
    TSNE_LEARNING_RATE = 200     # t-SNE学习率
    TSNE_MAX_ITER = 1000        # t-SNE最大迭代次数
    
    # 可视化参数
    FIGURE_SIZE = (12, 8)        # 图形尺寸
    ALPHA = 0.7                  # 点的透明度
    POINT_SIZE = 50              # 散点大小
    
    # 颜色配置 - 使用饱和度差别
    CATEGORY_COLORS = {
        1: {'high': '#FF0000', 'low': '#FFB3B3'},    # 类别1 - 红色系：深红/浅红
        2: {'high': '#0000FF', 'low': '#B3B3FF'},    # 类别2 - 蓝色系：深蓝/浅蓝  
        3: {'high': '#00AA00', 'low': '#B3FFB3'}     # 类别3 - 绿色系：深绿/浅绿
    }

# 全局配置
mfg = ManifoldConfig()

# %% 函数定义部分

def prepare_data_for_manifold(segments, labels, rr_neurons=None, use_stimulus_only=True):
    """
    为降维分析准备数据
    
    参数:
    segments: 神经数据片段 (trials, neurons, timepoints)
    labels: 标签数组
    rr_neurons: RR神经元索引，None表示使用所有神经元
    use_stimulus_only: 是否只使用刺激期数据
    
    返回:
    X: 特征矩阵 (trials, features)
    y: 标签向量
    """
    print("准备降维数据...")
    
    # 使用所有数据，不进行过滤
    all_segments = segments
    all_labels = labels
    
    # 选择神经元
    if rr_neurons is not None:
        all_segments = all_segments[:, rr_neurons, :]
        print(f"使用 {len(rr_neurons)} 个RR神经元")
    else:
        print(f"使用所有 {all_segments.shape[1]} 个神经元")
    
    # 选择时间窗口 - 刺激时间段
    if use_stimulus_only:
        # 使用刺激期数据
        stimulus_start = cfg.PRE_FRAMES
        stimulus_end = cfg.PRE_FRAMES + cfg.STIMULUS_DURATION
        time_window = np.arange(stimulus_start, min(stimulus_end, all_segments.shape[2]))
        all_segments = all_segments[:, :, time_window]
        print(f"使用刺激期数据，时间窗口: {stimulus_start}-{stimulus_end}")
    else:
        print("使用完整时间序列数据")
    
    # 展平为特征矩阵
    X = all_segments.reshape(all_segments.shape[0], -1)  # (trials, neurons * timepoints)
    y = all_labels
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"数据维度: {X_scaled.shape}, 标签分布: {np.unique(y, return_counts=True)}")
    return X_scaled, y

def perform_pca(X, n_components=mfg.PCA_COMPONENTS):
    """
    执行PCA降维
    
    参数:
    X: 输入特征矩阵
    n_components: 降维目标维度
    
    返回:
    X_pca: PCA降维结果
    pca: PCA对象
    """
    print(f"执行PCA降维到{n_components}维...")
    pca = PCA(n_components=n_components, random_state=cfg.RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    
    explained_ratio = np.sum(pca.explained_variance_ratio_)
    print(f"PCA完成，解释方差比: {explained_ratio:.3f}")
    print(f"各维度解释方差比: {pca.explained_variance_ratio_}")
    
    return X_pca, pca

def perform_tsne(X, n_components=mfg.TSNE_COMPONENTS):
    """
    执行t-SNE降维
    
    参数:
    X: 输入特征矩阵
    n_components: 降维目标维度
    
    返回:
    X_tsne: t-SNE降维结果
    """
    print(f"执行t-SNE降维到{n_components}维...")
    tsne = TSNE(
        n_components=n_components,
        perplexity=mfg.TSNE_PERPLEXITY,
        learning_rate=mfg.TSNE_LEARNING_RATE,
        max_iter=mfg.TSNE_MAX_ITER,
        random_state=cfg.RANDOM_STATE,
        verbose=1
    )
    X_tsne = tsne.fit_transform(X)
    print("t-SNE降维完成")
    
    return X_tsne

def get_color_by_intensity(category, intensity):
    """
    根据类别和强度获取颜色（使用饱和度差别）
    
    参数:
    category: 类别标签
    intensity: 强度值
    
    返回:
    color: 颜色（根据强度选择高/低饱和度）
    """
    color_dict = mfg.CATEGORY_COLORS.get(category, {'high': 'gray', 'low': 'lightgray'})
    
    # 强度0使用低饱和度颜色，强度1使用高饱和度颜色
    if intensity == 0:
        color = color_dict['low']   # 低饱和度（浅色）
    else:
        color = color_dict['high']  # 高饱和度（深色）
    
    return color

def plot_manifold_2d(X_reduced, labels, stimulus_data, title="2D Manifold"):
    """
    绘制2D降维结果
    
    参数:
    X_reduced: 降维后的2D数据
    labels: 类别标签（已过滤的有效标签）
    stimulus_data: 刺激数据（对应的有效试次数据）
    title: 图标题
    """
    plt.figure(figsize=mfg.FIGURE_SIZE)
    
    # 按类别分别绘制
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        if np.sum(mask) == 0:
            continue
            
        X_label = X_reduced[mask]
        stimulus_label = stimulus_data[mask]
        
        # 按强度分别绘制
        intensities = np.unique(stimulus_label[:, 1])
        for intensity in intensities:
            intensity_mask = stimulus_label[:, 1] == intensity
            if np.sum(intensity_mask) == 0:
                continue
                
            X_plot = X_label[intensity_mask]
            color = get_color_by_intensity(label, intensity)
            
            label_name = f"Category {int(label)}"
            if intensity == 1:
                label_name += "_Noise"
            
            plt.scatter(X_plot[:, 0], X_plot[:, 1], 
                       c=color, alpha=mfg.ALPHA, s=mfg.POINT_SIZE,
                       label=label_name, edgecolors='black', linewidth=0.5)
    
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_manifold_3d(X_reduced, labels, stimulus_data, title="3D Manifold"):
    """
    绘制3D降维结果
    
    参数:
    X_reduced: 降维后的3D数据
    labels: 类别标签（已过滤的有效标签）
    stimulus_data: 刺激数据（对应的有效试次数据）
    title: 图标题
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=mfg.FIGURE_SIZE)
    ax = fig.add_subplot(111, projection='3d')
    
    # 按类别和强度绘制
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        if np.sum(mask) == 0:
            continue
            
        X_label = X_reduced[mask]
        stimulus_label = stimulus_data[mask]
        
        intensities = np.unique(stimulus_label[:, 1])
        for intensity in intensities:
            intensity_mask = stimulus_label[:, 1] == intensity
            if np.sum(intensity_mask) == 0:
                continue
                
            X_plot = X_label[intensity_mask]
            color = get_color_by_intensity(label, intensity)
            
            label_name = f"Category {int(label)}"
            if intensity == 1:
                label_name += "_Noise"
            
            ax.scatter(X_plot[:, 0], X_plot[:, 1], X_plot[:, 2],
                      c=color, alpha=mfg.ALPHA, s=mfg.POINT_SIZE,
                      label=label_name, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

def prepare_cebra_data(segments, stimulus_data, rr_neurons=None, use_stimulus_only=True):
    """
    专门为CEBRA准备数据格式
    
    CEBRA需要的数据格式:
    - neural_data: (time_steps, n_neurons) 时间序列神经活动
    - labels: 各种形式的辅助变量/标签
    
    参数:
    segments: 神经数据片段 (trials, neurons, timepoints)
    stimulus_data: 刺激数据 (trials, [category, intensity])
    rr_neurons: RR神经元索引
    use_stimulus_only: 是否只使用刺激期数据
    
    返回:
    cebra_data: 为CEBRA准备的数据字典
    """
    print("为CEBRA准备数据...")
    
    # 选择神经元
    if rr_neurons is not None:
        neural_segments = segments[:, rr_neurons, :]
        print(f"使用 {len(rr_neurons)} 个RR神经元")
    else:
        neural_segments = segments
        print(f"使用所有 {neural_segments.shape[1]} 个神经元")
    
    # 选择时间窗口
    if use_stimulus_only:
        stimulus_start = cfg.PRE_FRAMES
        stimulus_end = cfg.PRE_FRAMES + cfg.STIMULUS_DURATION
        time_window = np.arange(stimulus_start, min(stimulus_end, neural_segments.shape[2]))
        neural_segments = neural_segments[:, :, time_window]
        print(f"使用刺激期数据，时间窗口: {stimulus_start}-{stimulus_end}")
    
    n_trials, n_neurons, n_timepoints = neural_segments.shape
    print(f"数据形状: {n_trials} trials × {n_neurons} neurons × {n_timepoints} timepoints")
    
    # 数据质量检查
    print(f"神经数据统计: min={neural_segments.min():.3f}, max={neural_segments.max():.3f}, mean={neural_segments.mean():.3f}")
    print(f"刺激标签分布: {np.unique(stimulus_data, axis=0, return_counts=True)}")
    
    # 方法1: 正确展平成连续时间序列 (CEBRA-Time)
    # 重要：保持每个trial内的时间连续性，按trial顺序连接
    neural_timeseries = neural_segments.transpose(0, 2, 1).reshape(-1, n_neurons)  # (trials*timepoints, neurons)
    
    # 创建对应的时间戳 - 保持全局时间连续性
    timestamps = np.arange(neural_timeseries.shape[0])
    
    # 创建对应的标签时间序列 - 每个trial内所有时间点使用相同标签
    category_timeseries = np.repeat(stimulus_data[:, 0], n_timepoints)  # 类别标签
    intensity_timeseries = np.repeat(stimulus_data[:, 1], n_timepoints)  # 强度标签
    
    # 创建trial索引用于追踪数据来源
    trial_indices = np.repeat(np.arange(n_trials), n_timepoints)
    
    # 方法2: 保持trial结构用于CEBRA-Behavior
    # trial级别的标签
    # 新的强度映射策略：基于类别的对称连续映射
    # 类别1: 1->0, 0.5->0.5, 0.2->0.8, 0->1.0
    # 类别2: 1->0, 0.5->-0.5, 0.2->-0.8, 0->-1.0
    
    intensity_continuous = np.zeros_like(stimulus_data[:, 1])
    for i in range(len(stimulus_data)):
        category = stimulus_data[i, 0]
        intensity = stimulus_data[i, 1]
        
        if intensity == 1.0:      # 噪音强度
            intensity_continuous[i] = 0.0
        elif intensity == 0.5:    # 中等强度
            intensity_continuous[i] = 0.5 if category == 1 else -0.5
        elif intensity == 0.2:    # 低强度  
            intensity_continuous[i] = 0.8 if category == 1 else -0.8
        elif intensity == 0.0:    # 无噪音
            intensity_continuous[i] = 1.0 if category == 1 else -1.0
    
    # 为离散标签保留旧的映射
    intensity_mapped = (stimulus_data[:, 1] * 10).astype(int)
    
    trial_labels = {
        'category': stimulus_data[:, 0].astype(int),
        'intensity_discrete': intensity_mapped,
        'intensity_continuous': intensity_continuous,
        'combined': stimulus_data[:, 0].astype(int) * 100 + intensity_mapped
    }
    
    print(f"原始强度: {np.unique(stimulus_data[:, 1])}")
    print(f"连续强度映射: {np.unique(intensity_continuous)}")
    print(f"强度分布: {np.unique(intensity_continuous, return_counts=True)}")
    print(f"组合标签: {np.unique(trial_labels['combined'])}")
    
    # 创建连续强度时间序列
    intensity_continuous_timeseries = np.repeat(intensity_continuous, n_timepoints)
    
    # 方法3: 创建连续标签用于CEBRA混合模式
    # 为每个时间点创建连续的"行为"标签，包含时间内信息
    behavioral_labels = np.zeros((neural_timeseries.shape[0], 4))
    behavioral_labels[:, 0] = category_timeseries  # 类别（1, 2）
    behavioral_labels[:, 1] = intensity_continuous_timeseries  # 连续强度（-1到1）
    behavioral_labels[:, 2] = trial_indices  # trial索引
    
    # 创建trial内时间位置标签 (0到n_timepoints-1, 在每个trial内重复)
    within_trial_time = np.tile(np.arange(n_timepoints), n_trials)
    behavioral_labels[:, 3] = within_trial_time / (n_timepoints - 1)  # 归一化的trial内时间
    
    return {
        # CEBRA-Time格式: 纯时间序列，无标签
        'timeseries_data': {
            'neural_data': neural_timeseries.astype(np.float32),
            'timestamps': timestamps.astype(np.int32)
        },
        
        # CEBRA-Behavior格式: 离散标签
        'discrete_label_data': {
            'neural_data': neural_timeseries.astype(np.float32),
            'category_labels': category_timeseries.astype(np.int32),
            'intensity_labels': (intensity_timeseries * 10).astype(np.int32),  # 离散强度映射
            'intensity_continuous_labels': intensity_continuous_timeseries.astype(np.float32),  # 连续强度映射
            'combined_labels': (category_timeseries.astype(int) * 100 + (intensity_timeseries * 10).astype(int)).astype(np.int32),
            'trial_indices': trial_indices.astype(np.int32),
            'within_trial_time': within_trial_time.astype(np.int32)
        },
        
        # CEBRA-Hybrid格式: 连续标签
        'continuous_label_data': {
            'neural_data': neural_timeseries.astype(np.float32),
            'behavioral_labels': behavioral_labels.astype(np.float32)
        },
        
        # Trial结构数据（用于其他分析）
        'trial_data': {
            'neural_segments': neural_segments.astype(np.float32),
            'trial_labels': trial_labels,
            'stimulus_data': stimulus_data.astype(np.int32)
        },
        
        # 元数据
        'metadata': {
            'n_trials': n_trials,
            'n_neurons': n_neurons,
            'n_timepoints': n_timepoints,
            'n_total_timepoints': neural_timeseries.shape[0],
            'stimulus_window': [stimulus_start, stimulus_end] if use_stimulus_only else None,
            'rr_neurons': rr_neurons if rr_neurons is not None else list(range(n_neurons)),
            'sampling_info': {
                'pre_frames': cfg.PRE_FRAMES,
                'post_frames': cfg.POST_FRAMES,
                'stimulus_duration': cfg.STIMULUS_DURATION
            }
        }
    }

def save_cebra_data(cebra_data, base_path):
    """
    保存CEBRA格式数据到多个文件
    
    参数:
    cebra_data: prepare_cebra_data返回的数据字典
    base_path: 保存路径的基础目录
    """
    import json
    
    print("保存CEBRA数据到:", base_path)
    os.makedirs(base_path, exist_ok=True)
    
    # 1. 保存CEBRA-Time数据 (纯时间序列)
    time_data = cebra_data['timeseries_data']
    np.savez_compressed(
        os.path.join(base_path, 'cebra_time_data.npz'),
        neural_data=time_data['neural_data'],
        timestamps=time_data['timestamps']
    )
    print("CEBRA-Time数据已保存")
    
    # 2. 保存CEBRA-Behavior数据 (离散标签)
    discrete_data = cebra_data['discrete_label_data']
    np.savez_compressed(
        os.path.join(base_path, 'cebra_behavior_data.npz'),
        neural_data=discrete_data['neural_data'],
        category_labels=discrete_data['category_labels'],
        intensity_labels=discrete_data['intensity_labels'],
        intensity_continuous_labels=discrete_data['intensity_continuous_labels'],
        combined_labels=discrete_data['combined_labels'],
        trial_indices=discrete_data['trial_indices'],
        within_trial_time=discrete_data['within_trial_time']
    )
    print("CEBRA-Behavior数据已保存")
    
    # 3. 保存CEBRA-Hybrid数据 (连续标签)
    continuous_data = cebra_data['continuous_label_data']
    np.savez_compressed(
        os.path.join(base_path, 'cebra_hybrid_data.npz'),
        neural_data=continuous_data['neural_data'],
        behavioral_labels=continuous_data['behavioral_labels']
    )
    print("CEBRA-Hybrid数据已保存")
    
    # 4. 保存Trial结构数据
    trial_data = cebra_data['trial_data']
    np.savez_compressed(
        os.path.join(base_path, 'trial_structure_data.npz'),
        neural_segments=trial_data['neural_segments'],
        stimulus_data=trial_data['stimulus_data'],
        **{f"trial_labels_{k}": v for k, v in trial_data['trial_labels'].items()}
    )
    print("Trial结构数据已保存")
    
    # 5. 保存元数据为JSON
    metadata = cebra_data['metadata']
    # 转换numpy数组为列表以便JSON序列化
    metadata_json = {}
    for k, v in metadata.items():
        if isinstance(v, np.ndarray):
            metadata_json[k] = v.tolist()
        elif isinstance(v, dict):
            metadata_json[k] = {kk: vv.tolist() if isinstance(vv, np.ndarray) else vv for kk, vv in v.items()}
        else:
            metadata_json[k] = v
    
    with open(os.path.join(base_path, 'metadata.json'), 'w') as f:
        json.dump(metadata_json, f, indent=2)
    print("元数据已保存")
    
    # 6. 创建README文件
    readme_content = """# CEBRA数据格式说明

## 文件说明:

### 1. cebra_time_data.npz (CEBRA-Time)
- neural_data: (n_timepoints, n_neurons) 神经活动时间序列
- timestamps: (n_timepoints,) 时间戳
- 用于: 无监督时间嵌入学习

### 2. cebra_behavior_data.npz (CEBRA-Behavior) 
- neural_data: (n_timepoints, n_neurons) 神经活动
- category_labels: (n_timepoints,) 类别标签
- intensity_labels: (n_timepoints,) 强度标签  
- combined_labels: (n_timepoints,) 组合标签
- 用于: 有监督行为嵌入学习

### 3. cebra_hybrid_data.npz (CEBRA-Hybrid)
- neural_data: (n_timepoints, n_neurons) 神经活动
- behavioral_labels: (n_timepoints, 3) [类别, 强度, 时间]
- 用于: 混合模式嵌入学习

### 4. trial_structure_data.npz
- neural_segments: (n_trials, n_neurons, n_timepoints) 原始trial结构
- stimulus_data: (n_trials, 2) 刺激信息
- trial_labels_*: 各种trial级别标签
- 用于: 其他分析和验证

### 5. metadata.json
- 包含所有数据维度信息、参数设置等元数据

## 使用示例:

```python
import numpy as np
import cebra

# CEBRA-Time示例
data = np.load('cebra_time_data.npz')
neural_data = data['neural_data']

cebra_time = cebra.CEBRA(model_architecture='offset10-model', 
                         batch_size=512, 
                         learning_rate=3e-4,
                         temperature=1,
                         output_dimension=16,
                         max_iterations=10000,
                         device='cuda')

cebra_time.fit(neural_data)
embedding = cebra_time.transform(neural_data)

# CEBRA-Behavior示例  
data = np.load('cebra_behavior_data.npz')
neural_data = data['neural_data'] 
labels = data['category_labels']

cebra_behavior = cebra.CEBRA(model_architecture='offset10-model',
                             batch_size=512,
                             learning_rate=3e-4, 
                             temperature=1,
                             output_dimension=16,
                             max_iterations=10000,
                             device='cuda')

cebra_behavior.fit(neural_data, labels)
embedding = cebra_behavior.transform(neural_data)
```
"""
    
    with open(os.path.join(base_path, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("README文档已保存")
    
    print("\n=== CEBRA数据保存完成 ===")
    print("保存位置:", base_path)
    print("数据形状:", metadata['n_total_timepoints'], "timepoints ×", metadata['n_neurons'], "neurons")
    print("Trial数量:", metadata['n_trials'])
    neuron_type = 'RR神经元' if len(metadata['rr_neurons']) < metadata['n_neurons'] else '所有神经元'
    print("神经元类型:", neuron_type)

def save_data_for_cebra(X, labels, save_path):
    """
    保存数据为CEBRA格式 (向后兼容的简单版本)
    
    参数:
    X: 特征数据
    labels: 标签数据
    save_path: 保存路径
    """
    print("保存简单CEBRA数据到:", save_path)
    
    # 保存为npz格式
    np.savez(save_path,
             neural_data=X,
             labels=labels)
    
    print("简单CEBRA数据保存完成")

# %% 主脚本
if __name__ == "__main__":
    print("=== 神经活动降维分析脚本 ===")
    
    # %% 数据加载与预处理
    print("\n1. 数据加载...")
    neuron_data, neuron_pos, trigger_data, stimulus_data = load_data(cfg.DATA_PATH)
    
    # 数据分割
    segments, labels = segment_neuron_data(neuron_data, trigger_data, stimulus_data)
    
    # 使用原始标签（第一列类别）作为降维标签
    original_labels = stimulus_data[:, 0]  # 第一列是类别
    
    # RR神经元筛选（可选）
    print("\n2. RR神经元筛选...")
    # 为RR筛选创建有效标签（排除0）
    rr_labels = reclassify_labels(stimulus_data)
    rr_results = fast_rr_selection(segments, rr_labels)
    rr_neurons = rr_results['rr_neurons'] if len(rr_results['rr_neurons']) > 0 else None
    
    # 准备降维数据 - 使用原始类别标签和刺激时间段
    X, y = prepare_data_for_manifold(segments, original_labels, rr_neurons, use_stimulus_only=True)
    
    # %% 数据降维分析
    print("\n3. 降维分析...")
    
    # PCA降维
    X_pca, pca = perform_pca(X, n_components=3)
    
    # t-SNE降维
    X_tsne = perform_tsne(X, n_components=2)
    
    # %% 可视化
    print("\n4. 可视化结果...")
    
    # 直接使用完整的刺激数据，不进行过滤
    # stimulus_data包含所有trial的信息：第一列类别，第二列强度
    
    # 绘制PCA 3D结果
    plot_manifold_3d(X_pca, y, stimulus_data, title="PCA 3D Manifold")
    
    # 绘制PCA 2D结果（前两个主成分）
    plot_manifold_2d(X_pca[:, :2], y, stimulus_data, title="PCA 2D Manifold")
    
    # 绘制t-SNE 2D结果
    plot_manifold_2d(X_tsne, y, stimulus_data, title="t-SNE 2D Manifold")
    
    # %% 专业CEBRA数据准备和保存
    print("\n5. 准备CEBRA数据...")
    
    # 为CEBRA准备专业格式的数据
    cebra_data = prepare_cebra_data(
        segments=segments, 
        stimulus_data=stimulus_data, 
        rr_neurons=rr_neurons, 
        use_stimulus_only=True
    )
    
    # 保存CEBRA数据到专门目录
    cebra_save_dir = os.path.join(cfg.DATA_PATH, 'cebra_data')
    save_cebra_data(cebra_data, cebra_save_dir)
    
    # %% 传统降维数据保存（向后兼容）
    print("\n6. 保存传统降维数据...")
    
    # 保存降维结果和原始数据
    save_dir = os.path.join(cfg.DATA_PATH, 'manifold_results')
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存PCA结果
    save_data_for_cebra(X_pca, y, os.path.join(save_dir, 'pca_data.npz'))
    
    # 保存原始高维数据
    save_data_for_cebra(X, y, os.path.join(save_dir, 'original_data.npz'))
    
    print("\n=== 降维分析完成 ===")   