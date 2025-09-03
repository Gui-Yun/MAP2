# 神经数据t-SNE降维分析
# guiy24@mails.tsinghua.edu.cn
# 2025-09-03
# 专注于t-SNE降维分析和科研风格可视化

# %% 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from matplotlib.patches import Ellipse
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

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

def setup_scientific_plot_style():
    """设置科研论文风格的绘图参数"""
    plt.style.use('default')
    
    # 设置字体和大小
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'font.family': 'Arial',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'legend.frameon': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

def plot_tsne_scientific(X_tsne, labels, stimulus_data, save_path=None, title="t-SNE Analysis"):
    """
    科研风格的t-SNE可视化
    
    参数:
    X_tsne: t-SNE降维结果 (n_samples, 2)
    labels: 类别标签
    stimulus_data: 刺激数据 (包含类别和强度信息)
    save_path: 保存路径
    title: 图标题
    """
    setup_scientific_plot_style()
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, y=0.95)
    
    # 准备数据
    unique_labels = np.unique(labels)
    n_categories = len(unique_labels)
    
    # 定义科研风格的颜色方案
    category_colors = {
        1: {'color': '#E31A1C', 'name': 'Category 1'},  # 红色
        2: {'color': '#1F78B4', 'name': 'Category 2'},  # 蓝色  
        3: {'color': '#33A02C', 'name': 'Category 3'}   # 绿色
    }
    
    # 子图1: 按类别着色
    ax1.set_title('A. t-SNE by Category', fontweight='bold', loc='left')
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if np.sum(mask) > 0:
            color_info = category_colors.get(label, {'color': 'gray', 'name': f'Category {label}'})
            ax1.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c=color_info['color'], 
                       alpha=0.7, 
                       s=30, 
                       label=color_info['name'],
                       edgecolors='white',
                       linewidths=0.5)
    
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.legend(loc='upper right', frameon=True, fancybox=False, shadow=False)
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    
    # 子图2: 按强度着色（如果有强度信息）
    ax2.set_title('B. t-SNE by Intensity', fontweight='bold', loc='left')
    if stimulus_data.shape[1] > 1:
        intensities = stimulus_data[:, 1]
        unique_intensities = np.unique(intensities)
        
        # 使用连续颜色映射
        cmap = plt.cm.viridis
        scatter = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                             c=intensities, 
                             cmap=cmap,
                             alpha=0.7, 
                             s=30,
                             edgecolors='white',
                             linewidths=0.5)
        
        cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8)
        cbar.set_label('Stimulus Intensity', rotation=270, labelpad=15)
    else:
        ax2.text(0.5, 0.5, 'No intensity data available', 
                transform=ax2.transAxes, ha='center', va='center')
    
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    
    # 子图3: 密度图
    ax3.set_title('C. Data Point Density', fontweight='bold', loc='left')
    
    # 创建2D直方图密度图
    hist, xedges, yedges = np.histogram2d(X_tsne[:, 0], X_tsne[:, 1], bins=50, density=True)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    im = ax3.imshow(hist.T, extent=extent, origin='lower', cmap='Blues', alpha=0.8)
    ax3.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.4, s=8, c='red', edgecolors='none')
    
    cbar3 = plt.colorbar(im, ax=ax3, shrink=0.8)
    cbar3.set_label('Density', rotation=270, labelpad=15)
    
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')
    
    # 子图4: 类别分离统计
    ax4.set_title('D. Category Separation Analysis', fontweight='bold', loc='left')
    
    # 计算类别间的距离统计
    category_stats = []
    for label in unique_labels:
        mask = labels == label
        if np.sum(mask) > 0:
            data_points = X_tsne[mask]
            # 计算类内距离
            centroid = np.mean(data_points, axis=0)
            intra_distances = np.sqrt(np.sum((data_points - centroid)**2, axis=1))
            
            category_stats.append({
                'category': label,
                'centroid': centroid,
                'intra_mean': np.mean(intra_distances),
                'intra_std': np.std(intra_distances),
                'n_points': len(data_points)
            })
    
    # 绘制统计信息
    categories = [stat['category'] for stat in category_stats]
    intra_means = [stat['intra_mean'] for stat in category_stats]
    intra_stds = [stat['intra_std'] for stat in category_stats]
    
    x_pos = np.arange(len(categories))
    bars = ax4.bar(x_pos, intra_means, yerr=intra_stds, 
                   capsize=5, alpha=0.7, 
                   color=[category_colors.get(cat, {'color': 'gray'})['color'] for cat in categories])
    
    ax4.set_xlabel('Category')
    ax4.set_ylabel('Intra-cluster Distance')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'Cat {cat}' for cat in categories])
    ax4.grid(True, axis='y', alpha=0.3, linewidth=0.5)
    
    # 添加数值标签
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, intra_means, intra_stds)):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std_val + 0.1,
                f'{mean_val:.2f}±{std_val:.2f}', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图形
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"t-SNE科研风格图已保存: {save_path}")
    
    plt.show()
    
    return category_stats

def plot_tsne_publication_figure(X_tsne, labels, stimulus_data, save_path=None):
    """
    生成适合论文发表的t-SNE单一主图
    
    参数:
    X_tsne: t-SNE降维结果
    labels: 类别标签
    stimulus_data: 刺激数据
    save_path: 保存路径
    """
    setup_scientific_plot_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # 准备数据和颜色
    unique_labels = np.unique(labels)
    category_colors = {
        1: {'color': '#E31A1C', 'marker': 'o', 'name': 'Category 1'},
        2: {'color': '#1F78B4', 'marker': 's', 'name': 'Category 2'},
        3: {'color': '#33A02C', 'marker': '^', 'name': 'Category 3'}
    }
    
    # 绘制散点图
    for label in unique_labels:
        mask = labels == label
        if np.sum(mask) > 0:
            color_info = category_colors.get(label, {
                'color': 'gray', 'marker': 'o', 'name': f'Category {label}'
            })
            
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                      c=color_info['color'],
                      marker=color_info['marker'],
                      alpha=0.8,
                      s=50,
                      label=color_info['name'],
                      edgecolors='white',
                      linewidths=0.8)
    
    # 设置坐标轴
    ax.set_xlabel('t-SNE Component 1', fontsize=13)
    ax.set_ylabel('t-SNE Component 2', fontsize=13)
    
    # 添加图例
    legend = ax.legend(loc='upper right', frameon=True, fancybox=False, 
                      shadow=False, fontsize=11)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # 设置网格
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # 移除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 设置坐标轴样式
    ax.tick_params(axis='both', which='major', labelsize=11, 
                  length=6, width=1.2, color='black')
    
    plt.tight_layout()
    
    # 保存图形
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"论文级t-SNE图已保存: {save_path}")
    
    plt.show()

def calculate_tsne_metrics(X_tsne, labels):
    """
    计算t-SNE结果的量化指标
    
    参数:
    X_tsne: t-SNE降维结果
    labels: 类别标签
    
    返回:
    metrics: 包含各种指标的字典
    """
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.cluster import KMeans
    
    metrics = {}
    
    # 轮廓系数
    if len(np.unique(labels)) > 1:
        silhouette = silhouette_score(X_tsne, labels)
        metrics['silhouette_score'] = silhouette
        print(f"轮廓系数 (Silhouette Score): {silhouette:.3f}")
    
    # 类别间分离度
    unique_labels = np.unique(labels)
    centroids = {}
    intra_variances = {}
    
    for label in unique_labels:
        mask = labels == label
        if np.sum(mask) > 0:
            data_points = X_tsne[mask]
            centroid = np.mean(data_points, axis=0)
            centroids[label] = centroid
            
            # 类内方差
            intra_var = np.mean(np.sum((data_points - centroid)**2, axis=1))
            intra_variances[label] = intra_var
    
    # 计算类间距离
    inter_distances = []
    for i, label1 in enumerate(unique_labels):
        for j, label2 in enumerate(unique_labels):
            if i < j:
                dist = np.sqrt(np.sum((centroids[label1] - centroids[label2])**2))
                inter_distances.append(dist)
    
    metrics['mean_inter_distance'] = np.mean(inter_distances)
    metrics['mean_intra_variance'] = np.mean(list(intra_variances.values()))
    metrics['separation_ratio'] = metrics['mean_inter_distance'] / metrics['mean_intra_variance']
    
    print(f"平均类间距离: {metrics['mean_inter_distance']:.3f}")
    print(f"平均类内方差: {metrics['mean_intra_variance']:.3f}")
    print(f"分离比率: {metrics['separation_ratio']:.3f}")
    
    return metrics

def save_manifold_2d_data(X_reduced, labels, stimulus_data, title="2D Manifold", save_dir="results"):
    """
    保存2D降维结果数据
    
    参数:
    X_reduced: 降维后的2D数据
    labels: 类别标签（已过滤的有效标签）
    stimulus_data: 刺激数据（对应的有效试次数据）
    title: 数据标题
    save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    manifold_2d_data = {
        'reduced_data': X_reduced,
        'labels': labels,
        'stimulus_data': stimulus_data,
        'title': title,
        'n_samples': X_reduced.shape[0],
        'n_dimensions': X_reduced.shape[1]
    }
    
    # 按类别和强度组织数据
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        if np.sum(mask) == 0:
            continue
            
        X_label = X_reduced[mask]
        stimulus_label = stimulus_data[mask]
        
        # 按强度分别保存
        intensities = np.unique(stimulus_label[:, 1])
        for intensity in intensities:
            intensity_mask = stimulus_label[:, 1] == intensity
            if np.sum(intensity_mask) == 0:
                continue
                
            X_plot = X_label[intensity_mask]
            color = get_color_by_intensity(label, intensity)
            
            key_name = f"category_{int(label)}_intensity_{intensity}"
            manifold_2d_data[key_name] = {
                'data': X_plot,
                'color': color,
                'n_samples': X_plot.shape[0]
            }
    
    filename = title.replace(' ', '_').replace(':', '').lower() + '_2d_data.npz'
    np.savez_compressed(
        os.path.join(save_dir, filename),
        **manifold_2d_data
    )
    print(f"2D降维数据已保存: {filename}")

def save_manifold_3d_data(X_reduced, labels, stimulus_data, title="3D Manifold", save_dir="results"):
    """
    保存3D降维结果数据
    
    参数:
    X_reduced: 降维后的3D数据
    labels: 类别标签（已过滤的有效标签）
    stimulus_data: 刺激数据（对应的有效试次数据）
    title: 数据标题
    save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    manifold_3d_data = {
        'reduced_data': X_reduced,
        'labels': labels,
        'stimulus_data': stimulus_data,
        'title': title,
        'n_samples': X_reduced.shape[0],
        'n_dimensions': X_reduced.shape[1]
    }
    
    # 按类别和强度组织数据
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
            
            key_name = f"category_{int(label)}_intensity_{intensity}"
            manifold_3d_data[key_name] = {
                'data': X_plot,
                'color': color,
                'n_samples': X_plot.shape[0]
            }
    
    filename = title.replace(' ', '_').replace(':', '').lower() + '_3d_data.npz'
    np.savez_compressed(
        os.path.join(save_dir, filename),
        **manifold_3d_data
    )
    print(f"3D降维数据已保存: {filename}")

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
    print("=== 神经活动t-SNE降维分析脚本 ===")
    
    # %% 数据加载与预处理
    print("\n1. 数据加载...")
    if cfg.LOADER_VERSION == 'new':
        neuron_data, neuron_pos, trigger_data, stimulus_data = load_data(cfg.DATA_PATH)
        segments, labels = segment_neuron_data(neuron_data, trigger_data, stimulus_data)
    elif cfg.LOADER_VERSION == 'old':
        from loaddata import load_old_version_data
        neuron_index, segments, labels, neuron_pos = load_old_version_data(
            cfg.OLD_VERSION_PATHS['neurons'],
            cfg.OLD_VERSION_PATHS['trials'],
            cfg.OLD_VERSION_PATHS['location']
        )
        # 对于旧版数据，需要创建兼容的stimulus_data
        stimulus_data = np.column_stack([labels, np.zeros(len(labels))])  # 简化处理
        neuron_pos = neuron_pos[0:2, :] if neuron_pos.shape[0] >= 2 else neuron_pos
        print("已切换到旧版数据加载模式")
    else:
        raise ValueError("无效的 LOADER_VERSION 配置")
    
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
    
    # %% t-SNE降维分析
    print("\n3. t-SNE降维分析...")
    
    # 首先进行PCA预降维以提高t-SNE效果
    print("执行PCA预降维...")
    X_pca, pca = perform_pca(X, n_components=min(50, X.shape[1]//2))
    
    # t-SNE降维（使用PCA预处理的数据）
    print("执行t-SNE降维...")
    X_tsne = perform_tsne(X_pca, n_components=2)
    
    # %% 科研风格可视化
    print("\n4. 生成科研风格可视化...")
    
    # 创建图片保存目录
    figures_dir = os.path.join(cfg.DATA_PATH if hasattr(cfg, 'DATA_PATH') else 'results', 'manifold_results')
    os.makedirs(figures_dir, exist_ok=True)
    
    # 生成综合分析图
    scientific_plot_path = os.path.join(figures_dir, 'tsne_scientific_analysis.png')
    category_stats = plot_tsne_scientific(
        X_tsne, y, stimulus_data, 
        save_path=scientific_plot_path,
        title="Neural Population t-SNE Analysis"
    )
    
    # 生成论文级主图
    publication_plot_path = os.path.join(figures_dir, 'tsne_publication_figure.png')
    plot_tsne_publication_figure(
        X_tsne, y, stimulus_data,
        save_path=publication_plot_path
    )
    
    # %% 量化分析
    print("\n5. t-SNE结果量化分析...")
    tsne_metrics = calculate_tsne_metrics(X_tsne, y)
    
    # %% 保存分析结果和数据
    print("\n6. 保存分析结果...")
    
    # 创建结果保存目录
    results_dir = os.path.join(cfg.DATA_PATH if hasattr(cfg, 'DATA_PATH') else 'results', 'manifold_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存t-SNE结果数据
    save_manifold_2d_data(X_tsne, y, stimulus_data, title="t-SNE Neural Manifold", save_dir=results_dir)
    
    # 保存分析统计信息
    analysis_stats = {
        # PCA预处理信息
        'pca_explained_variance_ratio': pca.explained_variance_ratio_,
        'pca_total_explained_variance': np.sum(pca.explained_variance_ratio_),
        'pca_n_components': X_pca.shape[1],
        
        # t-SNE结果信息
        'tsne_n_components': X_tsne.shape[1],
        'tsne_perplexity': mfg.TSNE_PERPLEXITY,
        'tsne_learning_rate': mfg.TSNE_LEARNING_RATE,
        'tsne_max_iter': mfg.TSNE_MAX_ITER,
        
        # 数据信息
        'original_features': X.shape[1],
        'n_samples': X.shape[0],
        'n_categories': len(np.unique(y)),
        'categories': np.unique(y).tolist(),
        'category_counts': np.unique(y, return_counts=True)[1].tolist(),
        
        # 神经元信息
        'rr_neurons': rr_neurons if rr_neurons is not None else 'all',
        'n_rr_neurons': len(rr_neurons) if rr_neurons is not None else X.shape[1],
        'used_rr_only': rr_neurons is not None,
        
        # 量化指标
        'tsne_metrics': tsne_metrics,
        'category_statistics': category_stats
    }
    
    np.savez_compressed(
        os.path.join(results_dir, 'tsne_analysis_complete.npz'),
        X_original=X,
        X_pca=X_pca,
        X_tsne=X_tsne,
        labels=y,
        stimulus_data=stimulus_data,
        **analysis_stats
    )
    
    # 保存简化的t-SNE数据（向后兼容）
    save_data_for_cebra(X_tsne, y, os.path.join(results_dir, 'tsne_data.npz'))
    
    print("t-SNE分析统计信息已保存")
    
    # %% 生成文本报告
    print("\n7. 生成分析报告...")
    
    report_path = os.path.join(results_dir, 'tsne_analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("神经群体t-SNE降维分析报告\n")
        f.write("="*60 + "\n\n")
        
        f.write("数据概览:\n")
        f.write(f"- 原始数据维度: {X.shape[0]} 样本 × {X.shape[1]} 特征\n")
        f.write(f"- PCA预降维: {X_pca.shape[1]} 个主成分 (解释方差: {np.sum(pca.explained_variance_ratio_):.1%})\n")
        f.write(f"- t-SNE最终维度: 2维\n")
        f.write(f"- 类别数量: {len(np.unique(y))} ({np.unique(y).tolist()})\n")
        f.write(f"- 使用神经元: {'RR神经元' if rr_neurons is not None else '所有神经元'}\n\n")
        
        f.write("t-SNE参数:\n")
        f.write(f"- Perplexity: {mfg.TSNE_PERPLEXITY}\n")
        f.write(f"- Learning Rate: {mfg.TSNE_LEARNING_RATE}\n")
        f.write(f"- Max Iterations: {mfg.TSNE_MAX_ITER}\n\n")
        
        f.write("量化指标:\n")
        for key, value in tsne_metrics.items():
            f.write(f"- {key}: {value:.3f}\n")
        f.write("\n")
        
        f.write("类别统计:\n")
        for stat in category_stats:
            f.write(f"- 类别 {stat['category']}: {stat['n_points']} 个样本\n")
            f.write(f"  类内平均距离: {stat['intra_mean']:.3f} ± {stat['intra_std']:.3f}\n")
        f.write("\n")
        
        f.write("生成文件:\n")
        f.write(f"- 科研分析图: tsne_scientific_analysis.png\n")
        f.write(f"- 论文级主图: tsne_publication_figure.png\n")
        f.write(f"- 完整数据: tsne_analysis_complete.npz\n")
        f.write(f"- 简化数据: tsne_data.npz\n")
        
    print(f"分析报告已保存: {report_path}")
    
    print("\n=== t-SNE降维分析完成 ===")
    print(f"结果保存位置: {results_dir}")
    print("主要输出:")
    print(f"  - 科研风格综合分析图: {scientific_plot_path}")
    print(f"  - 论文级发表图: {publication_plot_path}")
    print(f"  - 完整分析数据: {os.path.join(results_dir, 'tsne_analysis_complete.npz')}")
    print(f"  - 分析报告: {report_path}")   