# 神经数据预处理
# guiy24@mails.tsinghua.edu.cn
# %% 导入必要的库
import h5py
import os
import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
from collections import Counter
import seaborn as sns
import warnings

# %% 参数定义
# ================================= 参数配置区域 =================================
class Config:
    """统一参数配置类"""
    
    # 数据加载器版本: 'new' 或 'old'
    # 'new': 使用 process_trigger 和 load_data 加载新版数据
    # 'old': 使用 load_old_version_data 加载旧版 .mat 数据
    LOADER_VERSION = 'new'

    # 新版数据路径
    DATA_PATH = r'F:\brain\Micedata\M65_0816'
    
    # 旧版数据路径 (仅在 LOADER_VERSION = 'old' 时使用)
    OLD_VERSION_PATHS = {
        'neurons': r'F:\brain\Micedata\M27_1008\Neurons.mat',
        'trials': r'F:\brain\Micedata\M27_1008\Trial_data.mat',
        'location': r'F:\brain\Micedata\M27_1008\wholebrain_output.mat'
    }

    # 触发文件处理参数
    IPD = 5                    # 刺激呈现时长(s)
    ISI = 5                    # 刺激间隔(s)
    MIN_STI_GAP = 5           # 相邻刺激最小间隔(s)
    
    # 数据分割参数
    PRE_FRAMES = 10           # 刺激前帧数（基线期）
    POST_FRAMES = 40          # 刺激后帧数（响应期）
    STIMULUS_DURATION = 20    # 刺激持续时间（帧数）
    BASELINE_CORRECT = False  # 是否进行基线校正
    
    # 标签重分类参数
    NOISE_INTENSITY = 1       # 噪音刺激强度标识
    
    # RR神经元筛选参数
    T_STIMULUS = PRE_FRAMES   # 刺激开始时间点
    L_STIMULUS = STIMULUS_DURATION  # 刺激窗口长度
    ALPHA_FDR = 0.05        # FDR校正阈值
    ALPHA_LEVEL = 0.05       # 显著性水平
    RELIABILITY_THRESHOLD = 0.5  # 可靠性阈值
    
    # 快速RR筛选参数
    EFFECT_SIZE_THRESHOLD = 0.5   # 效应大小阈值
    SNR_THRESHOLD = 0.9          # 信噪比阈值
    RESPONSE_RATIO_THRESHOLD = 0.4  # 响应比例阈值
    # SNR_THRESHOLD = 0.9          # 信噪比阈值
    # RESPONSE_RATIO_THRESHOLD = 0.6  # 响应比例阈值
    # 分类参数
    TEST_SIZE = 0.3           # 测试集比例
    RANDOM_STATE = 42         # 随机种子
    CV_FOLDS = 5              # 交叉验证折数
    SVM_KERNEL = 'rbf'        # SVM核函数
    SVM_C = 1.0               # SVM正则化参数
    SVM_GAMMA = 'scale'       # SVM gamma参数
    
    # 特征提取参数
    USE_STIMULUS_PERIOD_ONLY = True  # 是否只使用刺激期数据
    
    # 可视化参数
    FIGURE_SIZE_LARGE = (50, 6)   # 大图尺寸
    FIGURE_SIZE_MEDIUM = (15, 5)  # 中图尺寸
    FIGURE_SIZE_SMALL = (8, 6)    # 小图尺寸
    FIGURE_SIZE_TINY = (6, 3)     # 迷你图尺寸
    
    # 数据处理阈值
    NEURON_THRESHOLD = 1000   # 使用原始RR方法的神经元数量阈值
    
    # 试验范围（用于去掉首尾）
    TRIAL_START_SKIP = 0     # 跳过开头的试验数
    TRIAL_END_SKIP = 0      # 跳过结尾的试验数
    TOTAL_TRIALS = 176      # 保持的试验总数
    
    # 预处理参数
    ENABLE_PREPROCESSING = True      # 是否启用预处理
    ENABLE_CLASS_BALANCE = True      # 是否启用类别平衡
    PREPROCESSING_METHOD = 'comprehensive'  # 预处理方法: 'simple', 'comprehensive'
    
    # 特征选择参数
    MAX_FEATURES = 500               # 特征选择最大特征数
    PCA_COMPONENTS = 100             # PCA降维目标维度
    VARIANCE_THRESHOLD = 0.01        # 方差过滤阈值
    
    # 类别平衡参数  
    BALANCE_METHOD = 'smote'         # 类别平衡方法: 'smote', 'random_oversample'
    
    # 信号处理参数
    GAUSSIAN_SIGMA = 0.8             # 高斯滤波标准差
    
    # 分类器参数
    ENABLE_MULTIPLE_CLASSIFIERS = True  # 是否测试多种分类器
    
    # 可视化配置参数
    VISUALIZATION_DPI = 300          # 图像分辨率
    VISUALIZATION_STYLE = 'seaborn-v0_8-whitegrid'  # 科研绘图风格
    COLOR_PALETTE = 'Set2'           # 配色方案
    PLOT_COLORS = {
        'primary': '#2E86AB',        # 主要颜色（蓝色）
        'secondary': '#A23B72',      # 次要颜色（紫色）
        'accent': '#F18F01',         # 强调色（橙色）
        'success': '#C73E1D',        # 成功色（红色）
        'neutral': '#6C757D',        # 中性色（灰色）
        'background': '#F8F9FA'      # 背景色（浅灰）
    }
    
    # 时间可视化参数
    TIME_UNIT = 'frames'             # 时间单位标签
    BASELINE_COLOR = '#95A5A6'       # 基线期颜色
    STIMULUS_COLOR = '#E74C3C'       # 刺激期颜色  
    RESPONSE_COLOR = '#3498DB'       # 响应期颜色

# 全局配置实例
cfg = Config()

# ===============================================================================
# %% 处理触发文件
# 从matlab改过来的，经过检查应该无误
def process_trigger(txt_file, IPD=cfg.IPD, ISI=cfg.ISI, fre=None, min_sti_gap=cfg.MIN_STI_GAP):
    """
    处理触发文件，修改自step1x_trigger_725right.m
    
    参数:
    txt_file: str, txt文件路径
    IPD: float, 刺激呈现时长(s)，默认2s
    ISI: float, 刺激间隔(s)，默认6s
    fre: float, 相机帧率Hz，None则从相机触发时间自动估计
    min_sti_gap: float, 相邻刺激"2"小于此间隔(s)视作同一次（用于去重合并），默认5s
    
    返回:
    dict: 包含start_edge, end_edge, stimuli_array的字典
    """
    
    # 读入文件
    data = []
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    time_val = float(parts[0])
                    ch_str = parts[1]
                    abs_ts = float(parts[2]) if len(parts) >= 3 else None
                    data.append((time_val, ch_str, abs_ts))
                except ValueError:
                    continue
    
    if not data:
        raise ValueError("未能从文件中读取到有效数据")
    
    # 解析数据
    times, channels, abs_timestamps = zip(*data)
    times = np.array(times)
    
    # 转换通道为数值，非数值的设为NaN
    ch_numeric = []
    valid_indices = []
    for i, ch_str in enumerate(channels):
        try:
            ch_val = float(ch_str)
            ch_numeric.append(ch_val)
            valid_indices.append(i)
        except ValueError:
            continue
    
    if not valid_indices:
        raise ValueError("未找到有效的数值通道数据")
    
    # 只保留有效数据
    t = times[valid_indices]
    ch = np.array(ch_numeric)
    
    # 相机帧与刺激起始时间
    cam_t_raw = t[ch == 1]
    sti_t_raw = t[ch == 2]
    
    if len(cam_t_raw) == 0:
        raise ValueError("未检测到相机触发(值=1)")
    if len(sti_t_raw) == 0:
        raise ValueError("未检测到刺激触发(值=2)")
    
    # 去重/合并：将时间靠得很近的"2"视作同一次刺激
    sti_t = np.sort(sti_t_raw)
    if len(sti_t) > 0:
        keep = np.ones(len(sti_t), dtype=bool)
        for i in range(1, len(sti_t)):
            if (sti_t[i] - sti_t[i-1]) < min_sti_gap:
                keep[i] = False  # 合并到前一个
        sti_t = sti_t[keep]
    
    # 帧率估计或使用给定值
    if fre is None:
        dt = np.diff(cam_t_raw)
        fre = 1 / np.median(dt)  # 用相机帧时间戳的中位间隔
        print(f'自动估计相机帧率：{fre:.4f} Hz')
    else:
        print(f'使用给定相机帧率：{fre:.4f} Hz')
    
    IPD_frames = max(1, round(IPD * fre))
    isi_frames = round((IPD + ISI) * fre)
    
    # 把每个刺激时间映射到最近的相机帧索引
    cam_t = cam_t_raw.copy()
    nFrames = len(cam_t)
    start_edge = np.zeros(len(sti_t), dtype=int)
    
    for k in range(len(sti_t)):
        idx = np.argmin(np.abs(cam_t - sti_t[k]))
        start_edge[k] = idx
    
    end_edge = start_edge + IPD_frames - 1
    
    # 边界裁剪，避免越界
    valid = (start_edge >= 0) & (end_edge < nFrames) & (start_edge <= end_edge)
    start_edge = start_edge[valid]
    end_edge = end_edge[valid]
    
    # 尾段完整性检查（与旧逻辑一致）
    if len(start_edge) >= 2:
        d = np.diff(start_edge)
        while len(d) > 0 and d[-1] not in [isi_frames-1, isi_frames, isi_frames+1, isi_frames+2]:
            # 丢掉最后一个可疑的刺激段
            start_edge = start_edge[:-1]
            end_edge = end_edge[:-1]
            if len(start_edge) >= 2:
                d = np.diff(start_edge)
            else:
                break
    
    # 生成0/1刺激数组（可视化/保存用）
    stimuli_array = np.zeros(nFrames)
    for i in range(len(start_edge)):
        stimuli_array[start_edge[i]:end_edge[i]+1] = 1
    
    # 保存结果到mat文件
    save_path = os.path.join(os.path.dirname(txt_file), 'visual_stimuli_with_label.mat')
    scipy.io.savemat(save_path, {
        'start_edge': start_edge,
        'end_edge': end_edge,
        'stimuli_array': stimuli_array
    })
    
    return {
        'start_edge': start_edge,
        'end_edge': end_edge,
        'stimuli_array': stimuli_array,
        'camera_frames': len(cam_t),
        'stimuli_count': len(start_edge)
    }

# %% 加载数据 (新版)
def load_data(data_path, start_idx=cfg.TRIAL_START_SKIP, end_idx=cfg.TRIAL_START_SKIP + cfg.TOTAL_TRIALS, interactive = True):
    ######### 读取神经数据 #########
    if interactive:
        print("start loading neuron data...")
    mat_file = os.path.join(data_path, 'wholebrain_output.mat')
    if not os.path.exists(mat_file):
        raise ValueError(f"未找到神经数据文件: {mat_file}")
    try:
        data = h5py.File(mat_file, 'r')
    except Exception as e:
        raise ValueError(f"无法读取mat文件: {mat_file}，错误信息: {e}")

    # 检查关键数据集是否存在
    if 'whole_trace_ori' not in data or 'whole_center' not in data:
        raise ValueError("mat文件缺少必要的数据集（'whole_trace_ori' 或 'whole_center'）")

    # 神经数据
    neuron_data = data['whole_trace_ori']
    # 转化成numpy数组
    neuron_data = np.array(neuron_data)
    print(f"原始神经数据形状: {neuron_data.shape}")
    
    # 移除全局标准化，改为在基线校正中处理
    # 只做基本的数据清理：移除NaN和Inf
    neuron_data = np.nan_to_num(neuron_data, nan=0.0, posinf=0.0, neginf=0.0)
    neuron_pos = data['whole_center']
    # 检查neuron_pos维度
    if len(neuron_pos.shape) != 2 or neuron_pos.shape[0] != 4:
        raise ValueError(f"neuron_pos 维度有误，期望为(3, n)，实际为: {neuron_pos.shape}")

    # 提取空间坐标前两维
    neuron_pos = neuron_pos[0:2, :]
    if interactive:
        print("neuron data loaded successfully!")
        print(f"Extracted {neuron_data.shape[1]} neurons, total time steps: {neuron_data.shape[0]}")

    ######### 读取触发文件 #########
    if interactive:
        print("start loading trigger data...")
    trigger_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.txt')]
    trigger_data = process_trigger(trigger_files[0])
    
    if interactive:
        print("trigger data loaded successfully!")
        print(f"Extracted {trigger_data['stimuli_count']} stimuli, total time steps: {trigger_data['camera_frames']}")

    ######### 读取刺激文件 #########
    if interactive:
        print("start loading stimulus data...")
    stimulus_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]
    stimulus_data = pd.read_csv(stimulus_files[0])
    # 转化成numpy数组
    stimulus_data = np.array(stimulus_data)
    
    # 保持指定试验数，去掉首尾 - 对触发数据和刺激数据同时处理
    start_edges = trigger_data['start_edge'][start_idx:end_idx]
    stimulus_data = stimulus_data[0:end_idx - start_idx, :]

    if interactive:
        print("stimulus data loaded successfully!")
        print(f"Using trials {start_idx} to {end_idx-1}, total: {len(start_edges)} trials")
    
    return neuron_data, neuron_pos, start_edges, stimulus_data

# %% 加载数据 (旧版)
def load_old_version_data(neurons_mat_path, trials_mat_path, location_mat_path):
    """
    读取旧版 .mat 格式的神经元数据、试次数据和位置数据。
    代码逻辑源自 code_old_version.py。
    """
    print("--- 开始加载旧版数据 ---")
    
    # 1. 读取神经元索引和预分割的试次数据
    print(f"正在读取: {neurons_mat_path} 和 {trials_mat_path}")
    try:
        neuron_data_mat = scipy.io.loadmat(neurons_mat_path)
        neuron_index = neuron_data_mat['rr_neurons'].flatten()
        
        trial_data_mat = scipy.io.loadmat(trials_mat_path)
        trials = trial_data_mat['trials']
        labels = trial_data_mat['labels'].ravel()
        
        print(f"数据加载完成：载入 {trials.shape[0]} 个试次，每个试次包含 {trials.shape[1]} 个神经元，时间点为 {trials.shape[2]}。")
        print(f"其中，预筛选出的神经元索引数量为: {len(neuron_index)}")
    except Exception as e:
        raise ValueError(f"读取神经元或试次 .mat 文件失败: {e}")

    # 2. 读取神经元位置信息
    print(f"正在读取位置数据: {location_mat_path}")
    try:
        with h5py.File(location_mat_path, 'r') as f:
            # 检查 'whole_center' 是否存在
            if 'whole_center' not in f:
                raise ValueError("mat文件中缺少必要的数据集 'whole_center'")
            location = np.array(f['whole_center'])
        print(f"位置数据加载完成：找到 {location.shape[1]} 个神经元的坐标。")
    except Exception as e:
        raise ValueError(f"读取位置 .mat 文件失败: {e}")
        
    print("--- 旧版数据加载成功 ---")
    return neuron_index, trials, labels, location

# %% 预处理函数
def preprocess_neural_data(segments, labels, method='comprehensive'):
    """
    神经数据预处理pipeline
    
    参数:
    segments: 神经数据片段 (trials, neurons, timepoints)
    labels: 标签数组
    method: 预处理方法 ('simple' 或 'comprehensive')
    
    返回:
    processed_data: 预处理后的特征矩阵
    processed_labels: 处理后的标签
    feature_info: 特征信息字典
    """
    warnings.filterwarnings('ignore')
    print("开始神经数据预处理...")
    
    # 过滤有效数据
    valid_mask = labels != 0
    valid_segments = segments[valid_mask]
    valid_labels = labels[valid_mask]
    
    print(f"过滤前试次数: {len(labels)}")
    print(f"过滤后试次数: {len(valid_labels)}")
    print(f"标签分布: {Counter(valid_labels)}")
    
    # 1. 提取特征：使用刺激期数据
    stimulus_period = np.arange(cfg.T_STIMULUS, min(cfg.T_STIMULUS + cfg.L_STIMULUS, valid_segments.shape[2]))
    stimulus_data = valid_segments[:, :, stimulus_period]  # (trials, neurons, stimulus_timepoints)
    
    # 提取基线用于normalization
    baseline_period = np.arange(0, cfg.T_STIMULUS)
    if len(baseline_period) > 0:
        baseline_data = valid_segments[:, :, baseline_period]
        baseline_mean = np.mean(baseline_data, axis=2, keepdims=True)
        baseline_std = np.std(baseline_data, axis=2, keepdims=True) + 1e-8
        
        # 避免除零和NaN错误
        baseline_mean = np.where((baseline_mean == 0) | np.isnan(baseline_mean), 1e-6, baseline_mean)
        
        # dF/F 归一化
        stimulus_data = (stimulus_data - baseline_mean) / baseline_mean
        stimulus_data = np.nan_to_num(stimulus_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 展平为特征矩阵
    X = stimulus_data.reshape(stimulus_data.shape[0], -1)  # (trials, neurons * timepoints)
    y = valid_labels
    
    print(f"原始特征维度: {X.shape}")
    
    if method == 'simple':
        # 简单预处理：只做标准化
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X)
        
        feature_info = {
            'method': 'simple',
            'scaler': scaler,
            'original_shape': X.shape,
            'final_shape': X_processed.shape
        }
        
    elif method == 'comprehensive':
        # 综合预处理pipeline
        
        # 2. 噪声处理 - 对每个trial的时序数据做平滑
        print("2. 噪声处理...")
        X_denoised = np.zeros_like(X)
        n_timepoints = len(stimulus_period)
        
        for i in range(X.shape[0]):
            trial_matrix = X[i].reshape(-1, n_timepoints)  # (neurons, timepoints)
            for neuron_idx in range(trial_matrix.shape[0]):
                trial_matrix[neuron_idx, :] = gaussian_filter1d(
                    trial_matrix[neuron_idx, :], sigma=cfg.GAUSSIAN_SIGMA)
            X_denoised[i] = trial_matrix.flatten()
        
        X = X_denoised
        
        # 3. 方差过滤
        print("3. 方差过滤...")
        variance_selector = VarianceThreshold(threshold=cfg.VARIANCE_THRESHOLD)
        X = variance_selector.fit_transform(X)
        print(f"   方差过滤后维度: {X.shape[1]}")
        
        # 4. 数据标准化
        print("4. 数据标准化...")
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 5. 特征选择
        print("5. 特征选择...")
        n_features = min(cfg.MAX_FEATURES, X_scaled.shape[1] // 2)
        if n_features < X_scaled.shape[1]:
            selector = SelectKBest(score_func=f_classif, k=n_features)
            X_selected = selector.fit_transform(X_scaled, y)
            print(f"   特征选择后维度: {X_selected.shape[1]}")
        else:
            selector = None
            X_selected = X_scaled
            print("   跳过特征选择（特征数已较小）")
        
        # 6. PCA降维
        print("6. PCA降维...")
        n_components = min(cfg.PCA_COMPONENTS, X_selected.shape[1], len(y)-1)
        if n_components < X_selected.shape[1] and n_components > 0:
            pca = PCA(n_components=n_components, random_state=cfg.RANDOM_STATE)
            X_pca = pca.fit_transform(X_selected)
            explained_ratio = np.sum(pca.explained_variance_ratio_)
            print(f"   PCA后维度: {X_pca.shape[1]}, 解释方差比: {explained_ratio:.3f}")
        else:
            pca = None
            X_pca = X_selected
            explained_ratio = 1.0
            print("   跳过PCA（维度已适中）")
        
        # 7. 最终标准化
        X_processed = zscore(X_pca, axis=0)
        X_processed = np.nan_to_num(X_processed, 0)
        
        feature_info = {
            'method': 'comprehensive',
            'variance_selector': variance_selector,
            'scaler': scaler,
            'feature_selector': selector,
            'pca': pca,
            'explained_variance_ratio': explained_ratio,
            'original_shape': stimulus_data.shape,
            'final_shape': X_processed.shape,
            'n_features_selected': n_features if selector else X_scaled.shape[1],
            'n_components': n_components if pca else X_selected.shape[1]
        }
    
    print(f"预处理完成！最终特征维度: {X_processed.shape}")
    return X_processed, y, feature_info

def handle_class_imbalance(X, y, method='smote'):
    """
    处理类别不平衡问题
    """
    try:
        from imblearn.over_sampling import SMOTE, RandomOverSampler
    except ImportError:
        print("警告: imbalanced-learn未安装，跳过类别平衡")
        return X, y
    
    print(f"原始类别分布: {Counter(y)}")
    
    try:
        if method == 'smote':
            # 确保有足够的近邻样本
            min_samples = min(Counter(y).values())
            k_neighbors = min(3, min_samples - 1) if min_samples > 1 else 1
            sampler = SMOTE(random_state=cfg.RANDOM_STATE, k_neighbors=k_neighbors)
        else:
            sampler = RandomOverSampler(random_state=cfg.RANDOM_STATE)
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        print(f"重采样后类别分布: {Counter(y_resampled)}")
        return X_resampled, y_resampled
        
    except Exception as e:
        print(f"类别平衡失败: {e}, 使用原始数据")
        return X, y

def improved_classification(X, y, test_size=0.3, enable_multiple=True):
    """
    改进的分类流程，测试多种分类器
    """
    print("开始改进的分类测试...")
    
    # 分层划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=cfg.RANDOM_STATE, stratify=y)
    
    print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")
    
    # 计算类权重
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    print(f"类权重: {class_weight_dict}")
    
    if not enable_multiple:
        # 只使用SVM
        clf = SVC(kernel=cfg.SVM_KERNEL, C=cfg.SVM_C, gamma=cfg.SVM_GAMMA, 
                 random_state=cfg.RANDOM_STATE, class_weight='balanced')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(clf, X, y, cv=cfg.CV_FOLDS, 
                                  scoring='accuracy')
        
        print(f"SVM测试准确率: {accuracy:.3f}")
        print(f"SVM交叉验证: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"分类报告:\n{classification_report(y_test, y_pred)}")
        
        return {
            'best_model': 'SVM',
            'best_accuracy': accuracy,
            'best_cv_mean': cv_scores.mean(),
            'best_cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    # 多分类器比较
    classifiers = {
        'SVM': SVC(kernel='rbf', class_weight='balanced', random_state=cfg.RANDOM_STATE, probability=True),
        'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', 
                                             random_state=cfg.RANDOM_STATE, max_depth=10),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=cfg.RANDOM_STATE),
        'LogisticRegression': LogisticRegression(class_weight='balanced', random_state=cfg.RANDOM_STATE, max_iter=1000)
    }
    
    results = {}
    cv = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)
    
    for name, clf in classifiers.items():
        print(f"\n=== {name} ===")
        
        # 训练
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # 评估
        test_acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        
        print(f"测试准确率: {test_acc:.3f}")
        print(f"交叉验证: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        results[name] = {
            'test_accuracy': test_acc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    # 找出最佳模型
    best_model = max(results.keys(), key=lambda k: results[k]['cv_mean'])
    print(f"\n最佳模型: {best_model} (CV准确率: {results[best_model]['cv_mean']:.3f})")
    
    return {
        'results': results,
        'best_model': best_model,
        'best_accuracy': results[best_model]['test_accuracy'],
        'best_cv_mean': results[best_model]['cv_mean'],
        'best_cv_std': results[best_model]['cv_std']
    }

# %% ========== 可视化函数 ========== 

def setup_plot_style():
    """设置科研绘图风格"""
    plt.style.use(cfg.VISUALIZATION_STYLE)
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'font.family': 'Arial',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.2,
        'axes.edgecolor': '#2C3E50',
        'grid.alpha': 0.3
    })

def visualize_classification_performance(results_dict, save_path=None):
    """
    Visualization of classification performance across different models
    
    Parameters:
    results_dict: Dictionary containing classification results
    save_path: Path to save the figure
    """
    setup_plot_style()
    
    if 'results' in results_dict:
        # Multiple classifier results
        models = list(results_dict['results'].keys())
        accuracies = [results_dict['results'][model]['cv_mean'] for model in models]
        stds = [results_dict['results'][model]['cv_std'] for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar plot of accuracies
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        bars = ax1.bar(models, accuracies, yerr=stds, capsize=5, 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        ax1.set_ylabel('Cross-validation Accuracy')
        ax1.set_title('Model Performance Comparison')
        ax1.set_ylim(0, max(accuracies) * 1.2)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc, std in zip(bars, accuracies, stds):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                    f'{acc:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Confusion matrix of best model
        best_model = results_dict['best_model']
        cm = results_dict['results'][best_model]['confusion_matrix']
        
        im = ax2.imshow(cm, interpolation='nearest', cmap='Blues', alpha=0.8)
        ax2.figure.colorbar(im, ax=ax2, shrink=0.6)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax2.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontweight='bold')
        
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        ax2.set_title(f'Confusion Matrix - {best_model}')
        ax2.set_xticks(range(len(np.unique(cm))))
        ax2.set_yticks(range(len(np.unique(cm))))
        ax2.set_xticklabels([f'Class {i+1}' for i in range(len(np.unique(cm)))])
        ax2.set_yticklabels([f'Class {i+1}' for i in range(len(np.unique(cm)))])
        
    else:
        # Single model results
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        cm = results_dict['confusion_matrix']
        
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues', alpha=0.8)
        fig.colorbar(im, ax=ax, shrink=0.8)
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontweight='bold')
        
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title(f'Classification Results\nAccuracy: {results_dict["best_cv_mean"]:.3f}±{results_dict["best_cv_std"]:.3f}')
        ax.set_xticks(range(cm.shape[1]))
        ax.set_yticks(range(cm.shape[0]))
        ax.set_xticklabels([f'Class {i+1}' for i in range(cm.shape[1])])
        ax.set_yticklabels([f'Class {i+1}' for i in range(cm.shape[0])])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=cfg.VISUALIZATION_DPI, bbox_inches='tight')
        print(f"Classification performance plot saved to: {save_path}")
    plt.show()

def visualize_roc_curves(X, y, models_dict, save_path=None):
    """
    Visualize ROC curves for multiple classifiers (binary or multiclass)
    
    Parameters:
    X: Feature matrix
    y: Labels
    models_dict: Dictionary of trained models
    save_path: Path to save the figure
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    # 移除不必要的scipy.interp导入，该函数中未使用
    from itertools import cycle
    
    setup_plot_style()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=cfg.RANDOM_STATE, stratify=y)
    
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    
    if n_classes == 2:
        # Binary classification
        fig, ax = plt.subplots(figsize=(8, 8))
        
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'])
        
        for (name, model), color in zip(models_dict.items(), colors):
            model.fit(X_train, y_train)
            if hasattr(model, "decision_function"):
                y_score = model.decision_function(X_test)
            else:
                y_score = model.predict_proba(X_test)[:, 1]
            
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=color, lw=2,
                   label=f'{name} (AUC = {roc_auc:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curves')
        ax.legend(loc="lower right")
        
    else:
        # Multiclass classification - show one-vs-rest ROC
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Binarize the output
        y_test_bin = label_binarize(y_test, classes=unique_classes)
        
        # Plot ROC curve for each class for the best model
        best_model_name = list(models_dict.keys())[0]  # Use first model
        best_model = models_dict[best_model_name]
        best_model.fit(X_train, y_train)
        y_score = best_model.predict_proba(X_test)
        
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'])
        
        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2,
                   label=f'Class {unique_classes[i]} (AUC = {roc_auc:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves - {best_model_name} (One-vs-Rest)')
        ax.legend(loc="lower right")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=cfg.VISUALIZATION_DPI, bbox_inches='tight')
        print(f"ROC curves saved to: {save_path}")
    plt.show()

def visualize_neural_activity_heatmap(neuron_data, title="Neural Activity Heatmap", 
                                     max_neurons=50, save_path=None):
    """
    Visualize neural activity as a professional heatmap
    
    Parameters:
    neuron_data: Neural data array (timepoints, neurons)
    title: Plot title
    max_neurons: Maximum number of neurons to display
    save_path: Path to save the figure
    """
    setup_plot_style()
    
    # Select neurons for display
    n_neurons_display = min(max_neurons, neuron_data.shape[1])
    if n_neurons_display < neuron_data.shape[1]:
        selected_neurons = np.linspace(0, neuron_data.shape[1]-1, n_neurons_display, dtype=int)
    else:
        selected_neurons = np.arange(n_neurons_display)
    
    # Select time points for display  
    max_timepoints = 800
    n_timepoints_display = min(max_timepoints, neuron_data.shape[0])
    time_indices = np.linspace(0, neuron_data.shape[0]-1, n_timepoints_display, dtype=int)
    
    # Create heatmap data
    heatmap_data = neuron_data[np.ix_(time_indices, selected_neurons)].T
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
    
    # Main heatmap
    im = ax1.imshow(heatmap_data, aspect='auto', cmap='viridis', 
                    interpolation='nearest', alpha=0.9)
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('Neural Activity (ΔF/F)', rotation=270, labelpad=20)
    
    ax1.set_xlabel(f'Time ({cfg.TIME_UNIT})')
    ax1.set_ylabel('Neurons')
    ax1.set_title(f'{title}\nDisplaying {n_neurons_display} neurons over {n_timepoints_display} time points')
    
    # Set ticks
    n_ticks_time = 8
    time_tick_indices = np.linspace(0, len(time_indices)-1, n_ticks_time, dtype=int)
    ax1.set_xticks(time_tick_indices)
    ax1.set_xticklabels([f'{time_indices[i]}' for i in time_tick_indices])
    
    n_ticks_neurons = min(8, n_neurons_display)
    neuron_tick_indices = np.linspace(0, n_neurons_display-1, n_ticks_neurons, dtype=int)
    ax1.set_yticks(neuron_tick_indices)
    ax1.set_yticklabels([f'N{selected_neurons[i]}' for i in neuron_tick_indices])
    
    # Activity profile over time
    mean_activity = np.mean(heatmap_data, axis=0)
    ax2.plot(range(len(mean_activity)), mean_activity, 
             linewidth=2, color=cfg.PLOT_COLORS['primary'], alpha=0.8)
    ax2.fill_between(range(len(mean_activity)), mean_activity, 
                     alpha=0.3, color=cfg.PLOT_COLORS['primary'])
    
    ax2.set_xlabel(f'Time ({cfg.TIME_UNIT})')
    ax2.set_ylabel('Mean Activity')
    ax2.set_title('Average Neural Activity Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Set x-ticks for activity profile
    ax2.set_xticks(time_tick_indices)
    ax2.set_xticklabels([f'{time_indices[i]}' for i in time_tick_indices])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=cfg.VISUALIZATION_DPI, bbox_inches='tight')
        print(f"Neural activity heatmap saved to: {save_path}")
    plt.show()

def visualize_trigger_distribution(trigger_data, title="Stimulus Trigger Distribution", save_path=None):
    """
    Visualize stimulus trigger temporal distribution with professional styling
    
    Parameters:
    trigger_data: Array of trigger time points
    title: Plot title
    save_path: Path to save the figure
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Trigger interval distribution
    ax1 = axes[0, 0]
    if len(trigger_data) > 1:
        intervals = np.diff(trigger_data)
        n_bins = min(20, len(intervals)//3)
        
        counts, bins, patches = ax1.hist(intervals, bins=n_bins, alpha=0.7, 
                                        color=cfg.PLOT_COLORS['primary'], 
                                        edgecolor='black', linewidth=1.2)
        
        # Add statistics
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        ax1.axvline(mean_interval, color=cfg.PLOT_COLORS['accent'], linestyle='--', 
                   linewidth=2, alpha=0.8, label=f'Mean: {mean_interval:.1f}')
        
        ax1.set_xlabel(f'Inter-trigger Interval ({cfg.TIME_UNIT})')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Trigger Interval Distribution\nMean: {mean_interval:.1f} ± {std_interval:.1f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Trigger timing over experiment
    ax2 = axes[0, 1]
    ax2.scatter(trigger_data, np.ones_like(trigger_data), 
               s=50, c=cfg.PLOT_COLORS['secondary'], alpha=0.7, 
               edgecolors='black', linewidth=0.5)
    ax2.set_xlabel(f'Time ({cfg.TIME_UNIT})')
    ax2.set_ylabel('Trigger Events')
    ax2.set_title(f'Trigger Time Points\nTotal: {len(trigger_data)} triggers')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.5, 1.5])
    ax2.set_yticks([1])
    ax2.set_yticklabels(['Triggers'])
    
    # 3. Cumulative trigger count
    ax3 = axes[1, 0]
    cumulative_count = np.arange(1, len(trigger_data) + 1)
    ax3.plot(trigger_data, cumulative_count, linewidth=2.5, 
            color=cfg.PLOT_COLORS['success'], alpha=0.8, marker='o', markersize=3)
    ax3.fill_between(trigger_data, cumulative_count, alpha=0.3, 
                    color=cfg.PLOT_COLORS['success'])
    
    ax3.set_xlabel(f'Time ({cfg.TIME_UNIT})')
    ax3.set_ylabel('Cumulative Trigger Count')
    ax3.set_title('Cumulative Trigger Timeline')
    ax3.grid(True, alpha=0.3)
    
    # 4. Trigger rate over time (sliding window)
    ax4 = axes[1, 1]
    if len(trigger_data) > 5:
        window_size = max(5, len(trigger_data) // 10)
        trigger_rates = []
        window_centers = []
        
        for i in range(window_size, len(trigger_data) - window_size):
            window_triggers = trigger_data[i-window_size:i+window_size]
            if len(window_triggers) > 1:
                rate = len(window_triggers) / (window_triggers[-1] - window_triggers[0])
                trigger_rates.append(rate)
                window_centers.append(trigger_data[i])
        
        if trigger_rates:
            ax4.plot(window_centers, trigger_rates, linewidth=2, 
                    color=cfg.PLOT_COLORS['primary'], alpha=0.8, marker='s', markersize=4)
            ax4.set_xlabel(f'Time ({cfg.TIME_UNIT})')
            ax4.set_ylabel('Trigger Rate (Hz)')
            ax4.set_title('Trigger Rate Over Time')
            ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, y=0.98)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=cfg.VISUALIZATION_DPI, bbox_inches='tight')
        print(f"Trigger distribution plot saved to: {save_path}")
    plt.show()

def visualize_stimulus_data_distribution(stimulus_data, title="Stimulus Data Distribution", save_path=None):
    """
    Visualize stimulus data distribution with professional styling
    
    Parameters:
    stimulus_data: Stimulus data array (trials, features)
    title: Plot title
    save_path: Path to save the figure
    """
    setup_plot_style()
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1])
    
    # Extract data
    categories = stimulus_data[:, 0]
    intensities = stimulus_data[:, 1]
    
    # 1. Category distribution
    ax1 = fig.add_subplot(gs[0, 0])
    unique_cats, counts_cats = np.unique(categories, return_counts=True)
    colors_cat = plt.cm.Set2(np.linspace(0, 1, len(unique_cats)))
    
    bars1 = ax1.bar(unique_cats, counts_cats, alpha=0.8, 
                   color=colors_cat, edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bar, count in zip(bars1, counts_cats):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts_cats) * 0.01,
               str(count), ha='center', va='bottom', fontweight='bold')
    
    ax1.set_xlabel('Stimulus Category')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Category Distribution')
    ax1.grid(True, alpha=0.3)
    
    # 2. Intensity distribution
    ax2 = fig.add_subplot(gs[0, 1])
    unique_ints, counts_ints = np.unique(intensities, return_counts=True)
    colors_int = plt.cm.viridis(np.linspace(0, 1, len(unique_ints)))
    
    bars2 = ax2.bar(unique_ints, counts_ints, alpha=0.8,
                   color=colors_int, edgecolor='black', linewidth=1.2)
    
    for bar, count in zip(bars2, counts_ints):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts_ints) * 0.01,
               str(count), ha='center', va='bottom', fontweight='bold')
    
    ax2.set_xlabel('Stimulus Intensity')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Intensity Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 3. Category-Intensity combination
    ax3 = fig.add_subplot(gs[0, 2])
    combinations = list(zip(categories, intensities))
    unique_combs, counts_combs = np.unique(combinations, return_counts=True, axis=0)
    
    comb_labels = [f"C{int(c[0])}-I{c[1]}" for c in unique_combs]
    colors_comb = plt.cm.tab10(np.linspace(0, 1, len(comb_labels)))
    
    bars3 = ax3.bar(range(len(comb_labels)), counts_combs, alpha=0.8,
                   color=colors_comb, edgecolor='black', linewidth=1.2)
    
    for bar, count in zip(bars3, counts_combs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts_combs) * 0.01,
               str(count), ha='center', va='bottom', fontweight='bold')
    
    ax3.set_xlabel('Category-Intensity Combination')
    ax3.set_ylabel('Frequency')  
    ax3.set_title('Category-Intensity Distribution')
    ax3.set_xticks(range(len(comb_labels)))
    ax3.set_xticklabels(comb_labels, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Time series of categories
    ax4 = fig.add_subplot(gs[1, :])
    trial_indices = np.arange(len(categories))
    
    # Plot categories as colored segments
    for i, cat in enumerate(unique_cats):
        cat_mask = categories == cat
        cat_trials = trial_indices[cat_mask]
        ax4.scatter(cat_trials, np.full_like(cat_trials, cat), 
                   color=colors_cat[i], s=40, alpha=0.7, 
                   edgecolors='black', linewidth=0.5, label=f'Category {int(cat)}')
    
    ax4.set_xlabel('Trial Index')
    ax4.set_ylabel('Stimulus Category')
    ax4.set_title('Stimulus Category Sequence Over Trials')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_yticks(unique_cats)
    
    # 5. Intensity time series
    ax5 = fig.add_subplot(gs[2, :])
    for i, intensity in enumerate(unique_ints):
        int_mask = intensities == intensity
        int_trials = trial_indices[int_mask]
        ax5.scatter(int_trials, np.full_like(int_trials, intensity),
                   color=colors_int[i], s=40, alpha=0.7,
                   edgecolors='black', linewidth=0.5, label=f'Intensity {intensity}')
    
    ax5.set_xlabel('Trial Index')
    ax5.set_ylabel('Stimulus Intensity')
    ax5.set_title('Stimulus Intensity Sequence Over Trials')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    ax5.set_yticks(unique_ints)
    
    plt.suptitle(title, y=0.98)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=cfg.VISUALIZATION_DPI, bbox_inches='tight')
        print(f"Stimulus distribution plot saved to: {save_path}")
    plt.show()

def visualize_rr_neurons_spatial_distribution(neuron_pos, rr_results, save_path=None):
    """
    Visualize spatial distribution of RR neurons
    
    Parameters:
    neuron_pos: Neuron positions array (2, n_neurons)
    rr_results: RR neuron selection results
    save_path: Path to save the figure
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # All neurons spatial distribution
    ax1 = axes[0, 0]
    ax1.scatter(neuron_pos[0, :], neuron_pos[1, :], 
               c=cfg.PLOT_COLORS['neutral'], s=20, alpha=0.5, 
               edgecolors='black', linewidths=0.3, label='All Neurons')
    
    ax1.set_xlabel('X Position (μm)')
    ax1.set_ylabel('Y Position (μm)')
    ax1.set_title(f'All Neurons Spatial Distribution\n(N = {neuron_pos.shape[1]})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # RR neurons highlighted
    ax2 = axes[0, 1]
    ax2.scatter(neuron_pos[0, :], neuron_pos[1, :], 
               c=cfg.PLOT_COLORS['neutral'], s=15, alpha=0.3, 
               edgecolors='none', label='All Neurons')
    
    if len(rr_results['rr_neurons']) > 0:
        rr_pos = neuron_pos[:, rr_results['rr_neurons']]
        ax2.scatter(rr_pos[0, :], rr_pos[1, :], 
                   c=cfg.PLOT_COLORS['accent'], s=40, alpha=0.8,
                   edgecolors='black', linewidths=0.5, label='RR Neurons')
    
    ax2.set_xlabel('X Position (μm)')
    ax2.set_ylabel('Y Position (μm)')
    ax2.set_title(f'RR Neurons Spatial Distribution\n(N = {len(rr_results["rr_neurons"])})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # Response vs non-response neurons
    ax3 = axes[1, 0]
    non_response_neurons = list(set(range(neuron_pos.shape[1])) - set(rr_results['response_neurons']))
    
    if non_response_neurons:
        ax3.scatter(neuron_pos[0, non_response_neurons], neuron_pos[1, non_response_neurons],
                   c=cfg.PLOT_COLORS['neutral'], s=15, alpha=0.4,
                   edgecolors='none', label='Non-responsive')
    
    if len(rr_results['response_neurons']) > 0:
        resp_pos = neuron_pos[:, rr_results['response_neurons']]
        ax3.scatter(resp_pos[0, :], resp_pos[1, :],
                   c=cfg.PLOT_COLORS['primary'], s=25, alpha=0.7,
                   edgecolors='black', linewidths=0.3, label='Responsive')
    
    ax3.set_xlabel('X Position (μm)')
    ax3.set_ylabel('Y Position (μm)')
    ax3.set_title(f'Responsive vs Non-responsive Neurons\n(Responsive: {len(rr_results["response_neurons"])})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')
    
    # Summary statistics
    ax4 = axes[1, 1]
    categories = ['Total', 'Responsive', 'Reliable', 'RR Neurons']
    counts = [
        neuron_pos.shape[1],
        len(rr_results['response_neurons']),
        len(rr_results['reliable_neurons']),
        len(rr_results['rr_neurons'])
    ]
    
    colors = [cfg.PLOT_COLORS['neutral'], cfg.PLOT_COLORS['primary'], 
              cfg.PLOT_COLORS['secondary'], cfg.PLOT_COLORS['accent']]
    
    bars = ax4.bar(categories, counts, alpha=0.8, color=colors, 
                   edgecolor='black', linewidth=1.2)
    
    # Add percentage labels
    total_neurons = neuron_pos.shape[1]
    for bar, count in zip(bars, counts):
        percentage = (count / total_neurons) * 100 if total_neurons > 0 else 0
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts) * 0.01,
                f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    ax4.set_ylabel('Number of Neurons')
    ax4.set_title('Neuron Selection Summary')
    ax4.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle('RR Neurons Spatial Analysis', y=0.98)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=cfg.VISUALIZATION_DPI, bbox_inches='tight')
        print(f"RR neurons spatial distribution saved to: {save_path}")
    plt.show()

def save_neuron_activity_stats(neuron_data, trigger_data, save_dir='results'):
    """保存神经信号基本统计信息"""
    os.makedirs(save_dir, exist_ok=True)
    
    stats = {
        'n_neurons': neuron_data.shape[1],
        'n_timepoints': neuron_data.shape[0],
        'mean_activity': np.mean(neuron_data),
        'std_activity': np.std(neuron_data),
        'min_activity': np.min(neuron_data),
        'max_activity': np.max(neuron_data),
        'trigger_times': trigger_data,
        'n_triggers': len(trigger_data)
    }
    
    np.savez_compressed(
        os.path.join(save_dir, 'neuron_activity_stats.npz'),
        **stats
    )
    print(f"神经活动统计信息已保存到 {save_dir}/neuron_activity_stats.npz")

def save_single_neuron_stats(neuron_idx, trials_idx, segments, save_dir='results'):
    """保存单个神经元的统计信息"""
    os.makedirs(save_dir, exist_ok=True)
    
    neuron_data = segments[trials_idx, neuron_idx, :]
    mean_response = np.mean(neuron_data, axis=0)
    
    stats = {
        'neuron_idx': neuron_idx,
        'trials_used': trials_idx,
        'mean_response': mean_response,
        'std_response': np.std(neuron_data, axis=0),
        'trial_responses': neuron_data,
        'peak_time': np.argmax(mean_response),
        'peak_value': np.max(mean_response),
        'baseline_mean': np.mean(mean_response[:cfg.PRE_FRAMES]),
        'response_mean': np.mean(mean_response[cfg.PRE_FRAMES:cfg.PRE_FRAMES+cfg.STIMULUS_DURATION])
    }
    
    np.savez_compressed(
        os.path.join(save_dir, f'neuron_{neuron_idx}_stats.npz'),
        **stats
    )
    print(f"神经元 {neuron_idx} 统计信息已保存")

def save_rr_neurons_distribution(neuron_pos, rr_results, save_dir='results'):
    """保存RR神经元的空间分布统计"""
    os.makedirs(save_dir, exist_ok=True)
    
    distribution_stats = {
        'total_neurons': neuron_pos.shape[1],
        'neuron_positions': neuron_pos,
        'response_neurons': rr_results['response_neurons'],
        'rr_neurons': rr_results['rr_neurons'],
        'n_response_neurons': len(rr_results['response_neurons']),
        'n_rr_neurons': len(rr_results['rr_neurons']),
        'rr_ratio': len(rr_results['rr_neurons']) / neuron_pos.shape[1] if neuron_pos.shape[1] > 0 else 0
    }
    
    # 计算RR神经元的空间分布统计
    if len(rr_results['rr_neurons']) > 0:
        rr_positions = neuron_pos[:, rr_results['rr_neurons']]
        distribution_stats.update({
            'rr_positions': rr_positions,
            'rr_center_x': np.mean(rr_positions[0, :]),
            'rr_center_y': np.mean(rr_positions[1, :]),
            'rr_spread_x': np.std(rr_positions[0, :]),
            'rr_spread_y': np.std(rr_positions[1, :])
        })
    
    np.savez_compressed(
        os.path.join(save_dir, 'rr_neurons_distribution.npz'),
        **distribution_stats
    )
    print(f"RR神经元分布统计信息已保存到 {save_dir}/rr_neurons_distribution.npz")

# %% ========== 数据分割函数 ========== 
def segment_neuron_data(neuron_data, trigger_data, stimulus_data, pre_frames=cfg.PRE_FRAMES, post_frames=cfg.POST_FRAMES, baseline_correct=cfg.BASELINE_CORRECT):
    """
    改进的数据分割函数
    
    参数:
    pre_frames: 刺激前的帧数（用于基线）
    post_frames: 刺激后的帧数（用于反应）
    baseline_correct: 是否进行基线校正 (ΔF/F)
    """
    total_frames = pre_frames + post_frames
    segments = np.zeros((len(trigger_data), neuron_data.shape[1], total_frames))
    labels = []

    for i in range(len(trigger_data)): # 遍历每个触发事件
        start = trigger_data[i] - pre_frames
        end = trigger_data[i] + post_frames
        
        # 边界检查
        if start < 0 or end >= neuron_data.shape[0]:
            print(f"警告: 第{i}个刺激的时间窗口超出边界，跳过")
            continue
            
        segment = neuron_data[start:end, :]
        
        # 基线校正 (ΔF/F)
        if baseline_correct:
            baseline = np.mean(segment[:pre_frames, :], axis=0, keepdims=True)
            # 避免除零错误和NaN
            baseline = np.where((baseline == 0) | np.isnan(baseline), 1e-6, baseline)
            segment = (segment - baseline) / baseline
            # 清理可能的NaN和Inf值
            segment = np.nan_to_num(segment, nan=0.0, posinf=0.0, neginf=0.0)
        
        segments[i] = segment.T
        labels.append(stimulus_data[i, 0])
        
    return segments, labels

# %% ========== 标签处理函数 ========== 
def reclassify_labels(stimulus_data):
    """重新分类标签：类别1和2强度为1的作为第3类"""
    new_labels = []
    for i in range(len(stimulus_data)):  
        category = stimulus_data[i, 0]  
        intensity = stimulus_data[i, 1]  
        
        if intensity == cfg.NOISE_INTENSITY:
            new_labels.append(3)  # 强度为1的噪音刺激作为第3类
        elif category == 1 and intensity == 0:
            new_labels.append(1)  # 类别1且强度不为1
        elif category == 2 and intensity == 0:
            new_labels.append(2)  # 类别2且强度不为1
        else:
            new_labels.append(0)  # 其他情况标记为0（会被过滤）
    
    return np.array(new_labels)

# %% ========== RR神经元筛选函数 ========== 
def fast_rr_selection(trials, labels, t_stimulus=cfg.T_STIMULUS, l=cfg.L_STIMULUS, 
                     alpha_fdr=cfg.ALPHA_FDR, alpha_level=cfg.ALPHA_LEVEL, 
                     reliability_threshold=cfg.RELIABILITY_THRESHOLD):
    """
    快速RR神经元筛选
    优化策略:
    1. 向量化计算替代循环
    2. 简化统计检验（t检验替代Mann-Whitney U）
    3. 批量处理所有神经元
    """
    import time
    start_time = time.time()
    
    print("使用快速RR筛选算法...")
    
    # 过滤有效数据
    valid_mask = (labels == 1) | (labels == 2)
    valid_trials = trials[valid_mask]
    valid_labels = labels[valid_mask]
    
    n_trials, n_neurons, n_timepoints = valid_trials.shape
    
    # 定义时间窗口
    baseline_pre = np.arange(0, t_stimulus)
    baseline_post = np.arange(t_stimulus + l, n_timepoints)
    stimulus_window = np.arange(t_stimulus, t_stimulus + l)
    
    print(f"处理 {n_trials} 个试次, {n_neurons} 个神经元")
    
    # 1. 响应性检测 - 向量化计算
    # 计算基线和刺激期的平均值
    baseline_pre_mean = np.mean(valid_trials[:, :, baseline_pre], axis=2)  # (trials, neurons)
    baseline_post_mean = np.mean(valid_trials[:, :, baseline_post], axis=2)  # (trials, neurons)
    # 合并前后基线的平均
    baseline_mean = (baseline_pre_mean + baseline_post_mean) / 2
    
    stimulus_mean = np.mean(valid_trials[:, :, stimulus_window], axis=2)  # (trials, neurons)
    
    # 简化的响应性检测：基于效应大小和标准误差
    baseline_pre_std = np.std(valid_trials[:, :, baseline_pre], axis=2)  # (trials, neurons)
    baseline_post_std = np.std(valid_trials[:, :, baseline_post], axis=2)  # (trials, neurons)
    # 合并前后基线的标准差
    baseline_std = (baseline_pre_std + baseline_post_std) / 2
    
    stimulus_std = np.std(valid_trials[:, :, stimulus_window], axis=2)
    
    # Cohen's d效应大小
    pooled_std = np.sqrt((baseline_std**2 + stimulus_std**2) / 2)
    effect_size = np.abs(stimulus_mean - baseline_mean) / (pooled_std + 1e-8)
    
    # 响应性标准：平均效应大小 > 阈值 且 至少指定比例试次有响应
    response_ratio = np.mean(effect_size > cfg.EFFECT_SIZE_THRESHOLD, axis=0)
    enhanced_neurons = np.where((response_ratio > cfg.RESPONSE_RATIO_THRESHOLD) & 
                              (np.mean(stimulus_mean > baseline_mean, axis=0) > cfg.RESPONSE_RATIO_THRESHOLD))[0].tolist()
    
    # 2. 可靠性检测 - 简化版本
    # 计算每个神经元在每个试次的信噪比
    signal_strength = np.abs(stimulus_mean - baseline_mean)
    noise_level = baseline_std + 1e-8
    snr = signal_strength / noise_level
    
    # 可靠性：指定比例的试次信噪比 > 阈值
    reliability_ratio = np.mean(snr > cfg.SNR_THRESHOLD, axis=0)
    reliable_neurons = np.where(reliability_ratio >= reliability_threshold)[0].tolist()
    
    # 3. 最终RR神经元
    rr_neurons = list(set(enhanced_neurons) & set(reliable_neurons))
    
    elapsed_time = time.time() - start_time
    print(f"快速RR筛选完成，耗时: {elapsed_time:.2f}秒")
    
    return {
        'rr_neurons': rr_neurons,
        'response_neurons': enhanced_neurons,
        'reliable_neurons': reliable_neurons,
        'enhanced_neurons_union': enhanced_neurons,
        'suppressed_neurons_union': [],  # 简化版本不区分抑制
        'processing_time': elapsed_time
    }

# %% ========== 时间点分析函数 ========== 
def classify_by_timepoints(segments, labels, rr_neurons, pre_frames=cfg.PRE_FRAMES, 
                          post_frames=cfg.POST_FRAMES, window_size=1, step_size=1):
    """
    分析trial中每个时间点的分类准确率
    
    参数:
    segments: 神经数据片段 (trials, neurons, timepoints)
    labels: 标签数组
    rr_neurons: RR神经元索引
    pre_frames: 刺激前帧数
    post_frames: 刺激后帧数
    window_size: 未使用（保持接口兼容）
    step_size: 未使用（保持接口兼容）
    
    返回:
    time_accuracies: 每个时间点的准确率
    time_points: 时间点数组
    """
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import RobustScaler
    from sklearn.utils.class_weight import compute_class_weight
    
    # 过滤有效数据和RR神经元
    valid_mask = labels != 0
    valid_segments = segments[valid_mask][:, rr_neurons, :]
    valid_labels = labels[valid_mask]
    
    n_trials, n_neurons, n_timepoints = valid_segments.shape
    print(f"时间点分类分析: {n_trials}个试次, {n_neurons}个神经元, {n_timepoints}个时间点")
    
    time_points = []
    accuracies = []
    
    # 逐个时间点分析
    for t in range(n_timepoints):
        # 提取单个时间点的数据
        timepoint_data = valid_segments[:, :, t]  # (trials, neurons)
        
        # 检查数据方差，跳过无变化的时间点
        if np.var(timepoint_data) < 1e-10:
            time_points.append(t)
            accuracies.append(1.0 / len(np.unique(valid_labels)))  # 随机水平
            continue
        
        # 使用RobustScaler进行标准化（更适合神经数据）
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(timepoint_data)
        
        # 使用与主分类相同的SVM参数
        clf = SVC(kernel=cfg.SVM_KERNEL, C=cfg.SVM_C, gamma=cfg.SVM_GAMMA,
                 random_state=cfg.RANDOM_STATE, class_weight='balanced')
        
        # 交叉验证
        cv = StratifiedKFold(n_splits=cfg.CV_FOLDS, 
                            shuffle=True, random_state=cfg.RANDOM_STATE)
        
        try:
            scores = cross_val_score(clf, X_scaled, valid_labels, cv=cv, scoring='accuracy')
            accuracy = scores.mean()
        except Exception as e:
            print(f"时间点 {t} 分类失败: {e}")
            accuracy = 1.0 / len(np.unique(valid_labels))  # 随机水平
        
        time_points.append(t)
        accuracies.append(accuracy)
        
        if t % 5 == 0:  # 每5个时间点打印一次进度
            print(f"时间点 {t}: 准确率 {accuracy:.3f}")
    
    return np.array(accuracies), np.array(time_points)

def save_accuracy_over_time(accuracies, time_points, pre_frames=cfg.PRE_FRAMES, 
                           stimulus_duration=cfg.STIMULUS_DURATION, save_dir='results'):
    """
    保存准确率随时间变化的数据和统计信息
    
    参数:
    accuracies: 准确率数组
    time_points: 时间点数组
    pre_frames: 刺激前帧数
    stimulus_duration: 刺激持续时间
    """
    os.makedirs(save_dir, exist_ok=True)
    
    stimulus_start = pre_frames
    stimulus_end = pre_frames + stimulus_duration
    
    # 计算统计信息
    max_acc_idx = np.argmax(accuracies)
    max_time = time_points[max_acc_idx]
    max_acc = accuracies[max_acc_idx]
    
    baseline_acc = np.mean(accuracies[:pre_frames]) if pre_frames > 0 else 0
    stimulus_acc = np.mean(accuracies[stimulus_start:stimulus_end])
    response_acc = np.mean(accuracies[stimulus_end:]) if stimulus_end < len(accuracies) else 0
    
    time_stats = {
        'time_points': time_points,
        'accuracies': accuracies,
        'stimulus_start': stimulus_start,
        'stimulus_end': stimulus_end,
        'baseline_accuracy': baseline_acc,
        'stimulus_accuracy': stimulus_acc,
        'response_accuracy': response_acc,
        'max_accuracy': max_acc,
        'max_time': max_time,
        'overall_mean': np.mean(accuracies),
        'overall_std': np.std(accuracies),
        'chance_level': 1.0 / 3  # 3类分类
    }
    
    np.savez_compressed(
        os.path.join(save_dir, 'accuracy_over_time.npz'),
        **time_stats
    )
    
    # 打印关键统计信息
    print(f"\n=== 时间点分类分析结果 ===")
    print(f"基线期平均准确率: {baseline_acc:.3f}")
    print(f"刺激期平均准确率: {stimulus_acc:.3f}")
    print(f"响应期平均准确率: {response_acc:.3f}")
    print(f"最高准确率: {max_acc:.3f} (时间点 {max_time})")
    print(f"整体平均准确率: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
    print(f"时间点分类结果已保存到 {save_dir}/accuracy_over_time.npz")

def visualize_accuracy_over_time(accuracies, time_points=None, 
                               pre_frames=cfg.PRE_FRAMES, stimulus_duration=cfg.STIMULUS_DURATION,
                               chance_level=1/3, save_path=None):
    """
    Visualize classification accuracy over time
    
    Parameters:
    accuracies: Array of accuracy scores
    time_points: Time point array  
    pre_frames: Number of baseline frames
    stimulus_duration: Duration of stimulus in frames
    chance_level: Random chance level
    save_path: Path to save the figure
    """
    setup_plot_style()
    
    if time_points is None:
        time_points = np.arange(len(accuracies))
    
    stimulus_start = pre_frames
    stimulus_end = pre_frames + stimulus_duration
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Main accuracy plot
    ax1.plot(time_points, accuracies, linewidth=2.5, color=cfg.PLOT_COLORS['primary'], 
             alpha=0.8, label='Classification Accuracy')
    
    # Add chance level
    ax1.axhline(chance_level, color=cfg.PLOT_COLORS['neutral'], linestyle='--', 
               linewidth=1.5, alpha=0.7, label=f'Chance Level ({chance_level:.3f})')
    
    # Add baseline and stimulus periods
    ax1.axvspan(0, stimulus_start, alpha=0.2, color=cfg.BASELINE_COLOR, 
               label='Baseline Period')
    ax1.axvspan(stimulus_start, stimulus_end, alpha=0.2, color=cfg.STIMULUS_COLOR, 
               label='Stimulus Period')
    ax1.axvspan(stimulus_end, len(time_points), alpha=0.2, color=cfg.RESPONSE_COLOR, 
               label='Response Period')
    
    # Mark peak
    max_idx = np.argmax(accuracies)
    max_time = time_points[max_idx]
    max_acc = accuracies[max_idx]
    ax1.plot(max_time, max_acc, 'o', markersize=8, color=cfg.PLOT_COLORS['accent'], 
             markeredgecolor='black', markeredgewidth=1.5, 
             label=f'Peak: {max_acc:.3f} at t={max_time}')
    
    ax1.set_xlabel(f'Time ({cfg.TIME_UNIT})')
    ax1.set_ylabel('Classification Accuracy')
    ax1.set_title('Classification Accuracy Over Time')
    ax1.set_ylim([0, max(1.0, max(accuracies) * 1.1)])
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Period-wise comparison
    periods = ['Baseline', 'Stimulus', 'Response']
    
    # Calculate period averages
    baseline_acc = np.mean(accuracies[:pre_frames]) if pre_frames > 0 else 0
    stimulus_acc = np.mean(accuracies[stimulus_start:stimulus_end])
    response_acc = np.mean(accuracies[stimulus_end:]) if stimulus_end < len(accuracies) else 0
    
    period_values = [baseline_acc, stimulus_acc, response_acc]
    period_colors = [cfg.BASELINE_COLOR, cfg.STIMULUS_COLOR, cfg.RESPONSE_COLOR]
    
    bars = ax2.bar(periods, period_values, color=period_colors, alpha=0.7, 
                   edgecolor='black', linewidth=1.2)
    
    # Add chance level line
    ax2.axhline(chance_level, color=cfg.PLOT_COLORS['neutral'], linestyle='--', 
               linewidth=1.5, alpha=0.7, label=f'Chance Level')
    
    # Add value labels
    for bar, value in zip(bars, period_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(period_values) * 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_ylabel('Average Classification Accuracy')
    ax2.set_title('Period-wise Classification Accuracy Comparison')
    ax2.set_ylim([0, max(1.0, max(period_values) * 1.2)])
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=cfg.VISUALIZATION_DPI, bbox_inches='tight')
        print(f"Accuracy over time plot saved to: {save_path}")
    plt.show()

def visualize_combined_analysis(accuracies, fisher_scores, time_points=None,
                              pre_frames=cfg.PRE_FRAMES, stimulus_duration=cfg.STIMULUS_DURATION,
                              save_path=None):
    """
    Visualize both classification accuracy and Fisher information over time in one plot
    
    Parameters:
    accuracies: Array of accuracy scores
    fisher_scores: Array of Fisher information scores
    time_points: Time point array
    pre_frames: Number of baseline frames  
    stimulus_duration: Duration of stimulus in frames
    save_path: Path to save the figure
    """
    setup_plot_style()
    
    if time_points is None:
        time_points = np.arange(len(accuracies))
    
    stimulus_start = pre_frames
    stimulus_end = pre_frames + stimulus_duration
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Classification accuracy plot
    ax1.plot(time_points, accuracies, linewidth=2.5, color=cfg.PLOT_COLORS['primary'], 
             alpha=0.8, label='Classification Accuracy')
    ax1.axhline(1/3, color=cfg.PLOT_COLORS['neutral'], linestyle='--', 
               linewidth=1.5, alpha=0.7, label='Chance Level (0.333)')
    
    # Add periods
    for ax in [ax1, ax2]:
        ax.axvspan(0, stimulus_start, alpha=0.15, color=cfg.BASELINE_COLOR)
        ax.axvspan(stimulus_start, stimulus_end, alpha=0.15, color=cfg.STIMULUS_COLOR)
        ax.axvspan(stimulus_end, len(time_points), alpha=0.15, color=cfg.RESPONSE_COLOR)
    
    ax1.set_ylabel('Classification Accuracy')
    ax1.set_title('Classification Accuracy Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, max(1.0, max(accuracies) * 1.1)])
    
    # Fisher information plot  
    ax2.plot(time_points, fisher_scores, linewidth=2.5, color=cfg.PLOT_COLORS['secondary'], 
             alpha=0.8, label='Fisher Information')
    
    ax2.set_ylabel('Fisher Information Score')
    ax2.set_title('Fisher Information Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Combined correlation plot
    # Normalize both measures to [0,1] for comparison
    norm_acc = (accuracies - np.min(accuracies)) / (np.max(accuracies) - np.min(accuracies) + 1e-8)
    norm_fisher = (fisher_scores - np.min(fisher_scores)) / (np.max(fisher_scores) - np.min(fisher_scores) + 1e-8)
    
    ax3.plot(time_points, norm_acc, linewidth=2, color=cfg.PLOT_COLORS['primary'], 
             alpha=0.8, label='Normalized Accuracy')
    ax3.plot(time_points, norm_fisher, linewidth=2, color=cfg.PLOT_COLORS['secondary'], 
             alpha=0.8, label='Normalized Fisher Info')
    
    # Calculate correlation
    correlation = np.corrcoef(accuracies, fisher_scores)[0, 1]
    
    ax3.axvspan(0, stimulus_start, alpha=0.15, color=cfg.BASELINE_COLOR, label='Baseline')
    ax3.axvspan(stimulus_start, stimulus_end, alpha=0.15, color=cfg.STIMULUS_COLOR, label='Stimulus')
    ax3.axvspan(stimulus_end, len(time_points), alpha=0.15, color=cfg.RESPONSE_COLOR, label='Response')
    
    ax3.set_xlabel(f'Time ({cfg.TIME_UNIT})')
    ax3.set_ylabel('Normalized Score')
    ax3.set_title(f'Combined Analysis (Correlation: {correlation:.3f})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=cfg.VISUALIZATION_DPI, bbox_inches='tight')
        print(f"Combined analysis plot saved to: {save_path}")
    plt.show()

def visualize_neuron_count_effect(neuron_counts, accuracies, accuracy_stds, 
                                fisher_scores, fisher_stds, save_path=None):
    """
    Visualize the effect of neuron count on classification performance and Fisher information
    
    Parameters:
    neuron_counts: Array of neuron counts tested
    accuracies: Array of accuracy scores
    accuracy_stds: Array of accuracy standard deviations
    fisher_scores: Array of Fisher information scores  
    fisher_stds: Array of Fisher information standard deviations
    save_path: Path to save the figure
    """
    setup_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Classification accuracy vs neuron count
    ax1.errorbar(neuron_counts, accuracies, yerr=accuracy_stds, 
                fmt='o-', linewidth=2.5, markersize=6, capsize=5,
                color=cfg.PLOT_COLORS['primary'], alpha=0.8, 
                markerfacecolor='white', markeredgewidth=2)
    
    # Add saturation analysis
    if len(accuracies) > 2:
        # Find where improvement becomes minimal
        improvements = np.diff(accuracies)
        saturation_threshold = 0.01  # 1% improvement threshold
        
        saturation_indices = np.where(improvements < saturation_threshold)[0]
        if len(saturation_indices) > 0:
            saturation_point = neuron_counts[saturation_indices[0] + 1]
            ax1.axvline(saturation_point, color=cfg.PLOT_COLORS['accent'], 
                       linestyle='--', linewidth=2, alpha=0.7,
                       label=f'Saturation Point (~{saturation_point} neurons)')
    
    ax1.set_xlabel('Number of Neurons')
    ax1.set_ylabel('Classification Accuracy')
    ax1.set_title('Classification Performance vs Neuron Count')
    ax1.grid(True, alpha=0.3)
    if 'saturation_point' in locals():
        ax1.legend()
    
    # Fisher information vs neuron count
    ax2.errorbar(neuron_counts, fisher_scores, yerr=fisher_stds,
                fmt='s-', linewidth=2.5, markersize=6, capsize=5,
                color=cfg.PLOT_COLORS['secondary'], alpha=0.8,
                markerfacecolor='white', markeredgewidth=2)
    
    # Add trend line
    if len(fisher_scores) > 2:
        z = np.polyfit(neuron_counts, fisher_scores, 1)
        p = np.poly1d(z)
        ax2.plot(neuron_counts, p(neuron_counts), color=cfg.PLOT_COLORS['accent'], 
                linestyle=':', linewidth=2, alpha=0.7, 
                label=f'Trend (slope: {z[0]:.3f})')
        ax2.legend()
    
    ax2.set_xlabel('Number of Neurons')
    ax2.set_ylabel('Fisher Information Score')
    ax2.set_title('Fisher Information vs Neuron Count')
    ax2.grid(True, alpha=0.3)
    
    # Add performance annotations
    max_acc_idx = np.argmax(accuracies)
    ax1.annotate(f'Peak: {accuracies[max_acc_idx]:.3f}',
                xy=(neuron_counts[max_acc_idx], accuracies[max_acc_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    max_fisher_idx = np.argmax(fisher_scores)
    ax2.annotate(f'Peak: {fisher_scores[max_fisher_idx]:.3f}',
                xy=(neuron_counts[max_fisher_idx], fisher_scores[max_fisher_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=cfg.VISUALIZATION_DPI, bbox_inches='tight')
        print(f"Neuron count effect plot saved to: {save_path}")
    plt.show()

# %% ========== Fisher信息分析函数 ========== 
def calculate_fisher_information(segments, labels, rr_neurons):
    """
    计算每个时间点的Fisher信息，用于衡量类别可分离性
    
    参数:
    segments: 神经数据片段 (trials, neurons, timepoints)
    labels: 标签数组
    rr_neurons: RR神经元索引
    
    返回:
    fisher_scores: 每个时间点的Fisher信息分数
    """
    from scipy.stats import f_oneway
    
    # 过滤有效数据和RR神经元
    valid_mask = labels != 0
    valid_segments = segments[valid_mask][:, rr_neurons, :]
    valid_labels = labels[valid_mask]
    
    n_trials, n_neurons, n_timepoints = valid_segments.shape
    print(f"Fisher信息计算: {n_trials}个试次, {n_neurons}个神经元, {n_timepoints}个时间点")
    
    fisher_scores = []
    unique_labels = np.unique(valid_labels)
    
    for t in range(n_timepoints):
        # 提取单个时间点的数据
        timepoint_data = valid_segments[:, :, t]  # (trials, neurons)
        
        # 使用多变量Fisher信息计算
        fisher_score = calculate_multivariate_fisher_single_timepoint(timepoint_data, valid_labels)
        fisher_scores.append(fisher_score)
        
        if t % 10 == 0:  # 每10个时间点打印一次进度
            print(f"时间点 {t}: Fisher信息 {fisher_score:.3f}")
    
    return np.array(fisher_scores)

def calculate_multivariate_fisher_single_timepoint(data, labels):
    """
    计算单个时间点的多变量Fisher信息
    
    参数:
    data: 神经数据 (trials, neurons)
    labels: 标签数组
    
    返回:
    fisher_score: 多变量Fisher信息分数
    """
    from scipy.linalg import pinv
    
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0
    
    n_trials, n_neurons = data.shape
    n_classes = len(unique_labels)
    
    # 检查是否有足够的数据
    min_samples_per_class = min([np.sum(labels == label) for label in unique_labels])
    if min_samples_per_class < 2:
        return 0.0
    
    # 计算总体均值
    grand_mean = np.mean(data, axis=0)  # (n_neurons,)
    
    # 计算类别均值和样本数
    class_means = []
    class_sizes = []
    
    for label in unique_labels:
        label_mask = labels == label
        label_data = data[label_mask]
        if len(label_data) > 0:
            class_means.append(np.mean(label_data, axis=0))
            class_sizes.append(len(label_data))
        else:
            class_means.append(grand_mean)
            class_sizes.append(0)
    
    class_means = np.array(class_means)  # (n_classes, n_neurons)
    class_sizes = np.array(class_sizes)
    
    # 计算类间散布矩阵 (Between-class scatter matrix)
    S_b = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    for i, (class_mean, n_i) in enumerate(zip(class_means, class_sizes)):
        if n_i > 0:
            diff = (class_mean - grand_mean).reshape(-1, 1).astype(np.float64)
            S_b += n_i * np.dot(diff, diff.T).astype(np.float64)
    
    # 计算类内散布矩阵 (Within-class scatter matrix)
    S_w = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    for label in unique_labels:
        label_mask = labels == label
        label_data = data[label_mask]
        if len(label_data) > 1:  # 需要至少2个样本才能计算协方差
            class_mean = np.mean(label_data, axis=0).astype(np.float64)
            centered_data = (label_data - class_mean).astype(np.float64)
            S_w += np.dot(centered_data.T, centered_data).astype(np.float64)
    
    # 添加正则化项避免奇异矩阵
    regularization = 1e-6 * np.eye(n_neurons, dtype=np.float64)
    S_w += regularization
    
    try:
        # 计算多变量Fisher判别比: trace(S_w^(-1) * S_b)
        S_w_inv = pinv(S_w)
        fisher_matrix = np.dot(S_w_inv, S_b)
        
        # Fisher信息是矩阵的迹
        fisher_score = np.trace(fisher_matrix)
        
        # 确保返回非负值
        fisher_score = max(0.0, fisher_score)
        
    except Exception as e:
        fisher_score = 0.0
    
    return fisher_score

def save_fisher_information(fisher_scores, time_points, pre_frames=cfg.PRE_FRAMES, 
                           stimulus_duration=cfg.STIMULUS_DURATION, save_dir='results'):
    """
    保存Fisher信息随时间变化的数据和统计信息
    
    参数:
    fisher_scores: Fisher信息分数数组
    time_points: 时间点数组
    pre_frames: 刺激前帧数
    stimulus_duration: 刺激持续时间
    """
    os.makedirs(save_dir, exist_ok=True)
    
    stimulus_start = pre_frames
    stimulus_end = pre_frames + stimulus_duration
    
    # 计算统计信息
    max_fisher_idx = np.argmax(fisher_scores)
    max_time = time_points[max_fisher_idx]
    max_fisher = fisher_scores[max_fisher_idx]
    
    baseline_fisher = np.mean(fisher_scores[:pre_frames]) if pre_frames > 0 else 0
    stimulus_fisher = np.mean(fisher_scores[stimulus_start:stimulus_end])
    response_fisher = np.mean(fisher_scores[stimulus_end:]) if stimulus_end < len(fisher_scores) else 0
    
    fisher_stats = {
        'time_points': time_points,
        'fisher_scores': fisher_scores,
        'stimulus_start': stimulus_start,
        'stimulus_end': stimulus_end,
        'baseline_fisher': baseline_fisher,
        'stimulus_fisher': stimulus_fisher,
        'response_fisher': response_fisher,
        'max_fisher': max_fisher,
        'max_time': max_time,
        'overall_mean': np.mean(fisher_scores),
        'overall_std': np.std(fisher_scores)
    }
    
    np.savez_compressed(
        os.path.join(save_dir, 'fisher_over_time.npz'),
        **fisher_stats
    )
    
    # 打印关键统计信息
    print(f"\n=== Fisher信息分析结果 ===")
    print(f"基线期平均Fisher信息: {baseline_fisher:.3f}")
    print(f"刺激期平均Fisher信息: {stimulus_fisher:.3f}")
    print(f"响应期平均Fisher信息: {response_fisher:.3f}")
    print(f"最高Fisher信息: {max_fisher:.3f} (时间点 {max_time})")
    print(f"整体平均Fisher信息: {np.mean(fisher_scores):.3f} ± {np.std(fisher_scores):.3f}")
    print(f"Fisher信息结果已保存到 {save_dir}/fisher_over_time.npz")

def visualize_fisher_information(fisher_scores, time_points=None, 
                               pre_frames=cfg.PRE_FRAMES, stimulus_duration=cfg.STIMULUS_DURATION,
                               save_path=None):
    """
    Visualize Fisher Information over time
    
    Parameters:
    fisher_scores: Array of Fisher information scores
    time_points: Time point array
    pre_frames: Number of baseline frames
    stimulus_duration: Duration of stimulus in frames
    save_path: Path to save the figure
    """
    setup_plot_style()
    
    if time_points is None:
        time_points = np.arange(len(fisher_scores))
    
    stimulus_start = pre_frames
    stimulus_end = pre_frames + stimulus_duration
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Main Fisher information plot
    ax1.plot(time_points, fisher_scores, linewidth=2.5, color=cfg.PLOT_COLORS['primary'], 
             alpha=0.8, label='Fisher Information')
    
    # Add baseline and stimulus periods
    ax1.axvspan(0, stimulus_start, alpha=0.2, color=cfg.BASELINE_COLOR, 
               label='Baseline Period')
    ax1.axvspan(stimulus_start, stimulus_end, alpha=0.2, color=cfg.STIMULUS_COLOR, 
               label='Stimulus Period')
    ax1.axvspan(stimulus_end, len(time_points), alpha=0.2, color=cfg.RESPONSE_COLOR, 
               label='Response Period')
    
    # Mark peak
    max_idx = np.argmax(fisher_scores)
    max_time = time_points[max_idx]
    max_fisher = fisher_scores[max_idx]
    ax1.plot(max_time, max_fisher, 'o', markersize=8, color=cfg.PLOT_COLORS['accent'], 
             markeredgecolor='black', markeredgewidth=1.5, 
             label=f'Peak: {max_fisher:.3f} at t={max_time}')
    
    ax1.set_xlabel(f'Time ({cfg.TIME_UNIT})')
    ax1.set_ylabel('Fisher Information Score')
    ax1.set_title('Fisher Information Over Time')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Period-wise comparison
    periods = ['Baseline', 'Stimulus', 'Response']
    
    # Calculate period averages
    baseline_fisher = np.mean(fisher_scores[:pre_frames]) if pre_frames > 0 else 0
    stimulus_fisher = np.mean(fisher_scores[stimulus_start:stimulus_end])
    response_fisher = np.mean(fisher_scores[stimulus_end:]) if stimulus_end < len(fisher_scores) else 0
    
    period_values = [baseline_fisher, stimulus_fisher, response_fisher]
    period_colors = [cfg.BASELINE_COLOR, cfg.STIMULUS_COLOR, cfg.RESPONSE_COLOR]
    
    bars = ax2.bar(periods, period_values, color=period_colors, alpha=0.7, 
                   edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bar, value in zip(bars, period_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(period_values) * 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_ylabel('Average Fisher Information')
    ax2.set_title('Period-wise Fisher Information Comparison')
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=cfg.VISUALIZATION_DPI, bbox_inches='tight')
        print(f"Fisher information plot saved to: {save_path}")
    plt.show()

def visualize_fisher_heatmap(segments, labels, rr_neurons, time_window=None, save_path=None):
    """
    Visualize Fisher Information as a heatmap across neurons and time
    
    Parameters:
    segments: Neural data segments (trials, neurons, timepoints)
    labels: Label array
    rr_neurons: RR neuron indices
    time_window: Specific time window to analyze (start, end)
    save_path: Path to save the figure
    """
    setup_plot_style()
    
    # Filter valid data and RR neurons
    valid_mask = labels != 0
    valid_segments = segments[valid_mask][:, rr_neurons, :]
    valid_labels = labels[valid_mask]
    
    n_trials, n_neurons, n_timepoints = valid_segments.shape
    
    if time_window:
        start_t, end_t = time_window
        valid_segments = valid_segments[:, :, start_t:end_t]
        n_timepoints = end_t - start_t
        time_points = np.arange(start_t, end_t)
    else:
        time_points = np.arange(n_timepoints)
    
    # Calculate Fisher information for each neuron at each timepoint
    fisher_matrix = np.zeros((n_neurons, n_timepoints))
    
    print(f"Calculating Fisher information heatmap for {n_neurons} neurons...")
    
    for neuron_idx in range(min(n_neurons, 50)):  # Limit to 50 neurons for visualization
        for t_idx in range(n_timepoints):
            timepoint_data = valid_segments[:, neuron_idx, t_idx].reshape(-1, 1)
            fisher_score = calculate_multivariate_fisher_single_timepoint(timepoint_data, valid_labels)
            fisher_matrix[neuron_idx, t_idx] = fisher_score
        
        if neuron_idx % 10 == 0:
            print(f"Processed {neuron_idx+1}/{min(n_neurons, 50)} neurons")
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    
    im = ax.imshow(fisher_matrix[:50], aspect='auto', cmap='viridis', 
                   interpolation='nearest', alpha=0.9)
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Fisher Information Score', rotation=270, labelpad=20)
    
    # Add stimulus period indication
    if not time_window:
        stimulus_start = cfg.PRE_FRAMES
        stimulus_end = cfg.PRE_FRAMES + cfg.STIMULUS_DURATION
        ax.axvline(stimulus_start, color='white', linestyle='--', linewidth=2, alpha=0.8)
        ax.axvline(stimulus_end, color='white', linestyle='--', linewidth=2, alpha=0.8)
        ax.text(stimulus_start + cfg.STIMULUS_DURATION/2, n_neurons*0.95, 'Stimulus', 
                ha='center', va='center', color='white', fontweight='bold')
    
    ax.set_xlabel(f'Time ({cfg.TIME_UNIT})')
    ax.set_ylabel('RR Neurons')
    ax.set_title('Fisher Information Heatmap Across Neurons and Time')
    
    # Set ticks
    n_ticks = 10
    time_tick_indices = np.linspace(0, n_timepoints-1, n_ticks, dtype=int)
    ax.set_xticks(time_tick_indices)
    ax.set_xticklabels([f'{time_points[i]}' for i in time_tick_indices])
    
    neuron_tick_indices = np.linspace(0, min(n_neurons, 50)-1, min(10, n_neurons), dtype=int)
    ax.set_yticks(neuron_tick_indices)
    ax.set_yticklabels([f'N{rr_neurons[i]}' for i in neuron_tick_indices])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=cfg.VISUALIZATION_DPI, bbox_inches='tight')
        print(f"Fisher information heatmap saved to: {save_path}")
    plt.show()

def visualize_fisher_comparison(fisher_data_dict, save_path=None):
    """
    Compare Fisher Information across different conditions or methods
    
    Parameters:
    fisher_data_dict: Dictionary with condition names as keys and fisher scores as values
    save_path: Path to save the figure
    """
    setup_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Time series comparison
    colors = plt.cm.Set2(np.linspace(0, 1, len(fisher_data_dict)))
    
    for (condition, fisher_scores), color in zip(fisher_data_dict.items(), colors):
        time_points = np.arange(len(fisher_scores))
        ax1.plot(time_points, fisher_scores, linewidth=2.5, 
                label=condition, color=color, alpha=0.8)
    
    # Add stimulus period
    stimulus_start = cfg.PRE_FRAMES
    stimulus_end = cfg.PRE_FRAMES + cfg.STIMULUS_DURATION
    ax1.axvspan(stimulus_start, stimulus_end, alpha=0.2, color=cfg.STIMULUS_COLOR, 
               label='Stimulus Period')
    
    ax1.set_xlabel(f'Time ({cfg.TIME_UNIT})')
    ax1.set_ylabel('Fisher Information Score')
    ax1.set_title('Fisher Information Comparison Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Summary statistics comparison
    conditions = list(fisher_data_dict.keys())
    means = [np.mean(scores) for scores in fisher_data_dict.values()]
    peaks = [np.max(scores) for scores in fisher_data_dict.values()]
    
    x = np.arange(len(conditions))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, means, width, label='Mean Fisher Info', 
                   color=cfg.PLOT_COLORS['primary'], alpha=0.7, edgecolor='black')
    bars2 = ax2.bar(x + width/2, peaks, width, label='Peak Fisher Info',
                   color=cfg.PLOT_COLORS['accent'], alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(peaks) * 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_ylabel('Fisher Information Score')
    ax2.set_title('Fisher Information Summary Statistics')
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions)
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=cfg.VISUALIZATION_DPI, bbox_inches='tight')
        print(f"Fisher information comparison saved to: {save_path}")
    plt.show()


# %% ========== 神经元数量对性能影响分析 ========== 

def analyze_neuron_count_effect(segments, labels, rr_neurons, time_start=20, time_end=30, 
                               neuron_counts=None, n_iterations=10):
    """
    分析随着神经元数量增加对分类准确率和Fisher信息的影响
    
    参数:
    segments: 神经数据片段 (trials, neurons, timepoints)
    labels: 标签数组
    rr_neurons: RR神经元索引列表
    time_start: 分析时间窗口开始点
    time_end: 分析时间窗口结束点
    neuron_counts: 要测试的神经元数量列表
    n_iterations: 每个神经元数量的重复次数（随机采样）
    
    返回:
    results: 包含分类准确率和Fisher信息结果的字典
    """
    from scipy.stats import f_oneway
    
    print(f"分析神经元数量对性能的影响（时间窗口: {time_start}-{time_end}）")
    
    # 过滤有效数据和RR神经元
    valid_mask = labels != 0
    valid_segments = segments[valid_mask][:, rr_neurons, :]
    valid_labels = labels[valid_mask]
    
    n_trials, n_rr_neurons, n_timepoints = valid_segments.shape
    
    # 设置默认的神经元数量测试点
    if neuron_counts is None:
        max_neurons = min(n_rr_neurons, 200)  # 最多测试200个神经元
        neuron_counts = [5, 10, 20, 30, 50, 75, 100, 150, 200]
        neuron_counts = [n for n in neuron_counts if n <= max_neurons]
    
    print(f"可用RR神经元数: {n_rr_neurons}")
    print(f"测试神经元数量: {neuron_counts}")
    
    results = {
        'neuron_counts': neuron_counts,
        'accuracies': [],
        'accuracy_stds': [],
        'fisher_scores': [],
        'fisher_stds': []
    }
    
    for n_neurons in neuron_counts:
        print(f"\n测试 {n_neurons} 个神经元...")
        
        iteration_accuracies = []
        iteration_fishers = []
        
        for iteration in range(n_iterations):
            # 随机选择神经元子集
            if n_neurons >= n_rr_neurons:
                selected_neurons = list(range(n_rr_neurons))
            else:
                selected_neurons = np.random.choice(n_rr_neurons, n_neurons, replace=False)
            
            # 提取选定神经元和时间窗口的数据
            subset_data = valid_segments[:, selected_neurons, time_start:time_end]
            
            # 1. 计算分类准确率
            accuracy = calculate_classification_accuracy_window(subset_data, valid_labels)
            iteration_accuracies.append(accuracy)
            
            # 2. 计算Fisher信息
            fisher_score = calculate_fisher_information_window(subset_data, valid_labels)
            iteration_fishers.append(fisher_score)
        
        # 统计结果
        mean_accuracy = np.mean(iteration_accuracies)
        std_accuracy = np.std(iteration_accuracies)
        mean_fisher = np.mean(iteration_fishers)
        std_fisher = np.std(iteration_fishers)
        
        results['accuracies'].append(mean_accuracy)
        results['accuracy_stds'].append(std_accuracy)
        results['fisher_scores'].append(mean_fisher)
        results['fisher_stds'].append(std_fisher)
        
        print(f"  分类准确率: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
        print(f"  Fisher信息: {mean_fisher:.3f} ± {std_fisher:.3f}")
    
    return results


def calculate_classification_accuracy_window(data, labels):
    """
    计算指定时间窗口数据的分类准确率
    
    参数:
    data: 神经数据 (trials, neurons, timepoints)
    labels: 标签数组
    
    返回:
    accuracy: 分类准确率
    """
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    
    # 将数据展平为特征向量
    n_trials, n_neurons, n_timepoints = data.shape
    features = data.reshape(n_trials, n_neurons * n_timepoints)
    
    # 检查是否有足够的数据进行分类
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0
    
    # 检查每个类别是否有足够的样本
    min_samples = min([np.sum(labels == label) for label in unique_labels])
    if min_samples < 2:
        return 0.0
    
    try:
        # 数据标准化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 分割数据
        test_size = min(0.3, 0.5)  # 确保有足够的训练数据
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels, test_size=test_size, 
            random_state=42, stratify=labels
        )
        
        # 训练分类器
        classifier = SVC(kernel='linear', random_state=42, C=1.0)
        classifier.fit(X_train, y_train)
        
        # 预测和评估
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
    
    except Exception as e:
        print(f"分类计算出错: {e}")
        return 0.0


def calculate_fisher_information_window(data, labels):
    """
    计算指定时间窗口数据的多变量Fisher信息（改进版本，包含PCA降维）
    
    参数:
    data: 神经数据 (trials, neurons, timepoints)
    labels: 标签数组
    
    返回:
    fisher_score: 多变量Fisher信息分数
    """
    from scipy.linalg import pinv, eigvals
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # 对时间维度取平均
    mean_data = np.mean(data, axis=2)  # (trials, neurons)
    
    # 只使用类别1和2进行Fisher信息计算，排除类别3（噪音条件）
    target_labels = [1, 2]
    target_mask = np.isin(labels, target_labels)
    
    if np.sum(target_mask) < 10:  # 至少需要10个样本
        return 0.0
        
    # 过滤数据和标签
    filtered_data = mean_data[target_mask]
    filtered_labels = labels[target_mask]
    
    unique_labels = np.unique(filtered_labels)
    if len(unique_labels) < 2:
        return 0.0
    
    n_trials, n_neurons = filtered_data.shape
    n_classes = len(unique_labels)
    
    # 检查样本数是否足够
    min_samples_per_class = min([np.sum(filtered_labels == label) for label in unique_labels])
    if min_samples_per_class < 2:
        return 0.0
    
    # 数据标准化避免数值问题
    scaler = StandardScaler()
    mean_data_scaled = scaler.fit_transform(filtered_data)
    
    # 关键改进：当神经元数量接近或超过试次数时，使用PCA降维
    effective_dim = min(n_neurons, n_trials - n_classes - 1)  # 有效维度上限
    
    # 设置PCA目标维度：确保远小于试次数
    if n_neurons > n_trials * 0.5:  # 当神经元数 > 试次数的50%时进行降维
        # 目标维度：试次数的1/3，但至少保留2维，最多不超过15维
        target_dim = max(2, min(15, n_trials // 3))
        
        print(f"使用PCA降维: {n_neurons}维 -> {target_dim}维 (试次数: {n_trials})")
        
        # 执行PCA降维
        pca = PCA(n_components=target_dim, random_state=42)
        mean_data_scaled = pca.fit_transform(mean_data_scaled)
        
        # 更新维度信息
        n_neurons = target_dim
        print(f"PCA解释方差比: {np.sum(pca.explained_variance_ratio_):.3f}")
    
    # 现在在降维后的数据上计算多变量Fisher信息
    
    # 计算总体均值
    grand_mean = np.mean(mean_data_scaled, axis=0)  # (n_neurons,)
    
    # 计算类别均值和样本数
    class_means = []
    class_sizes = []
    
    for label in unique_labels:
        label_mask = filtered_labels == label
        label_data = mean_data_scaled[label_mask]
        if len(label_data) > 0:
            class_means.append(np.mean(label_data, axis=0))
            class_sizes.append(len(label_data))
        else:
            class_means.append(grand_mean)
            class_sizes.append(0)
    
    class_means = np.array(class_means)  # (n_classes, n_neurons)
    class_sizes = np.array(class_sizes)
    
    # 计算类间散布矩阵 (Between-class scatter matrix)
    S_b = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    for i, (class_mean, n_i) in enumerate(zip(class_means, class_sizes)):
        if n_i > 0:
            diff = (class_mean - grand_mean).reshape(-1, 1).astype(np.float64)
            S_b += n_i * np.dot(diff, diff.T).astype(np.float64)
    
    # 计算类内散布矩阵 (Within-class scatter matrix)
    S_w = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    for label in unique_labels:
        label_mask = filtered_labels == label
        label_data = mean_data_scaled[label_mask]
        if len(label_data) > 1:
            class_mean = np.mean(label_data, axis=0).astype(np.float64)
            centered_data = (label_data - class_mean).astype(np.float64)
            S_w += np.dot(centered_data.T, centered_data).astype(np.float64)
    
    # 自适应正则化：基于数据规模和条件数
    # 计算S_w的条件数来决定正则化强度
    try:
        eigenvals = eigvals(S_w).real.astype(np.float64)  # 确保实数且为float64
        eigenvals = eigenvals[eigenvals > 0]  # 只考虑正特征值
        if len(eigenvals) > 1:
            condition_number = float(np.max(eigenvals) / np.min(eigenvals))
            # 根据条件数自适应调整正则化
            reg_strength = max(1e-6, float(np.max(eigenvals)) * 1e-10 * condition_number)
        else:
            reg_strength = 1e-3
    except:
        reg_strength = 1e-3
    
    regularization = (reg_strength * np.eye(n_neurons)).astype(np.float64)
    S_w += regularization
    
    try:
        # 使用更稳定的方法计算多变量Fisher信息
        # 方法1: 直接计算trace(S_w^(-1) * S_b)
        S_w_inv = pinv(S_w).astype(np.float64)
        fisher_matrix = np.dot(S_w_inv, S_b).astype(np.float64)
        fisher_score = float(np.trace(fisher_matrix).real)  # 确保实数
        
        # 数值稳定性检查
        if np.isnan(fisher_score) or np.isinf(fisher_score) or fisher_score < 0:
            # 方法2: 使用广义特征值问题求解
            from scipy.linalg import eigh
            try:
                eigenvals, _ = eigh(S_b, S_w)
                eigenvals_real = eigenvals.real.astype(np.float64)
                fisher_score = float(np.sum(eigenvals_real[eigenvals_real > 0]))
            except:
                # 方法3: 简化的多变量Fisher比率
                trace_s_b = float(np.trace(S_b).real)
                trace_s_w = float(np.trace(S_w).real)
                fisher_score = trace_s_b / (trace_s_w + 1e-10)
        
        # 确保返回非负有限值
        fisher_score = max(0.0, float(fisher_score))
        if not np.isfinite(fisher_score):
            fisher_score = 0.0
        
    except Exception as e:
        # 最后的备选方案：使用简化版本
        try:
            trace_s_b = float(np.trace(S_b).real)
            trace_s_w = float(np.trace(S_w).real)
            fisher_score = trace_s_b / (trace_s_w + 1e-10)
            fisher_score = max(0.0, float(fisher_score))
        except:
            fisher_score = 0.0
    
    return fisher_score


def save_neuron_count_analysis(results, save_dir='results'):
    """
    保存神经元数量分析结果
    
    参数:
    results: analyze_neuron_count_effect的返回结果
    """
    os.makedirs(save_dir, exist_ok=True)
    
    neuron_counts = results['neuron_counts']
    accuracies = results['accuracies']
    accuracy_stds = results['accuracy_stds']
    fisher_scores = results['fisher_scores']
    fisher_stds = results['fisher_stds']
    
    # 计算关键发现
    accuracy_diffs = np.diff(accuracies)
    fisher_diffs = np.diff(fisher_scores)
    
    # 找到饱和点
    accuracy_saturation_point = None
    fisher_saturation_point = None
    
    if len(accuracy_diffs) > 0:
        min_improvement_idx = np.where(accuracy_diffs < 0.01)[0]  # 改善小于1%
        if len(min_improvement_idx) > 0:
            accuracy_saturation_point = neuron_counts[min_improvement_idx[0] + 1]
    
    if len(fisher_diffs) > 0:
        min_fisher_improvement_idx = np.where(fisher_diffs < np.max(fisher_scores) * 0.05)[0]
        if len(min_fisher_improvement_idx) > 0:
            fisher_saturation_point = neuron_counts[min_fisher_improvement_idx[0] + 1]
    
    # 保存结果
    count_analysis = {
        'neuron_counts': neuron_counts,
        'accuracies': accuracies,
        'accuracy_stds': accuracy_stds,
        'fisher_scores': fisher_scores,
        'fisher_stds': fisher_stds,
        'max_accuracy': max(accuracies),
        'max_accuracy_neurons': neuron_counts[np.argmax(accuracies)],
        'max_fisher': max(fisher_scores),
        'max_fisher_neurons': neuron_counts[np.argmax(fisher_scores)],
        'accuracy_saturation_point': accuracy_saturation_point,
        'fisher_saturation_point': fisher_saturation_point
    }
    
    np.savez_compressed(
        os.path.join(save_dir, 'neuron_count_analysis.npz'),
        **count_analysis
    )
    
    # 打印关键发现
    print("\n=== 神经元数量分析总结 ===")
    if accuracy_saturation_point:
        print(f"分类准确率饱和点: ~{accuracy_saturation_point} 个神经元")
    if fisher_saturation_point:
        print(f"Fisher信息饱和点: ~{fisher_saturation_point} 个神经元")
    
    print(f"最高分类准确率: {max(accuracies):.3f} ({neuron_counts[np.argmax(accuracies)]} 个神经元)")
    print(f"最高Fisher信息: {max(fisher_scores):.3f} ({neuron_counts[np.argmax(fisher_scores)]} 个神经元)")
    print(f"神经元数量分析结果已保存到 {save_dir}/neuron_count_analysis.npz")


def run_neuron_count_analysis_if_requested(segments, new_labels, rr_neurons, enable_analysis=True):
    """
    运行神经元数量分析（可选）
    
    参数:
    segments: 神经数据片段
    new_labels: 重分类后的标签
    rr_neurons: RR神经元索引
    enable_analysis: 是否启用分析
    """
    if not enable_analysis:
        return
    
    if len(rr_neurons) < 10:
        print("RR神经元数量不足，跳过神经元数量分析")
        return
    
    print("\n" + "="*60)
    print("神经元数量对性能影响分析")
    print("="*60)
    
    # 执行分析
    results = analyze_neuron_count_effect(segments, new_labels, rr_neurons)
    
    # 保存结果
    save_neuron_count_analysis(results, save_dir='results')
    
    return results


# %% ========== 主脚本 ========== 
if __name__ == '__main__':
    print("start neuron data processing!") 
    
    # %% 加载数据
    if cfg.LOADER_VERSION == 'new':
        neuron_data, neuron_pos, trigger_data, stimulus_data = load_data(cfg.DATA_PATH)
    elif cfg.LOADER_VERSION == 'old':
        neuron_index, segments, labels, neuron_pos_old = load_old_version_data(
            cfg.OLD_VERSION_PATHS['neurons'],
            cfg.OLD_VERSION_PATHS['trials'],
            cfg.OLD_VERSION_PATHS['location']
        )
        # 对于旧版数据，需要将加载的segments和labels转换为新版接口的格式
        # 假设旧版segments已经是 (trials, neurons, timepoints) 格式
        # 并且 labels 已经是处理好的标签
        # neuron_data 和 trigger_data 需要从 segments 和 labels 中反推或简化处理
        # 这里为了兼容性，我们假设旧版数据直接提供了 segments, labels, neuron_pos
        # 并且 neuron_data 和 trigger_data 可以从 segments 中提取或不再需要
        # 实际应用中，可能需要更复杂的转换逻辑
        neuron_data = np.mean(segments, axis=2) # 简化处理，仅为兼容后续函数签名
        trigger_data = np.arange(segments.shape[0]) # 简化处理
        stimulus_data = np.zeros((segments.shape[0], 2)) # 简化处理
        # 更新全局的segments和labels，以便后续函数使用
        # 注意：这里直接覆盖了，如果旧版数据和新版数据处理流程差异大，需要更细致的逻辑
        segments = segments
        new_labels = labels # 旧版数据假设标签已处理
        neuron_pos = neuron_pos_old[0:2, :] # 提取前两维
        print("已切换到旧版数据加载模式，部分可视化和分析可能需要调整")
    else:
        raise ValueError("无效的 LOADER_VERSION 配置")

    
    # %% 可视化原始数据
    print("\n=== 原始数据可视化 ===")
    os.makedirs('results/figures', exist_ok=True)
    
    # 设置科研绘图风格
    setup_plot_style()
    
    # 神经活动热图
    visualize_neural_activity_heatmap(neuron_data, "Neural Activity Heatmap", 
                                     save_path='results/figures/neural_activity_heatmap.png')
    
    # 触发信号分布  
    visualize_trigger_distribution(trigger_data, "Stimulus Trigger Distribution",
                                  save_path='results/figures/trigger_distribution.png')
    
    # 刺激数据分布
    visualize_stimulus_data_distribution(stimulus_data, "Stimulus Data Distribution",
                                        save_path='results/figures/stimulus_distribution.png')
    
    # RR神经元空间分布（稍后在RR分析后调用）
    
    # %% 保存神经信号统计信息
    save_neuron_activity_stats(neuron_data, trigger_data, save_dir='results')
    
    # %% 将神经信号划分为trail，并标记label
    # 在分割前先检查原始神经数据的维度
    print(f"\n=== 原始数据维度检查 ===")
    print(f"neuron_data形状: {neuron_data.shape}")
    print(f"neuron_data[0:5, 0:3]: \n{neuron_data[0:5, 0:3]}")
    
    # 如果是旧版数据，segments和new_labels已经加载，跳过重新分割和标签重分类
    if cfg.LOADER_VERSION == 'new':
        segments, labels = segment_neuron_data(neuron_data, trigger_data, stimulus_data)
        # 重新分类标签：类别1和2强度为1的作为第3类
        new_labels = reclassify_labels(stimulus_data)
    
    print(f"标签分布: {np.unique(new_labels, return_counts=True)}")

    # %% 保存示例神经元统计
    # 保存单个神经元统计信息作为示例
    # 确保 segments 和 new_labels 在这里可用
    if 'segments' in locals() and 'new_labels' in locals():
        # 找到一个有效的神经元索引进行保存
        if segments.shape[1] > 0:
            example_neuron_idx = 0 # 使用第一个神经元作为示例
            # 找到一些有效的试次索引
            valid_trial_indices = np.where(new_labels != 0)[0]
            if len(valid_trial_indices) > 3:
                example_trials_idx = valid_trial_indices[:3].tolist() # 取前3个有效试次
            elif len(valid_trial_indices) > 0:
                example_trials_idx = valid_trial_indices.tolist() # 取所有有效试次
            else:
                example_trials_idx = [0] # 如果没有有效试次，就用第一个试次
            
            save_single_neuron_stats(example_neuron_idx, example_trials_idx, segments, save_dir='results')
        else:
            print("没有神经元数据，跳过保存示例神经元统计")
    else:
        print("segments 或 new_labels 未定义，跳过保存示例神经元统计")

    # %% RR神经元提取
    print("\n开始RR神经元筛选...")
    # 导入RR神经元筛选模块
    from rr_neuron_selection import RRNeuronSelector
    # 准备数据进行RR神经元筛选
    trials_for_rr = segments
    labels_for_rr = new_labels
    
    # 性能对比：使用原始方法和快速方法
    print("\n=== 性能对比测试 ===")
    
    # 快速方法
    import time
    fast_results = fast_rr_selection(trials_for_rr, labels_for_rr)
    
    # 如果数据量不大，也可以运行原始方法进行对比
    if trials_for_rr.shape[1] < 1000:  # 神经元数少于1000时才运行原始方法
        print("\n运行原始RR筛选方法...")
        start_time = time.time()
        
        # 创建RR神经元筛选器
        rr_selector = RRNeuronSelector(
            t_stimulus=20,
            l=60,
            whole_length=trials_for_rr.shape[2],
            alpha_fdr=0.005,
            alpha_level=0.05,
            reliability_ratio_threshold=0.8
        )
        
        # 执行RR神经元筛选
        rr_results = rr_selector.select_rr_neurons(trials_for_rr, labels_for_rr)
        original_time = time.time() - start_time
        rr_results['processing_time'] = original_time
        
        print(f"原始方法耗时: {original_time:.2f}秒")
        print(f"加速比: {original_time / fast_results['processing_time']:.1f}x")
        
        # 结果对比
        print(f"\n结果对比:")
        print(f"原始方法RR神经元: {len(rr_results['rr_neurons'])}")
        print(f"快速方法RR神经元: {len(fast_results['rr_neurons'])}")
        
        # 使用原始结果
        final_results = rr_results
    else:
        print("神经元数量较多，只使用快速方法")
        final_results = fast_results
    
    # 更新变量名以保持后续代码兼容
    rr_results = final_results
    
    # 显示筛选结果
    print(f"\n=== RR神经元筛选结果 ===")
    print(f"增强响应神经元: {len(rr_results['enhanced_neurons_union'])} 个")
    print(f"抑制响应神经元: {len(rr_results['suppressed_neurons_union'])} 个")
    print(f"响应性神经元总数: {len(rr_results['response_neurons'])} 个")
    print(f"可靠性神经元总数: {len(rr_results['reliable_neurons'])} 个")
    print(f"最终RR神经元: {len(rr_results['rr_neurons'])} 个")
    
    if len(rr_results['rr_neurons']) > 0:
        print(f"RR神经元索引: {sorted(rr_results['rr_neurons'][:20])}{'...' if len(rr_results['rr_neurons']) > 20 else ''}")
    
    # 保存RR神经元筛选结果
    rr_save_path = os.path.join(cfg.DATA_PATH, 'RR_Neurons_Results.mat')
    scipy.io.savemat(rr_save_path, {
        'rr_neurons': np.array(rr_results['rr_neurons']),
        'response_neurons': np.array(rr_results['response_neurons']),
        'reliable_neurons': np.array(rr_results['reliable_neurons']),
        'enhanced_neurons_union': np.array(rr_results['enhanced_neurons_union']),
        'suppressed_neurons_union': np.array(rr_results['suppressed_neurons_union']),
        'trials_data': trials_for_rr,
        'labels': labels_for_rr,
        'neuron_positions': neuron_pos
    })
    
    print(f"RR神经元结果已保存到: {rr_save_path}")
    
    # 保存神经元空间分布统计
    if neuron_pos.shape[1] > 0:
        save_rr_neurons_distribution(neuron_pos, rr_results, save_dir='results')
        
        # 可视化RR神经元空间分布
        visualize_rr_neurons_spatial_distribution(neuron_pos, rr_results,
                                                save_path='results/figures/rr_neurons_spatial.png')
    
    print("\nRR神经元筛选完成！")

    # %% 分类测试 - 使用预处理功能
    print("\n=== 分类测试 ===")
    
    if len(rr_results['rr_neurons']) == 0:
        print("没有RR神经元，跳过分类步骤")
    else:
        # 提取RR神经元数据
        print(f"使用 {len(rr_results['rr_neurons'])} 个RR神经元进行分类")
        rr_segments = segments[:, rr_results['rr_neurons'], :]
        
        print("\n--- 原始方法分类 ---")
        # 原始分类方法（仅标准化）
        original_X, original_y, _ = preprocess_neural_data(rr_segments, new_labels, method='simple')
        
        original_results = improved_classification(original_X, original_y, 
                                                 test_size=cfg.TEST_SIZE, 
                                                 enable_multiple=False)
        
        if cfg.ENABLE_PREPROCESSING:
            print("\n--- 预处理方法分类 ---")
            # 预处理方法
            processed_X, processed_y, feature_info = preprocess_neural_data(
                rr_segments, new_labels, method=cfg.PREPROCESSING_METHOD)
            
            # 类别平衡
            if cfg.ENABLE_CLASS_BALANCE:
                print("\n处理类别不平衡...")
                balanced_X, balanced_y = handle_class_imbalance(processed_X, processed_y, 
                                                              method=cfg.BALANCE_METHOD)
            else:
                balanced_X, balanced_y = processed_X, processed_y
            
            # 改进分类
            improved_results = improved_classification(balanced_X, balanced_y,
                                                     test_size=cfg.TEST_SIZE,
                                                     enable_multiple=cfg.ENABLE_MULTIPLE_CLASSIFIERS)
            
            # 结果对比
            print(f"\n=== 结果对比 ===")
            print(f"原始方法准确率: {original_results['best_cv_mean']:.3f} ± {original_results['best_cv_std']:.3f}")
            
            if cfg.ENABLE_MULTIPLE_CLASSIFIERS:
                print(f"预处理方法最佳准确率: {improved_results['best_cv_mean']:.3f} ± {improved_results['best_cv_std']:.3f}")
                print(f"最佳模型: {improved_results['best_model']}")
                
                improvement = improved_results['best_cv_mean'] - original_results['best_cv_mean']
                print(f"准确率提升: {improvement:+.3f}")
                
                # 保存最佳模型的混淆矩阵
                best_model = improved_results['best_model']
                best_cm = improved_results['results'][best_model]['confusion_matrix']
                
                # 保存分类结果
                os.makedirs('results', exist_ok=True)
                np.savez_compressed(
                    'results/classification_results.npz',
                    best_model=best_model,
                    confusion_matrix=best_cm,
                    original_accuracy=original_results['best_cv_mean'],
                    improved_accuracy=improved_results['best_cv_mean'],
                    improvement=improved_results['best_cv_mean'] - original_results['best_cv_mean']
                )
                print("分类结果已保存到 results/classification_results.npz")
                
                # 可视化分类效果
                print("\n=== 分类效果可视化 ===")
                visualize_classification_performance(improved_results, 
                                                   save_path='results/figures/classification_performance.png')
                
                # 可视化ROC曲线
                print("生成ROC曲线...")
                # 重新训练模型用于ROC曲线绘制
                classifiers_for_roc = {
                    'SVM': SVC(kernel='rbf', class_weight='balanced', random_state=cfg.RANDOM_STATE, probability=True),
                    'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', 
                                                         random_state=cfg.RANDOM_STATE, max_depth=10),
                    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=cfg.RANDOM_STATE),
                    'LogisticRegression': LogisticRegression(class_weight='balanced', random_state=cfg.RANDOM_STATE, max_iter=1000)
                }
                visualize_roc_curves(balanced_X, balanced_y, classifiers_for_roc,
                                   save_path='results/figures/roc_curves.png')
            else:
                print(f"预处理方法准确率: {improved_results['best_cv_mean']:.3f} ± {improved_results['best_cv_std']:.3f}")
                improvement = improved_results['best_cv_mean'] - original_results['best_cv_mean']
                print(f"准确率提升: {improvement:+.3f}")
        
        else:
            print("预处理功能已禁用，仅使用原始方法")
    
    # %% 按时间点分类分析
    if len(rr_results['rr_neurons']) > 0:
        print("\n=== 按时间点分类分析 ===")
        
        # 执行时间点分类分析
        time_accuracies, time_points = classify_by_timepoints(
            segments, new_labels, rr_results['rr_neurons'])
        
        # 保存结果
        save_accuracy_over_time(time_accuracies, time_points, save_dir='results')
        
        # 可视化时间点分类准确率
        print("\n=== 时间点分析可视化 ===")
        visualize_accuracy_over_time(time_accuracies, time_points,
                                   save_path='results/figures/accuracy_over_time.png')
        
        # 计算Fisher信息
        print("\n=== Fisher信息分析 ===")
        fisher_scores = calculate_fisher_information(segments, new_labels, rr_results['rr_neurons'])
        save_fisher_information(fisher_scores, np.arange(len(fisher_scores)), save_dir='results')
        
        # 可视化Fisher信息
        print("\n=== Fisher信息可视化 ===")
        visualize_fisher_information(fisher_scores, np.arange(len(fisher_scores)),
                                   save_path='results/figures/fisher_information.png')
        
        # 可视化Fisher信息热图
        print("生成Fisher信息热图...")
        visualize_fisher_heatmap(segments, new_labels, rr_results['rr_neurons'],
                               time_window=(cfg.PRE_FRAMES, cfg.PRE_FRAMES + cfg.STIMULUS_DURATION),
                               save_path='results/figures/fisher_heatmap.png')
        
        # 可视化组合分析
        print("\n=== 组合分析可视化 ===")  
        visualize_combined_analysis(time_accuracies, fisher_scores, np.arange(len(fisher_scores)),
                                  save_path='results/figures/combined_analysis.png')
        
        # 神经元数量分析
        print("\n=== 神经元数量对性能影响分析 ===")
        if len(rr_results['rr_neurons']) >= 10:
            neuron_results = run_neuron_count_analysis_if_requested(
                segments, new_labels, rr_results['rr_neurons'], enable_analysis=True
            )
            
            # 可视化神经元数量效果
            if neuron_results is not None:
                print("\n=== 神经元数量效果可视化 ===")
                visualize_neuron_count_effect(
                    neuron_results['neuron_counts'],
                    neuron_results['accuracies'],
                    neuron_results['accuracy_stds'],
                    neuron_results['fisher_scores'],
                    neuron_results['fisher_stds'],
                    save_path='results/figures/neuron_count_effect.png'
                )
        else:
            print("RR神经元数量不足，跳过神经元数量分析")
    
    # %% 可视化总结
    print("\n" + "="*60)
    print("可视化总结")
    print("="*60)
    print("所有可视化图像已保存到 results/figures/ 目录:")
    print("- neural_activity_heatmap.png: 神经活动热图")
    print("- trigger_distribution.png: 刺激触发分布")
    print("- stimulus_distribution.png: 刺激数据分布")
    print("- rr_neurons_spatial.png: RR神经元空间分布")
    
    if len(rr_results['rr_neurons']) > 0:
        print("- classification_performance.png: 分类性能对比")
        print("- roc_curves.png: ROC曲线分析")
        print("- accuracy_over_time.png: 时间点分类准确率")
        print("- fisher_information.png: Fisher信息分析")
        print("- fisher_heatmap.png: Fisher信息热图")
        print("- combined_analysis.png: 准确率与Fisher信息组合分析")
        
        if len(rr_results['rr_neurons']) >= 10:
            print("- neuron_count_effect.png: 神经元数量对性能影响")
    
    print("\n数据分析完成！所有结果和图像已保存。")


# %%