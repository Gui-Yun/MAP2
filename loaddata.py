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
    
    # 数据路径
    DATA_PATH = r'F:\brain\Micedata\M65_0816'
    
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
    ALPHA_FDR = 0.005        # FDR校正阈值
    ALPHA_LEVEL = 0.05       # 显著性水平
    RELIABILITY_THRESHOLD = 0.5  # 可靠性阈值
    
    # 快速RR筛选参数
    EFFECT_SIZE_THRESHOLD = 0.5   # 效应大小阈值
    SNR_THRESHOLD = 1.0          # 信噪比阈值
    RESPONSE_RATIO_THRESHOLD = 0.45  # 响应比例阈值
    
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

# %% 加载数据
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
    stimulus_data = stimulus_data[start_idx:end_idx, :]
    
    if interactive:
        print("stimulus data loaded successfully!")
        print(f"Using trials {start_idx} to {end_idx-1}, total: {len(start_edges)} trials")
    
    return neuron_data, neuron_pos, start_edges, stimulus_data

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
        baseline_mean = np.mean(baseline_data, axis=2, keepdims=True)  # (trials, neurons, 1)
        baseline_std = np.std(baseline_data, axis=2, keepdims=True) + 1e-8
        
        # dF/F 归一化
        stimulus_data = (stimulus_data - baseline_mean) / baseline_mean
        stimulus_data = np.nan_to_num(stimulus_data, 0)
    
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

# %%
if __name__ == '__main__':
    print("start neuron data processing!") 
    # %% 加载数据
    neuron_data, neuron_pos, trigger_data, stimulus_data = load_data(cfg.DATA_PATH)
    


    # %% 简单可视化一下原始神经信号
    def plot_neuron_data(neuron_data, trigger_data, stimulus_data):
        
        plt.figure(figsize=cfg.FIGURE_SIZE_LARGE)
        # 神经元数量太大，选择一些神经元
        plt.plot(neuron_data.T, color='gray')
        plt.title('Original Neuron Signals')
        plt.xlabel('Time (frames)')
        # 用红色虚线在trigger_data每个位置画竖直线
        for t in trigger_data:
            plt.axvline(x=t, color='red', linestyle='--', linewidth=1)
        plt.ylabel('Neural Activity')
        plt.show()
    plot_neuron_data(neuron_data[:,14858], trigger_data, stimulus_data)
    # %% 将神经信号划分为trail，并标记label
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
                # 避免除零错误
                baseline = np.where(baseline == 0, 1e-6, baseline)
                segment = (segment - baseline) / baseline
            
            segments[i] = segment.T
            labels.append(stimulus_data[i, 0])
            
        return segments, labels

    # 在分割前先检查原始神经数据的维度
    print(f"\n=== 原始数据维度检查 ===")
    print(f"neuron_data形状: {neuron_data.shape}")
    print(f"neuron_data[0:5, 0:3]: \n{neuron_data[0:5, 0:3]}")
    
    segments, labels = segment_neuron_data(neuron_data, trigger_data, stimulus_data)
    
    # %% 验证标签对齐
    # 简单绘制一下单个神经元所有trail的平均神经活动
    def plot_simple_neuron(neuron_idx, trials_idx, segments):
        plt.figure(figsize=cfg.FIGURE_SIZE_TINY)
        # 绘制所有的trial，颜色为浅灰色
        for trial_idx in trials_idx:
            plt.plot(segments[trial_idx, neuron_idx, :], label=f'Trial {trial_idx}')
        # 绘制上面所有trial的平均发放，颜色为黑色，加粗
        plt.plot(np.mean(segments[trials_idx, neuron_idx, :], axis=0), color='black', linewidth=2, label='Mean')
        plt.title(f'Neuron {neuron_idx} Activity')
        plt.xlabel('Time (frames)')
        plt.ylabel('Neural Activity')
        plt.legend()
        plt.show()

    plot_simple_neuron(14858, [10, 20, 30], segments)
    # 重新分类标签：类别1和2强度为1的作为第3类
    def reclassify_labels(stimulus_data):
        new_labels = []
        for i in range(len(stimulus_data)):  
            category = stimulus_data[i, 0]  
            intensity = stimulus_data[i, 1]  
            
            if intensity == cfg.NOISE_INTENSITY:
                new_labels.append(3)  # 强度为1的噪音刺激作为第3类
            # elif category == 1 and intensity != cfg.NOISE_INTENSITY:
            elif category == 1 and intensity == 0:
                new_labels.append(1)  # 类别1且强度不为1
            # elif category == 2 and intensity != cfg.NOISE_INTENSITY:
            elif category == 2 and intensity == 0:
                new_labels.append(2)  # 类别2且强度不为1
            else:
                new_labels.append(0)  # 其他情况标记为0（会被过滤）
        
        return np.array(new_labels)
    
    new_labels = reclassify_labels(stimulus_data)
    print(f"标签分布: {np.unique(new_labels, return_counts=True)}")
    
    # %% RR神经元提取
    print("\n开始RR神经元筛选...")
    # 导入RR神经元筛选模块
    from rr_neuron_selection import RRNeuronSelector
    # 准备数据进行RR神经元筛选
    # 将segments转换为RR筛选所需的格式: (trials, neurons, timepoints)
    # trials_for_rr = np.transpose(segments, (0, 2, 1))  # 从(trials, timepoints, neurons)转为(trials, neurons, timepoints)
    trials_for_rr = segments
    labels_for_rr = new_labels
    # 快速RR神经元筛选函数
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
    
    # 可选：可视化RR神经元的空间分布
    def plot_rr_neurons_distribution(neuron_pos, rr_results):
        """可视化RR神经元的空间分布"""
        plt.figure(figsize=(15, 5))
        
        # 所有神经元位置
        plt.subplot(1, 3, 1)
        plt.scatter(neuron_pos[0, :], neuron_pos[1, :], c='lightgray', alpha=0.5, s=1)
        plt.title(f'All Neurons (n={neuron_pos.shape[1]})')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.axis('equal')
        
        # 响应性神经元位置
        if len(rr_results['response_neurons']) > 0:
            plt.subplot(1, 3, 2)
            plt.scatter(neuron_pos[0, :], neuron_pos[1, :], c='lightgray', alpha=0.3, s=1)
            response_idx = rr_results['response_neurons']
            plt.scatter(neuron_pos[0, response_idx], neuron_pos[1, response_idx], 
                       c='blue', alpha=0.7, s=3)
            plt.title(f'Responsive Neurons (n={len(response_idx)})')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.axis('equal')
        
        # RR神经元位置
        if len(rr_results['rr_neurons']) > 0:
            plt.subplot(1, 3, 3)
            plt.scatter(neuron_pos[0, :], neuron_pos[1, :], c='lightgray', alpha=0.3, s=1)
            rr_idx = rr_results['rr_neurons']
            plt.scatter(neuron_pos[0, rr_idx], neuron_pos[1, rr_idx], 
                       c='red', alpha=0.8, s=3)
            plt.title(f'RR Neurons (n={len(rr_idx)})')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.axis('equal')
        
        plt.tight_layout()
        plt.show()
    
    # 绘制神经元空间分布图
    if neuron_pos.shape[1] > 0:
        plot_rr_neurons_distribution(neuron_pos, rr_results)
    
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
                
                # 可视化最佳模型的混淆矩阵
                best_model = improved_results['best_model']
                best_cm = improved_results['results'][best_model]['confusion_matrix']
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=['Category 1', 'Category 2', 'Category 3'],
                           yticklabels=['Category 1', 'Category 2', 'Category 3'])
                plt.title(f'{best_model} Classification Confusion Matrix (Preprocessed)')
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.show()
            else:
                print(f"预处理方法准确率: {improved_results['best_cv_mean']:.3f} ± {improved_results['best_cv_std']:.3f}")
                improvement = improved_results['best_cv_mean'] - original_results['best_cv_mean']
                print(f"准确率提升: {improvement:+.3f}")
        
        else:
            print("预处理功能已禁用，仅使用原始方法")
    

# %%
