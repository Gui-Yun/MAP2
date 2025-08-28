# 神经数据预处理
# guiy24@mails.tsinghua.edu.cn
# %% 导入必要的库
import h5py
import os
import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
# %% 处理触发文件
def process_trigger(txt_file, IPD=5, ISI=5, fre=None, min_sti_gap=5):
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
def load_data(data_path, interactive = True):
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
    if interactive:
        print("stimulus data loaded successfully!")
    return neuron_data, neuron_pos, trigger_data['start_edge'], stimulus_data

# %% 预处理数据
def preprocess_data(neuron_data, neuron_pos, trigger_data, stimulus_data, 
                   stim_duration=20, baseline_duration=10):
    """
    神经数据预处理pipeline
    
    参数:
    neuron_data: 神经元数据 (时间点 x 神经元数)
    neuron_pos: 神经元位置
    trigger_data: 刺激触发时间点
    stimulus_data: 刺激标签数据
    stim_duration: 刺激持续时间帧数
    baseline_duration: 基线时间帧数
    
    返回:
    processed_data: 预处理后的特征矩阵
    labels: 标签数组
    feature_info: 特征信息字典
    """
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
    from scipy import signal
    from scipy.stats import zscore
    import warnings
    warnings.filterwarnings('ignore')
    
    print("开始数据预处理...")
    
    # 1. 提取试次数据和标签
    print("1. 提取试次数据...")
    trials_data = []
    labels = []
    
    for i, start_frame in enumerate(trigger_data):
        if i >= len(stimulus_data):
            break
            
        # 确保不越界
        if start_frame + stim_duration <= neuron_data.shape[0]:
            # 提取刺激期数据
            stim_data = neuron_data[start_frame:start_frame+stim_duration, :]
            
            # 提取基线数据 (刺激前)
            baseline_start = max(0, start_frame - baseline_duration)
            baseline_data = neuron_data[baseline_start:start_frame, :]
            
            if baseline_data.shape[0] > 0:
                # 基线归一化: (stim - baseline) / baseline_std
                baseline_mean = np.mean(baseline_data, axis=0)
                baseline_std = np.std(baseline_data, axis=0) + 1e-8  # 避免除零
                
                # dF/F计算
                df_f = (stim_data - baseline_mean) / baseline_mean
                df_f = np.nan_to_num(df_f, 0)  # 处理NaN值
                
                trials_data.append(df_f.flatten())  # 展平为1D特征向量
                
                # 获取标签
                label = stimulus_data.iloc[i]['label'] if 'label' in stimulus_data.columns else 1
                labels.append(label)
    
    if not trials_data:
        raise ValueError("未能提取到有效的试次数据")
    
    X = np.array(trials_data)
    y = np.array(labels)
    
    print(f"   提取到 {len(trials_data)} 个试次")
    print(f"   原始特征维度: {X.shape[1]}")
    print(f"   标签分布: {np.unique(y, return_counts=True)}")
    
    # 2. 噪声处理和平滑
    print("2. 噪声处理...")
    X_denoised = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        # 重塑为时间x神经元矩阵
        trial_matrix = X[i].reshape(stim_duration, -1)
        
        # 时域平滑 (高斯滤波)
        for neuron_idx in range(trial_matrix.shape[1]):
            trial_matrix[:, neuron_idx] = signal.gaussian_filter1d(
                trial_matrix[:, neuron_idx], sigma=0.8)
        
        X_denoised[i] = trial_matrix.flatten()
    
    X = X_denoised
    
    # 3. 方差过滤 - 去除低方差特征
    print("3. 方差过滤...")
    variance_selector = VarianceThreshold(threshold=0.01)  # 去除方差<0.01的特征
    X = variance_selector.fit_transform(X)
    print(f"   方差过滤后特征维度: {X.shape[1]}")
    
    # 4. 数据标准化
    print("4. 数据标准化...")
    scaler = RobustScaler()  # 对异常值更鲁棒
    X_scaled = scaler.fit_transform(X)
    
    # 5. 特征选择 - 选择最有判别力的特征
    print("5. 特征选择...")
    n_features = min(500, X_scaled.shape[1] // 2)  # 选择500个特征或一半特征
    selector = SelectKBest(score_func=f_classif, k=n_features)
    X_selected = selector.fit_transform(X_scaled, y)
    print(f"   特征选择后维度: {X_selected.shape[1]}")
    
    # 6. PCA降维
    print("6. PCA降维...")
    n_components = min(100, X_selected.shape[1], len(y)-1)  # 保留主要成分
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_selected)
    
    explained_ratio = np.sum(pca.explained_variance_ratio_)
    print(f"   PCA后维度: {X_pca.shape[1]}")
    print(f"   解释方差比: {explained_ratio:.3f}")
    
    # 7. Z-score标准化最终特征
    X_final = zscore(X_pca, axis=0)
    X_final = np.nan_to_num(X_final, 0)  # 处理NaN
    
    feature_info = {
        'original_shape': X.shape,
        'final_shape': X_final.shape,
        'variance_selector': variance_selector,
        'scaler': scaler,
        'feature_selector': selector,
        'pca': pca,
        'explained_variance_ratio': explained_ratio,
        'selected_features': selector.get_support(),
        'pca_components': pca.components_
    }
    
    print("预处理完成!")
    print(f"最终特征维度: {X_final.shape}")
    
    return X_final, y, feature_info

def handle_class_imbalance(X, y, method='smote', random_state=42):
    """
    处理类别不平衡问题
    
    参数:
    X: 特征矩阵
    y: 标签数组
    method: 处理方法 ('smote', 'adasyn', 'borderline', 'random_oversample')
    random_state: 随机种子
    
    返回:
    X_resampled: 重采样后的特征矩阵
    y_resampled: 重采样后的标签
    """
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
    from collections import Counter
    
    print(f"原始类别分布: {Counter(y)}")
    
    # 选择重采样策略
    if method == 'smote':
        sampler = SMOTE(random_state=random_state, k_neighbors=min(3, min(Counter(y).values())-1))
    elif method == 'adasyn':
        sampler = ADASYN(random_state=random_state, n_neighbors=min(3, min(Counter(y).values())-1))
    elif method == 'borderline':
        sampler = BorderlineSMOTE(random_state=random_state, k_neighbors=min(3, min(Counter(y).values())-1))
    elif method == 'random_oversample':
        sampler = RandomOverSampler(random_state=random_state)
    else:
        raise ValueError(f"不支持的重采样方法: {method}")
    
    try:
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        print(f"重采样后类别分布: {Counter(y_resampled)}")
        return X_resampled, y_resampled
    except Exception as e:
        print(f"重采样失败: {e}")
        print("使用随机过采样作为备选方案...")
        sampler = RandomOverSampler(random_state=random_state)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        print(f"重采样后类别分布: {Counter(y_resampled)}")
        return X_resampled, y_resampled

def improved_classification(X, y, test_size=0.3, random_state=42):
    """
    改进的分类流程
    
    参数:
    X: 预处理后的特征矩阵
    y: 标签数组
    test_size: 测试集比例
    random_state: 随机种子
    
    返回:
    results: 分类结果字典
    """
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    from sklearn.utils.class_weight import compute_class_weight
    import warnings
    warnings.filterwarnings('ignore')
    
    print("开始改进的分类测试...")
    
    # 分层划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
    
    # 计算类权重
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    print(f"类权重: {class_weight_dict}")
    
    # 定义多个分类器
    classifiers = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100, 
            class_weight='balanced',
            random_state=random_state,
            max_depth=10,
            min_samples_split=5
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=random_state,
            max_depth=5
        ),
        'SVM': SVC(
            kernel='rbf', 
            class_weight='balanced',
            random_state=random_state,
            probability=True
        ),
        'LogisticRegression': LogisticRegression(
            class_weight='balanced',
            random_state=random_state,
            max_iter=1000,
            C=1.0
        )
    }
    
    results = {}
    
    # 交叉验证设置
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    for name, clf in classifiers.items():
        print(f"\n=== {name} ===")
        
        # 训练模型
        clf.fit(X_train, y_train)
        
        # 预测
        y_pred = clf.predict(X_test)
        
        # 测试准确率
        test_acc = accuracy_score(y_test, y_pred)
        
        # 交叉验证
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"测试准确率: {test_acc:.3f}")
        print(f"交叉验证准确率: {cv_mean:.3f} ± {cv_std:.3f}")
        
        # 分类报告
        report = classification_report(y_test, y_pred)
        print("分类报告:")
        print(report)
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        print("混淆矩阵:")
        print(cm)
        
        results[name] = {
            'model': clf,
            'test_accuracy': test_acc,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
    
    # 找出最佳模型
    best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
    print(f"\n最佳模型: {best_model_name}")
    print(f"最佳交叉验证准确率: {results[best_model_name]['cv_mean']:.3f}")
    
    results['best_model'] = best_model_name
    results['X_train'] = X_train
    results['X_test'] = X_test
    results['y_train'] = y_train
    results['y_test'] = y_test
    
    return results

def run_complete_pipeline(data_path):
    """
    运行完整的预处理和分类流程
    
    参数:
    data_path: 数据路径
    
    返回:
    results: 完整结果
    """
    print("=== 开始完整流程 ===")
    
    # 1. 加载数据
    neuron_data, neuron_pos, trigger_data, stimulus_data = load_data(data_path)
    trigger_data = trigger_data[1:180]  # 保持180维，去掉首尾各1个
    
    # 2. 预处理
    X_processed, y, feature_info = preprocess_data(
        neuron_data, neuron_pos, trigger_data, stimulus_data)
    
    print(f"\n原始数据 - 特征: {feature_info['original_shape']}, 标签: {len(y)}")
    print(f"预处理后 - 特征: {feature_info['final_shape']}, 解释方差: {feature_info['explained_variance_ratio']:.3f}")
    
    # 3. 处理类别不平衡
    X_balanced, y_balanced = handle_class_imbalance(X_processed, y, method='smote')
    
    # 4. 分类测试
    results = improved_classification(X_balanced, y_balanced)
    
    return {
        'feature_info': feature_info,
        'original_data': (X_processed, y),
        'balanced_data': (X_balanced, y_balanced),
        'classification_results': results
    }

# %%
if __name__ == '__main__':
    print("start neuron data processing!")
    # %%
    data_path = r'E:\Cloud\桂沄\我的资料库\Micedata\M74_0816'
    neuron_data, neuron_pos, trigger_data, stimulus_data = load_data(data_path)
    # 保持180维，去掉首尾各1个
    trigger_data = trigger_data[1:180]
    # %% 简单可视化一下原始神经信号
    def plot_neuron_data(neuron_data, trigger_data, stimulus_data):
        
        plt.figure(figsize=(12, 6))
        # 神经元数量太大，选择一些神经元
        plt.plot(neuron_data.T, color='gray')
        plt.title('Original Neuron Signals')
        plt.xlabel('Time (frames)')
        # 用红色虚线在trigger_data每个位置画竖直线
        for t in trigger_data:
            plt.axvline(x=t, color='red', linestyle='--', linewidth=1)
        plt.ylabel('Neurons')
        plt.show()
    plot_neuron_data(neuron_data[:,100], trigger_data, stimulus_data)
    # %% 将神经信号划分为trail，并标记label



# %%
