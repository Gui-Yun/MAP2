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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
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

    if interactive:
        print("stimulus data loaded successfully!")
    return neuron_data, neuron_pos, trigger_data['start_edge'], stimulus_data

# %%
if __name__ == '__main__':
    print("start neuron data processing!")
    # %%
    data_path = r'E:\Cloud\桂沄\我的资料库\Micedata\M74_0816'
    neuron_data, neuron_pos, trigger_data, stimulus_data = load_data(data_path)
    # 保持180维，去掉首尾各1个
    trigger_data = trigger_data[1:181]

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
    def segment_neuron_data(neuron_data, trigger_data, stimulus_data, pre_frames=20, post_frames=500, baseline_correct=True):
        """
        改进的数据分割函数
        
        参数:
        pre_frames: 刺激前的帧数（用于基线）
        post_frames: 刺激后的帧数（用于反应）
        baseline_correct: 是否进行基线校正 (ΔF/F)
        """
        total_frames = pre_frames + post_frames
        segments = np.zeros((len(trigger_data)-1, neuron_data.shape[1], total_frames))
        labels = []
        
        for i in range(len(trigger_data) - 1):
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

    segments, labels = segment_neuron_data(neuron_data, trigger_data, stimulus_data, pre_frames=20, post_frames=80, baseline_correct=True)
    
    # %% 验证标签对齐
    print(f"验证数据对齐:")
    print(f"trigger数量: {len(trigger_data)-1}")  # -1因为我们用的是len(trigger_data)-1
    print(f"stimulus数量: {len(stimulus_data)}")
    print(f"提取的labels数量: {len(labels)}")
    print(f"提取的segments数量: {segments.shape[0]}")
    
    if len(labels) != len(stimulus_data):
        print("⚠️ 警告: labels数量与stimulus数量不匹配!")
    
    # 显示前10个标签和stimulus的对应关系
    print(f"前10个stimulus类别: {stimulus_data[:10, 0] if len(stimulus_data) >= 10 else stimulus_data[:, 0]}")
    print(f"前10个提取的labels: {labels[:10] if len(labels) >= 10 else labels}")
    
    # %% 重新分类标签：类别1和2强度为1的作为第3类
    def reclassify_labels(stimulus_data):
        new_labels = []
        for i in range(len(stimulus_data)):  
            category = stimulus_data[i, 0]  
            intensity = stimulus_data[i, 1]  
            
            if intensity == 1:
                new_labels.append(3)  # 强度为1的噪音刺激作为第3类
            elif category == 1 and intensity != 1:
                new_labels.append(1)  # 类别1且强度不为1
            elif category == 2 and intensity != 1:
                new_labels.append(2)  # 类别2且强度不为1
            else:
                new_labels.append(0)  # 其他情况标记为0（会被过滤）
        
        return np.array(new_labels)
    
    new_labels = reclassify_labels(stimulus_data)
    print(f"标签分布: {np.unique(new_labels, return_counts=True)}")
    
    # %% 数据预处理：使用原始神经元发放
    def preprocess_segments_raw(segments, n=10000, selection_method='mean_activity'):
        """
        改进的特征预处理函数
        
        参数:
        segments: 分割的神经数据
        n: 选择的神经元数量
        selection_method: 选择方法 ('variance', 'mean_activity', 'random')
        """
        n_trials, n_neurons, n_timepoints = segments.shape
        print(f"原始数据形状: {segments.shape}")
        
        # 基于方差或活动强度选择神经元
        if n < n_neurons:
            if selection_method == 'variance':
                # 计算每个神经元在所有trial和时间点上的方差
                neuron_variance = np.var(segments.reshape(n_trials * n_timepoints, n_neurons), axis=0)
                neuron_indices = np.argsort(neuron_variance)[-n:]  # 选择方差最大的n个
                print(f"基于方差选择了{n}个神经元，方差范围: {neuron_variance[neuron_indices].min():.4f} - {neuron_variance[neuron_indices].max():.4f}")
                
            elif selection_method == 'mean_activity':
                # 计算每个神经元的平均活动强度
                neuron_activity = np.mean(np.abs(segments.reshape(n_trials * n_timepoints, n_neurons)), axis=0)
                neuron_indices = np.argsort(neuron_activity)[-n:]  # 选择活动最强的n个
                print(f"基于平均活动选择了{n}个神经元，活动范围: {neuron_activity[neuron_indices].min():.4f} - {neuron_activity[neuron_indices].max():.4f}")
                
            elif selection_method == 'random':
                np.random.seed(42)  # 设置随机种子保证可重现性
                neuron_indices = np.random.choice(n_neurons, n, replace=False)
                print(f"随机选择了{n}个神经元")
                
            else:
                raise ValueError(f"未知的选择方法: {selection_method}")
        else:
            neuron_indices = np.arange(n_neurons)
            print(f"使用所有{n_neurons}个神经元")
            
        # 重新组织数据：(trials, timepoints, neurons)
        segments_transposed = np.transpose(segments, (0, 2, 1))
        segments_selected = segments_transposed[:, :, neuron_indices]
        
        print(f"最终数据形状: {segments_selected.shape}")
        return segments_selected

    # 提取原始特征 - 使用方差选择最活跃的神经元
    segments_raw = preprocess_segments_raw(segments, n=5000, selection_method='mean_activity')
    n_trials, n_timepoints, n_neurons = segments_raw.shape
    
    # 过滤掉标签为0的样本
    valid_indices = new_labels != 0
    segments_valid = segments_raw[valid_indices]
    y_valid = new_labels[valid_indices]
    
    print(f"有效试验数: {segments_valid.shape[0]}")
    print(f"时间点数: {segments_valid.shape[1]}")
    print(f"神经元数: {segments_valid.shape[2]}")
    print(f"有效标签分布: {np.unique(y_valid, return_counts=True)}")
    
    # %% 改进的时间点分类
    def train_svm_timewise(segments_valid, y_valid):
        """改进的SVM时间点分类函数"""
        n_trials, n_timepoints, n_neurons = segments_valid.shape
        
        # 分割训练集和测试集
        train_idx, test_idx = train_test_split(range(len(y_valid)), 
                                               test_size=0.3,  # 增加测试集比例
                                               random_state=42, 
                                               stratify=y_valid)
        
        segments_train = segments_valid[train_idx]
        segments_test = segments_valid[test_idx]
        y_train = y_valid[train_idx]
        y_test = y_valid[test_idx]
        
        print(f"训练集大小: {len(y_train)}, 测试集大小: {len(y_test)}")
        print(f"训练集标签分布: {np.unique(y_train, return_counts=True)}")
        print(f"测试集标签分布: {np.unique(y_test, return_counts=True)}")
        
        accuracies = []
        models = []
        scalers = []
        
        print(f"对每个时间点({n_timepoints}个)进行分类...")
        
        for t in range(n_timepoints):
            # 当前时间点的特征矩阵
            X_train_t = segments_train[:, t, :]  # (n_train_trials, n_neurons)
            X_test_t = segments_test[:, t, :]    # (n_test_trials, n_neurons)
            
            # 检查是否有NaN或Inf
            if np.any(np.isnan(X_train_t)) or np.any(np.isinf(X_train_t)):
                print(f"警告: 时间点{t}的训练数据包含NaN或Inf")
                X_train_t = np.nan_to_num(X_train_t)
                X_test_t = np.nan_to_num(X_test_t)
            
            # 特征标准化 - 只有在方差不为0时才标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_t)
            X_test_scaled = scaler.transform(X_test_t)
            
            # 改进的SVM参数
            svm_model = SVC(kernel='rbf', 
                          C=1.0,  # 可以尝试调整
                          gamma='scale',  # 自动缩放
                          random_state=42, 
                          probability=True,
                          class_weight='balanced')  # 平衡类别权重
            
            try:
                svm_model.fit(X_train_scaled, y_train)
                y_pred = svm_model.predict(X_test_scaled)
                accuracy = np.mean(y_pred == y_test)
            except Exception as e:
                print(f"时间点{t}训练失败: {e}")
                accuracy = 0.0
                svm_model = None
            
            accuracies.append(accuracy)
            models.append(svm_model)
            scalers.append(scaler)
            
            if (t + 1) % 10 == 0:
                print(f"时间点 {t+1}/{n_timepoints} 完成，准确率: {accuracy:.3f}")
        
        return {
            'accuracies': np.array(accuracies),
            'models': models,
            'scalers': scalers,
            'y_test': y_test,
            'train_idx': train_idx,
            'test_idx': test_idx
        }
    
    # 结果可视化：准确率随时间变化的曲线
    def plot_timewise_accuracy(timewise_results, y_valid):
        plt.figure(figsize=(16, 8))
        
        # 准确率随时间变化曲线
        plt.subplot(2, 2, 1)
        time_points = range(1, len(timewise_results['accuracies']) + 1)
        plt.plot(time_points, timewise_results['accuracies'], 'b-', linewidth=2, alpha=0.7)
        plt.axhline(y=timewise_results['accuracies'].mean(), color='r', linestyle='--', 
                   label=f'平均准确率: {timewise_results["accuracies"].mean():.3f}')
        plt.fill_between(time_points, timewise_results['accuracies'], alpha=0.3)
        plt.xlabel('时间点')
        plt.ylabel('分类准确率')
        plt.title('SVM分类准确率随时间变化')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 1)
        
        # 准确率分布直方图
        plt.subplot(2, 2, 2)
        plt.hist(timewise_results['accuracies'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(timewise_results['accuracies'].mean(), color='r', linestyle='--', linewidth=2,
                   label=f'平均: {timewise_results["accuracies"].mean():.3f}')
        plt.xlabel('准确率')
        plt.ylabel('频次')
        plt.title('准确率分布直方图')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 类别分布
        plt.subplot(2, 2, 3)
        unique_labels = np.unique(y_valid)
        label_counts = [np.sum(y_valid == label) for label in unique_labels]
        bars = plt.bar(unique_labels, label_counts, color=['lightcoral', 'lightblue', 'lightgreen'])
        plt.title('类别分布')
        plt.xlabel('类别 (1:类别1, 2:类别2, 3:噪音)')
        plt.ylabel('样本数量')
        for bar, count in zip(bars, label_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom')
        
        # 最高和最低准确率时间点标记
        plt.subplot(2, 2, 4)
        plt.plot(time_points, timewise_results['accuracies'], 'b-', linewidth=2, alpha=0.7)
        
        # 标记最高准确率点
        max_idx = timewise_results['accuracies'].argmax()
        plt.scatter(max_idx + 1, timewise_results['accuracies'][max_idx], 
                   color='red', s=100, zorder=5, label=f'最高: {timewise_results["accuracies"][max_idx]:.3f}')
        
        # 标记最低准确率点  
        min_idx = timewise_results['accuracies'].argmin()
        plt.scatter(min_idx + 1, timewise_results['accuracies'][min_idx], 
                   color='orange', s=100, zorder=5, label=f'最低: {timewise_results["accuracies"][min_idx]:.3f}')
        
        plt.xlabel('时间点')
        plt.ylabel('分类准确率')
        plt.title('关键时间点标记')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
    
    # 检查是否有足够的类别进行分类
    if len(np.unique(y_valid)) >= 2:
        print("开始每个时间点的SVM分类...")
        timewise_results = train_svm_timewise(segments_valid, y_valid)
        
        print(f"\n各时间点准确率统计:")
        print(f"平均准确率: {timewise_results['accuracies'].mean():.3f}")
        print(f"最高准确率: {timewise_results['accuracies'].max():.3f} (时间点 {timewise_results['accuracies'].argmax()+1})")
        print(f"最低准确率: {timewise_results['accuracies'].min():.3f} (时间点 {timewise_results['accuracies'].argmin()+1})")
        
        plot_timewise_accuracy(timewise_results, y_valid)
        
    else:
        print("数据中类别不足，无法进行分类")
    
# %%
