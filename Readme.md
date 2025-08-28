# MAP2 - 神经数据分析处理项目

## 项目概述

MAP2是一个综合的神经数据分析处理框架，专门用于处理双光子显微镜记录的神经元活动数据。该项目实现了完整的数据处理流水线，包括数据加载、预处理、RR神经元筛选、降维分析和CEBRA嵌入准备等功能。

## 项目结构

```
MAP2/
├── loaddata.py          # 核心数据处理模块
├── manifold.py          # 降维分析和CEBRA数据处理
├── rr_neuron_selection.py # RR神经元筛选算法
└── README.md           # 项目文档
```

## 主要模块详细说明

### 1. loaddata.py - 核心数据处理模块

#### 功能概述
- 神经数据加载和预处理
- 触发信号处理和刺激事件对齐
- RR（Reliable and Responsive）神经元筛选
- 时间点分类分析
- Fisher信息计算

#### 主要类和配置

##### Config类 - 统一参数配置
```python
class Config:
    # 数据路径
    DATA_PATH = r'F:\brain\Micedata\M74_0816'
    
    # 数据分割参数
    PRE_FRAMES = 10           # 刺激前帧数（基线期）
    POST_FRAMES = 40          # 刺激后帧数（响应期）
    STIMULUS_DURATION = 20    # 刺激持续时间（帧数）
    
    # RR神经元筛选参数
    ALPHA_FDR = 0.05         # FDR校正阈值
    RELIABILITY_THRESHOLD = 0.5  # 可靠性阈值
    EFFECT_SIZE_THRESHOLD = 0.5  # 效应大小阈值
    
    # 分类参数
    SVM_KERNEL = 'rbf'       # SVM核函数
    SVM_C = 1.0              # SVM正则化参数
    CV_FOLDS = 5             # 交叉验证折数
```

#### 核心函数详解

##### 1. process_trigger() - 触发信号处理
```python
def process_trigger(txt_file, IPD=5, ISI=5, fre=None, min_sti_gap=5)
```
**功能**: 处理触发文件，提取刺激时间点并映射到相机帧
**参数**:
- `txt_file`: 触发文件路径
- `IPD`: 刺激呈现时长(秒)
- `ISI`: 刺激间隔(秒)
- `fre`: 相机帧率，None则自动估计
- `min_sti_gap`: 相邻刺激最小间隔(秒)

**返回**: 包含start_edge, end_edge, stimuli_array的字典

##### 2. load_data() - 数据加载
```python
def load_data(data_path, start_idx=1, end_idx=181, interactive=True)
```
**功能**: 加载神经数据、触发数据和刺激数据
**参数**:
- `data_path`: 数据目录路径
- `start_idx/end_idx`: 试验范围索引
- `interactive`: 是否显示加载进度

**返回**: neuron_data, neuron_pos, trigger_data, stimulus_data

##### 3. segment_neuron_data() - 数据分割
```python
def segment_neuron_data(neuron_data, trigger_data, stimulus_data, 
                       pre_frames=10, post_frames=40, baseline_correct=True)
```
**功能**: 将连续神经数据按试验分割，并进行基线校正
**参数**:
- `neuron_data`: 神经活动时间序列 (时间点, 神经元数)
- `trigger_data`: 刺激触发时间点数组
- `stimulus_data`: 刺激信息数组
- `baseline_correct`: 是否进行ΔF/F基线校正

**返回**: segments (试验, 神经元, 时间点), labels

##### 4. fast_rr_selection() - 快速RR神经元筛选
```python
def fast_rr_selection(trials, labels, t_stimulus=10, l=20, 
                     alpha_fdr=0.05, reliability_threshold=0.5)
```
**功能**: 使用优化算法快速筛选RR神经元
**方法**: 
- 向量化计算替代循环
- 基于效应大小和信噪比的响应性检测
- 批量统计检验

**返回**: 包含rr_neurons, response_neurons等的结果字典

##### 5. classify_by_timepoints() - 时间点分类分析
```python
def classify_by_timepoints(segments, labels, rr_neurons)
```
**功能**: 分析每个时间点的分类准确率，识别信息量最高的时间窗口
**方法**: 对每个时间点使用SVM进行交叉验证分类

##### 6. calculate_fisher_information() - Fisher信息分析
```python
def calculate_fisher_information(segments, labels, rr_neurons)
```
**功能**: 计算各时间点的Fisher信息，衡量类别可分离性
**公式**: Fisher比率 = 类间方差 / 类内方差

#### 预处理流水线

##### preprocess_neural_data() - 综合预处理
支持两种模式:
- **simple**: 仅标准化
- **comprehensive**: 完整预处理流水线
  1. dF/F归一化
  2. 高斯滤波降噪
  3. 方差过滤
  4. 鲁棒标准化
  5. 特征选择
  6. PCA降维

### 2. manifold.py - 降维分析和CEBRA处理

#### 功能概述
- PCA和t-SNE降维分析
- 多维数据可视化
- CEBRA格式数据准备和保存
- 基于类别和强度的颜色编码

#### 主要功能模块

##### 1. 降维分析
```python
def perform_pca(X, n_components=3)          # PCA降维
def perform_tsne(X, n_components=2)         # t-SNE降维
```

##### 2. 数据可视化
```python
def plot_manifold_2d(X_reduced, labels, stimulus_data)  # 2D可视化
def plot_manifold_3d(X_reduced, labels, stimulus_data)  # 3D可视化
```
**颜色编码规则**:
- 类别1: 红色系 (深红#FF0000 / 浅红#FFB3B3)
- 类别2: 蓝色系 (深蓝#0000FF / 浅蓝#B3B3FF)
- 类别3: 绿色系 (深绿#00AA00 / 浅绿#B3FFB3)
- 强度0: 浅色(低饱和度), 强度1: 深色(高饱和度)

##### 3. CEBRA数据处理
```python
def prepare_cebra_data(segments, stimulus_data, rr_neurons=None)
```
**功能**: 将试验结构数据转换为CEBRA兼容的时间序列格式
**输出格式**:
- **CEBRA-Time**: 纯时间序列，用于无监督学习
- **CEBRA-Behavior**: 离散标签，用于有监督学习  
- **CEBRA-Hybrid**: 连续标签，用于混合模式

```python
def save_cebra_data(cebra_data, base_path)
```
**保存文件**:
- `cebra_time_data.npz`: 时间序列数据
- `cebra_behavior_data.npz`: 行为标签数据
- `cebra_hybrid_data.npz`: 混合模式数据
- `trial_structure_data.npz`: 原始试验结构
- `metadata.json`: 元数据信息
- `README.md`: 使用说明

### 3. rr_neuron_selection.py - RR神经元筛选

#### RRNeuronSelector类
实现完整的RR神经元筛选算法，包括:
- 响应性检测 (Mann-Whitney U检验)
- 可靠性分析 (试验间一致性)
- FDR多重比较校正
- 效应大小计算

## 数据格式要求

### 输入数据格式
1. **神经数据**: `wholebrain_output.mat`
   - `whole_trace_ori`: (时间点, 神经元) 神经活动矩阵
   - `whole_center`: (4, 神经元) 神经元空间坐标

2. **触发数据**: `.txt`文件格式
   ```
   时间戳    通道值    绝对时间戳
   0.0000    1        timestamp
   0.0333    1        timestamp
   2.1000    2        timestamp  # 刺激触发
   ```

3. **刺激数据**: `.csv`文件格式
   ```
   类别,强度
   1,0      # 类别1，强度0
   2,1      # 类别2，强度1
   ```

### 输出数据格式
- **RR神经元结果**: `RR_Neurons_Results.mat`
- **降维结果**: `manifold_results/`
- **CEBRA数据**: `cebra_data/`

## 使用示例

### 基本数据处理流程
```python
from loaddata import load_data, segment_neuron_data, reclassify_labels, fast_rr_selection, cfg

# 1. 加载数据
neuron_data, neuron_pos, trigger_data, stimulus_data = load_data(cfg.DATA_PATH)

# 2. 数据分割
segments, labels = segment_neuron_data(neuron_data, trigger_data, stimulus_data)

# 3. 标签重分类
new_labels = reclassify_labels(stimulus_data)

# 4. RR神经元筛选
rr_results = fast_rr_selection(segments, new_labels)
```

### 降维分析
```python
from manifold import prepare_data_for_manifold, perform_pca, perform_tsne, plot_manifold_2d

# 1. 准备数据
X, y = prepare_data_for_manifold(segments, labels, rr_results['rr_neurons'])

# 2. 降维
X_pca, pca = perform_pca(X, n_components=3)
X_tsne = perform_tsne(X, n_components=2)

# 3. 可视化
plot_manifold_2d(X_tsne, y, stimulus_data, title="t-SNE 2D Manifold")
```

### CEBRA数据准备
```python
from manifold import prepare_cebra_data, save_cebra_data

# 1. 准备CEBRA数据
cebra_data = prepare_cebra_data(segments, stimulus_data, rr_results['rr_neurons'])

# 2. 保存数据
save_cebra_data(cebra_data, 'path/to/cebra_output')
```

## 参数调优指南

### RR神经元筛选参数
- `EFFECT_SIZE_THRESHOLD`: 增大以提高筛选严格性
- `RELIABILITY_THRESHOLD`: 调节可靠性要求
- `ALPHA_FDR`: FDR校正的显著性水平

### 分类参数
- `SVM_C`: SVM正则化强度，过拟合时减小
- `CV_FOLDS`: 交叉验证折数，数据量大时可增加
- `TEST_SIZE`: 测试集比例

### 可视化参数
- `PCA_COMPONENTS`: PCA降维目标维度
- `TSNE_PERPLEXITY`: t-SNE复杂度参数
- `TSNE_LEARNING_RATE`: t-SNE学习率

## 常见问题解决

### 1. 数据加载错误
- 检查文件路径是否正确
- 确认mat文件包含必要的数据集
- 验证触发文件格式

### 2. 内存不足
- 减少`MAX_FEATURES`参数
- 使用`fast_rr_selection`替代原始方法
- 分批处理大数据集

### 3. 分类效果差
- 增加RR神经元数量 (降低筛选阈值)
- 启用预处理功能
- 调整分类器参数

### 4. 可视化问题
- 检查标签格式是否正确
- 确认颜色编码参数设置
- 验证降维数据维度

## 性能优化

- **快速RR筛选**: 相比原始方法可提速10-50倍
- **向量化计算**: 大量使用NumPy向量操作
- **内存优化**: 适当的数据类型转换和内存释放
- **并行处理**: 支持多核心分类器训练

## 依赖库版本

```
numpy >= 1.19.0
scipy >= 1.5.0
matplotlib >= 3.3.0
seaborn >= 0.11.0
scikit-learn >= 0.24.0
pandas >= 1.1.0
h5py >= 2.10.0
```

## 更新日志

- v1.0: 初始版本，基础数据处理功能
- v1.1: 添加快速RR筛选算法
- v1.2: 集成降维分析和可视化
- v1.3: 添加CEBRA数据处理支持
- v1.4: 完善预处理流水线和文档

## 联系方式

如有问题或建议，请联系：guiy24@mails.tsinghua.edu.cn