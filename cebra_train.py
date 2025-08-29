# CEBRA训练和可视化脚本 - Docker版本
# guiy24@mails.tsinghua.edu.cn
# 2025-08-29
# 在Docker环境中运行CEBRA并保存可视化结果

# 
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# CEBRA路径配置
CEBRA_PATH = "/app/CEBRA"  # Docker内CEBRA路径
DATA_PATH = "/app/data"    # Docker内数据路径

# 训练配置
class CEBRAConfig:
    # CEBRA模型参数
    MODEL_ARCH = 'offset10-model'
    OUTPUT_DIM = 3           # 降低输出维度用于可视化
    BATCH_SIZE = 256         # 减小batch size适应CPU
    LEARNING_RATE = 3e-4
    TEMPERATURE = 1.0
    MAX_ITERATIONS = 10000   # 增加迭代次数确保收敛
    DEVICE = 'cpu'
    
    # 行为模型专用参数
    BEHAVIOR_OUTPUT_DIM = 8  # 行为模型可以用稍高维度
    TIME_OUTPUT_DIM = 3      # 时间模型用低维度
    
    # 可视化参数
    FIGURE_SIZE = (12, 8)
    SAVE_DPI = 150
    
    # 保存路径
    RESULTS_DIR = "/app/results"

cfg = CEBRAConfig()

def setup_environment():
    """设置CEBRA环境"""
    sys.path.insert(0, CEBRA_PATH)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    print(f"CEBRA路径: {CEBRA_PATH}")
    print(f"数据路径: {DATA_PATH}")
    print(f"结果保存: {cfg.RESULTS_DIR}")

def load_cebra_data(data_type='time'):
    """加载CEBRA数据"""
    cebra_data_dir = os.path.join(DATA_PATH, 'cebra_data')
    
    if data_type == 'time':
        data_file = os.path.join(cebra_data_dir, 'cebra_time_data.npz')
        data = np.load(data_file)
        neural_data = data['neural_data']
        print(f"Time数据统计: shape={neural_data.shape}, min={neural_data.min():.3f}, max={neural_data.max():.3f}")
        return neural_data, None
    
    elif data_type == 'behavior':
        data_file = os.path.join(cebra_data_dir, 'cebra_behavior_data.npz')
        data = np.load(data_file)
        neural_data = data['neural_data'] 
        labels = data['category_labels']  # 使用类别标签而非连续标签
        print(f"Behavior数据统计: neural_shape={neural_data.shape}, labels_shape={labels.shape}")
        print(f"Neural数据: min={neural_data.min():.3f}, max={neural_data.max():.3f}")
        print(f"标签分布: {np.unique(labels, return_counts=True)}")
        return neural_data, labels
    
    elif data_type == 'hybrid':
        data_file = os.path.join(cebra_data_dir, 'cebra_hybrid_data.npz')
        data = np.load(data_file)
        neural_data = data['neural_data']
        behavioral_labels = data['behavioral_labels']
        print(f"Hybrid数据统计: neural_shape={neural_data.shape}, behavioral_shape={behavioral_labels.shape}")
        return neural_data, behavioral_labels
    
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

def train_cebra_time(neural_data):
    """训练CEBRA-Time模型"""
    import cebra
    
    print("训练CEBRA-Time模型...")
    print(f"输入数据形状: {neural_data.shape}")
    
    model = cebra.CEBRA(
        model_architecture=cfg.MODEL_ARCH,
        batch_size=cfg.BATCH_SIZE,
        learning_rate=cfg.LEARNING_RATE,
        temperature=cfg.TEMPERATURE,
        output_dimension=cfg.TIME_OUTPUT_DIM,
        max_iterations=cfg.MAX_ITERATIONS,
        device=cfg.DEVICE,
        verbose=True
    )
    
    model.fit(neural_data)
    embedding = model.transform(neural_data)
    print(f"Time嵌入形状: {embedding.shape}")
    
    return model, embedding

def train_cebra_behavior(neural_data, labels):
    """训练CEBRA-Behavior模型"""
    import cebra
    
    print("训练CEBRA-Behavior模型...")
    print(f"输入数据形状: {neural_data.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"标签分布: {np.unique(labels, return_counts=True)}")
    
    model = cebra.CEBRA(
        model_architecture=cfg.MODEL_ARCH,
        batch_size=cfg.BATCH_SIZE,
        learning_rate=cfg.LEARNING_RATE,
        temperature=cfg.TEMPERATURE,
        output_dimension=cfg.BEHAVIOR_OUTPUT_DIM,
        max_iterations=cfg.MAX_ITERATIONS,
        device=cfg.DEVICE,
        verbose=True
    )
    
    model.fit(neural_data, labels)
    embedding = model.transform(neural_data)
    print(f"Behavior嵌入形状: {embedding.shape}")
    
    return model, embedding

def plot_embedding_2d(embedding, labels=None, title="CEBRA Embedding", save_path=None):
    """绘制2D嵌入结果"""
    plt.figure(figsize=cfg.FIGURE_SIZE)
    
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                       c=[colors[i]], label=f'Category {int(label)}', alpha=0.6)
        plt.legend()
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6)
    
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=cfg.SAVE_DPI, bbox_inches='tight')
        print(f"图像已保存: {save_path}")
    
    plt.close()

def plot_embedding_3d(embedding, labels=None, title="CEBRA Embedding 3D", save_path=None):
    """绘制3D嵌入结果"""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=cfg.FIGURE_SIZE)
    ax = fig.add_subplot(111, projection='3d')
    
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
                      c=[colors[i]], label=f'Category {int(label)}', alpha=0.6)
        ax.legend()
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], alpha=0.6)
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=cfg.SAVE_DPI, bbox_inches='tight')
        print(f"图像已保存: {save_path}")
    
    plt.close()

def save_embedding_data(embedding, labels=None, save_path=None):
    """保存嵌入数据"""
    if save_path:
        if labels is not None:
            np.savez(save_path, embedding=embedding, labels=labels)
        else:
            np.savez(save_path, embedding=embedding)
        print(f"嵌入数据已保存: {save_path}")

def main():
    """主函数"""
    print("=== CEBRA训练脚本启动 ===")
    
    # 环境设置
    setup_environment()
    
    try:
        # 先检查数据可用性
        print("\n0. 数据检查...")
        cebra_data_dir = os.path.join(DATA_PATH, 'cebra_data')
        if not os.path.exists(cebra_data_dir):
            print(f"错误：CEBRA数据目录不存在: {cebra_data_dir}")
            print("请先运行manifold.py生成CEBRA数据")
            return
            
        # 1. CEBRA-Time训练
        print("\n1. CEBRA-Time训练...")
        neural_data, _ = load_cebra_data('time')
        
        model_time, embedding_time = train_cebra_time(neural_data)
        
        # 可视化Time结果
        plot_embedding_2d(embedding_time[:, :2], 
                         title="CEBRA-Time 2D Embedding",
                         save_path=os.path.join(cfg.RESULTS_DIR, "cebra_time_2d.png"))
        
        if embedding_time.shape[1] >= 3:
            plot_embedding_3d(embedding_time[:, :3], 
                             title="CEBRA-Time 3D Embedding",
                             save_path=os.path.join(cfg.RESULTS_DIR, "cebra_time_3d.png"))
        
        save_embedding_data(embedding_time, 
                           save_path=os.path.join(cfg.RESULTS_DIR, "cebra_time_embedding.npz"))
        
        # 2. CEBRA-Behavior训练
        print("\n2. CEBRA-Behavior训练...")
        neural_data, labels = load_cebra_data('behavior')
        
        model_behavior, embedding_behavior = train_cebra_behavior(neural_data, labels)
        
        # 可视化Behavior结果
        plot_embedding_2d(embedding_behavior[:, :2], labels,
                         title="CEBRA-Behavior 2D Embedding",
                         save_path=os.path.join(cfg.RESULTS_DIR, "cebra_behavior_2d.png"))
        
        if embedding_behavior.shape[1] >= 3:
            plot_embedding_3d(embedding_behavior[:, :3], labels,
                             title="CEBRA-Behavior 3D Embedding", 
                             save_path=os.path.join(cfg.RESULTS_DIR, "cebra_behavior_3d.png"))
        
        save_embedding_data(embedding_behavior, labels,
                           save_path=os.path.join(cfg.RESULTS_DIR, "cebra_behavior_embedding.npz"))
        
        print("\n=== CEBRA训练完成 ===")
        print(f"结果已保存到: {cfg.RESULTS_DIR}")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()