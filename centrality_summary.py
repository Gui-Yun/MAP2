# 网络中心性与神经信息关系分析结果总结
# guiy24@mails.tsinghua.edu.cn
# 2025-08-29

import matplotlib.pyplot as plt
import numpy as np

# 分析结果数据
centrality_metrics = ['Degree', 'Betweenness', 'Closeness', 'Eigenvector', 'PageRank', 'Clustering']
fisher_correlations = [0.963, 0.884, 0.601, 0.951, 0.749, 0.151]
fisher_pvalues = [0.009, 0.046, 0.284, 0.013, 0.145, 0.808]
accuracy_correlations = [0.942, 0.000, 0.297, 0.901, 0.473, 0.498]
accuracy_pvalues = [0.016, 1.000, 0.628, 0.037, 0.422, 0.394]

# 显著性阈值
significance_threshold = 0.05

# 创建总结图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Fisher信息相关性
colors_fisher = ['red' if p < significance_threshold else 'gray' for p in fisher_pvalues]
bars1 = ax1.bar(centrality_metrics, fisher_correlations, color=colors_fisher, alpha=0.7)
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax1.set_ylabel('Correlation with Fisher Information')
ax1.set_title('Network Centrality vs Fisher Information')
ax1.set_ylim(-0.1, 1.0)
ax1.grid(True, alpha=0.3)

# 添加p值标注
for i, (corr, p) in enumerate(zip(fisher_correlations, fisher_pvalues)):
    if p < significance_threshold:
        ax1.text(i, corr + 0.05, f'p={p:.3f}*', ha='center', fontweight='bold')
    else:
        ax1.text(i, corr + 0.05, f'p={p:.3f}', ha='center', alpha=0.7)

# 分类准确率相关性
colors_accuracy = ['red' if p < significance_threshold else 'gray' for p in accuracy_pvalues]
bars2 = ax2.bar(centrality_metrics, accuracy_correlations, color=colors_accuracy, alpha=0.7)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.set_ylabel('Correlation with Classification Accuracy')
ax2.set_title('Network Centrality vs Classification Accuracy')
ax2.set_ylim(-0.1, 1.0)
ax2.grid(True, alpha=0.3)

# 添加p值标注
for i, (corr, p) in enumerate(zip(accuracy_correlations, accuracy_pvalues)):
    if p < significance_threshold:
        ax2.text(i, corr + 0.05, f'p={p:.3f}*', ha='center', fontweight='bold')
    else:
        ax2.text(i, corr + 0.05, f'p={p:.3f}', ha='center', alpha=0.7)

# 旋转x轴标签
for ax in [ax1, ax2]:
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('centrality_analysis_summary.png', dpi=150, bbox_inches='tight')
print("Summary plot saved as 'centrality_analysis_summary.png'")
# plt.show()  # Comment out to avoid GUI timeout

# 打印总结
print("=== 网络拓扑中心性与神经信息关系分析总结 ===")
print("\n主要发现:")
print("1. 度中心性 (Degree): 与Fisher信息强相关 (r=0.963, p=0.009**)")
print("2. 度中心性 (Degree): 与分类准确率强相关 (r=0.942, p=0.016*)")
print("3. 特征向量中心性 (Eigenvector): 与Fisher信息强相关 (r=0.951, p=0.013*)")
print("4. 特征向量中心性 (Eigenvector): 与分类准确率强相关 (r=0.901, p=0.037*)")
print("5. 介数中心性 (Betweenness): 与Fisher信息中等相关 (r=0.884, p=0.046*)")

print("\n结论:")
print("研究假设得到强有力支持：网络拓扑中心性高的神经元承载更多信息内容")
print("- 度中心性和特征向量中心性表现出最强的预测能力")
print("- 这些神经元不仅具有更高的Fisher判别信息，也具有更高的分类准确率")
print("- 网络拓扑结构可以作为识别重要信息处理神经元的有效指标")

print(f"\n网络统计:")
print(f"- 分析神经元数: 343 (RR神经元)")
print(f"- 网络边数: 4,682")
print(f"- 网络密度: 0.0798")
print(f"- 阈值方法: 密度阈值 (保留最强10%连接)")
print(f"- 显著性水平: p < 0.05")