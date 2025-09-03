# 神经数据完整分析主脚本
# guiy24@mails.tsinghua.edu.cn
# 一键运行所有分析模块，生成完整的分析结果

import os
import sys
import time
import traceback
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
# 导入配置
from loaddata import cfg

def create_analysis_log():
    """创建分析日志记录"""
    # 使用版本感知的路径管理
    base_dir = cfg.get_results_dir() if hasattr(cfg, 'get_results_dir') else 'results'
    log_dir = os.path.join(base_dir, 'analysis_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'complete_analysis_{timestamp}.log')
    
    return log_file

def log_message(message, log_file=None, print_console=True):
    """记录日志消息"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    if print_console:
        print(log_entry)
    
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')

def run_analysis_module(module_name, description, log_file):
    """运行单个分析模块"""
    log_message(f"开始运行: {description}", log_file)
    start_time = time.time()
    
    try:
        if module_name == 'loaddata':
            # 基础数据处理和RR神经元筛选
            import subprocess
            import sys
            result = subprocess.run([sys.executable, 'loaddata.py'], 
                                  capture_output=True, text=True, cwd='.')
            if result.returncode == 0:
                log_message("✓ 基础数据处理完成", log_file)
            else:
                log_message(f"✗ 基础数据处理失败: {result.stderr}", log_file)
                raise Exception(f"loaddata.py execution failed: {result.stderr}")
            
        elif module_name == 'network':
            # 网络拓扑分析
            import network
            log_message("✓ 网络拓扑分析完成", log_file)
            
        elif module_name == 'advanced_network':
            # 高级网络分析
            from advanced_network_analysis import run_advanced_network_analysis
            run_advanced_network_analysis()
            log_message("✓ 高级网络分析完成", log_file)
            
        elif module_name == 'noise_correlation':
            # 噪声相关性分析
            from noise_correlation_analysis import run_noise_correlation_analysis
            run_noise_correlation_analysis()
            log_message("✓ 噪声相关性分析完成", log_file)
            
        elif module_name == 'degree_centrality':
            # 度中心性与神经信息关系分析
            import degree
            log_message("✓ 度中心性分析完成", log_file)
            
        elif module_name == 'manifold':
            # 流形学习分析
            import manifold
            log_message("✓ 流形学习分析完成", log_file)
            
        
        elapsed = time.time() - start_time
        log_message(f"✓ {description} 完成，耗时: {elapsed:.1f}秒", log_file)
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        log_message(f"✗ {description} 失败，耗时: {elapsed:.1f}秒", log_file)
        log_message(f"错误信息: {str(e)}", log_file)
        log_message(f"详细错误:\n{traceback.format_exc()}", log_file, print_console=False)
        return False

def collect_visualization_results():
    """收集所有可视化结果"""
    visualizations = {}
    seen_files = set()  # 用于去重
    
    # 定义要收集的目录和文件模式，使用版本感知的路径
    results_dir = cfg.get_results_dir() if hasattr(cfg, 'get_results_dir') else 'results'
    figures_dir = cfg.get_figures_dir() if hasattr(cfg, 'get_figures_dir') else 'results/figures'
    
    search_paths = [
        (figures_dir, '基础分析可视化'),
        (os.path.join(results_dir, 'advanced_analysis'), '高级网络分析可视化'),
        (os.path.join(results_dir, 'noise_correlation'), '噪声相关分析可视化'),
        (os.path.join(results_dir, 'centrality_results'), '中心性分析可视化'),
        (os.path.join(results_dir, 'manifold_results'), 't-SNE流形学习可视化')
    ]
    
    for path, category in search_paths:
        if path and os.path.exists(path):
            visualizations[category] = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.png'):
                        full_path = os.path.abspath(os.path.join(root, file))
                        # 使用绝对路径去重
                        if full_path not in seen_files:
                            seen_files.add(full_path)
                            visualizations[category].append({
                                'filename': file,
                                'path': full_path,
                                'description': get_figure_description(file)
                            })
    
    return visualizations

def get_figure_description(filename):
    """根据文件名生成图表描述"""
    descriptions = {
        # 基础分析可视化
        'neural_activity_heatmap.png': '神经活动热图 - 显示神经元活动的时空模式',
        'classification_performance.png': '分类性能对比 - 多种分类器的准确率比较',
        'roc_curves.png': 'ROC曲线分析 - 各分类器的接收者操作特征曲线对比',
        'accuracy_over_time.png': '时间序列分类准确率 - 不同时间点的分类性能',
        'fisher_information.png': 'Fisher信息分析 - 信息编码能力随时间变化',
        'fisher_heatmap.png': 'Fisher信息热图 - 神经元在时间和类别上的信息编码强度',
        'combined_analysis.png': '综合分析 - Fisher信息与分类准确率的联合分析',
        'neuron_count_effect.png': '神经元数量效应 - 神经元数量对性能的影响',
        'rr_neurons_spatial.png': 'RR神经元空间分布 - 响应可靠神经元的空间位置',
        'trigger_distribution.png': '刺激触发分布 - 刺激时间序列的统计特性',
        'stimulus_distribution.png': '刺激数据分布 - 刺激类别和强度的分布',
        'network_topology.png': '网络拓扑结构 - 神经元连接网络的整体结构',
        
        # 中心性分析
        'degree_centrality_vs_information.png': '度中心性与信息关系 - 网络中心性与信息编码的关系',
        'centrality_distributions.png': '中心性指标分布 - 多种中心性指标的统计分布',
        'centrality_relationships.png': '中心性指标相关性 - 不同中心性指标间的关系',
        'multivariate_fisher_regression.png': '多变量Fisher信息回归 - 层级中心性与集合信息编码关系',
        
        # 高级网络分析
        'rich_club_condition_1.png': '富人俱乐部分析(条件1) - 条件1下高度连接节点的集群特性',
        'rich_club_condition_2.png': '富人俱乐部分析(条件2) - 条件2下高度连接节点的集群特性', 
        'rich_club_condition_3.png': '富人俱乐部分析(条件3) - 条件3下高度连接节点的集群特性',
        'pid_condition_multi_condition_breakdown.png': 'PID多条件分解分析 - 信息分解的条件依赖性',
        'pid_condition_multi_condition_components.png': 'PID多条件成分分析 - 各信息成分的条件比较',
        'small_world_analysis.png': '小世界网络分析 - 网络的小世界特性量化',
        
        # 噪声相关分析
        'correlation_matrices.png': '相关性矩阵 - 神经元间信号和噪声相关性对比',
        'hub_peripheral_analysis.png': '中心-外围分析 - 网络中心节点与外围节点的相关性特征',
        'network_metrics_comparison.png': '网络指标比较 - 信号网络与噪声网络的拓扑指标对比',
        'neuron_pairs_scatter.png': '神经元对散点图 - 神经元对间相关性的分布模式',
        'noise_signal_comparison.png': '噪声-信号比较 - 噪声相关与信号相关的定量比较',
        'shuffle_effects.png': '随机化效应 - 数据随机化对相关性分析的影响',
        'noise_correlation_analysis.png': '噪声相关分析 - 神经元间的噪声相关性模式',
        
        # t-SNE流形学习分析
        'tsne_scientific_analysis.png': 't-SNE科研分析 - 神经群体降维的综合科研风格可视化',
        'tsne_publication_figure.png': 't-SNE论文图 - 适合学术发表的神经群体流形可视化',
        'manifold_visualization.png': '流形可视化 - 高维神经数据的低维嵌入',
        
        # 传统降维方法（如果存在）
        'pca_2d_analysis.png': 'PCA二维分析 - 主成分分析降维可视化',
        'pca_3d_analysis.png': 'PCA三维分析 - 主成分分析降维可视化'
    }
    
    # 模糊匹配
    for key, desc in descriptions.items():
        if key.replace('.png', '') in filename.replace('.png', ''):
            return desc
    
    return '分析结果可视化'

def generate_markdown_report(log_file, results_summary, total_elapsed):
    """生成整合的Markdown报告"""
    timestamp = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
    report_file = os.path.join(os.path.dirname(log_file), f'分析报告_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md')
    report_dir = os.path.dirname(report_file)
    
    # 收集可视化结果
    visualizations = collect_visualization_results()
    
    # 读取日志文件中的关键信息
    important_logs = []
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if any(keyword in line for keyword in ['✓', '✗', '相关性分析', 'Fisher信息', '准确率', '神经元数', '网络密度']):
                    important_logs.append(line.strip())
    
    markdown_content = f"""# 神经数据分析完整报告

**分析时间**: {timestamp}  
**总耗时**: {total_elapsed:.1f}秒 ({total_elapsed/60:.1f}分钟)  
**分析版本**: {getattr(cfg, 'LOADER_VERSION', 'unknown')}  

## 📊 分析结果概览

### ✅ 成功完成的分析模块
"""
    
    # 添加成功模块
    successful = [item for item, success in results_summary.items() if success]
    failed = [item for item, success in results_summary.items() if not success]
    
    for item in successful:
        markdown_content += f"- ✅ **{item}**\n"
    
    if failed:
        markdown_content += "\n### ❌ 失败的分析模块\n"
        for item in failed:
            markdown_content += f"- ❌ **{item}**\n"
    
    markdown_content += f"\n**成功率**: {len(successful)}/{len(results_summary)} ({len(successful)/len(results_summary)*100:.1f}%)\n"
    
    # 添加重要日志信息
    if important_logs:
        markdown_content += "\n## 📋 重要分析结果\n\n```\n"
        for log in important_logs[-20:]:  # 只显示最后20条重要日志
            markdown_content += log + "\n"
        markdown_content += "```\n"
    
    # 添加可视化结果
    markdown_content += "\n## 🎨 可视化结果展示\n"
    
    if not visualizations:
        markdown_content += "\n⚠️ 未找到可视化结果文件\n"
    else:
        for category, figures in visualizations.items():
            if figures:
                markdown_content += f"\n### {category}\n"
                for fig in figures:
                    # 使用标准绝对路径，兼容更多Markdown渲染器
                    abs_fig_path = os.path.abspath(fig['path']).replace('\\', '/')
                    
                    markdown_content += f"\n#### {fig['filename']}\n"
                    markdown_content += f"{fig['description']}\n\n"
                    markdown_content += f"![{fig['filename']}]({abs_fig_path})\n\n"
                    markdown_content += f"**图片路径**: `{abs_fig_path}`\n\n"
                    markdown_content += "---\n"
    
    # 添加数据文件信息
    markdown_content += "\n## 📁 生成的数据文件\n"
    
    # 使用版本感知的路径
    results_dir = cfg.get_results_dir() if hasattr(cfg, 'get_results_dir') else 'results'
    base_dirs = [results_dir]
    
    for base_dir in base_dirs:
        if os.path.exists(base_dir):
            markdown_content += f"\n### {base_dir}/ 目录\n"
            data_files = []
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if file.endswith(('.npz', '.mat', '.csv')):
                        relative_path = os.path.relpath(os.path.join(root, file), base_dir)
                        data_files.append(relative_path)
            
            if data_files:
                for file in sorted(data_files):
                    file_desc = get_data_file_description(file)
                    markdown_content += f"- **{file}** - {file_desc}\n"
            else:
                markdown_content += "- 暂无数据文件\n"
    
    # 添加分析方法说明
    markdown_content += f"""

## 🔬 分析方法说明

### 数据预处理
- **RR神经元筛选**: 使用统计检验识别响应可靠的神经元
- **基线校正**: dF/F标准化处理
- **时间窗口**: 基线期{getattr(cfg, 'PRE_FRAMES', 10)}帧，刺激期{getattr(cfg, 'STIMULUS_DURATION', 20)}帧

### 网络分析
- **网络构建**: 基于Pearson相关系数构建功能连接网络
- **拓扑指标**: 度中心性、介数中心性、接近中心性、特征向量中心性
- **高级分析**: 小世界网络特性、富人俱乐部效应

### 信息分析
- **单神经元Fisher信息**: 类间方差/类内方差
- **多变量Fisher信息**: 基于散布矩阵的判别分析
- **分类分析**: 多种机器学习算法的性能比较

### 其他分析
- **噪声相关分析**: 神经元间信号相关性和噪声相关性
- **流形学习**: t-SNE、UMAP等降维可视化方法

## 📞 技术支持

如有问题请联系：guiy24@mails.tsinghua.edu.cn

---
*报告生成时间: {timestamp}*  
*分析工具版本: Neural Analysis Toolkit v1.0*
"""
    
    # 保存Markdown报告
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    # 同时生成HTML版本，图片链接更兼容
    html_file = report_file.replace('.md', '.html')
    generate_html_report(markdown_content, html_file, visualizations)
    
    return report_file

def generate_html_report(markdown_content, html_file, visualizations):
    """生成HTML版本报告，图片显示更兼容"""
    
    # 先处理图片链接，使用file://协议
    html_content = markdown_content
    for category, figures in visualizations.items():
        if figures:
            for fig in figures:
                abs_fig_path = os.path.abspath(fig['path']).replace('\\', '/')
                file_url = f"file:///{abs_fig_path}"
                
                # 替换图片标记为HTML img标签
                img_pattern = f"![{fig['filename']}]({abs_fig_path})"
                img_html = f'<img src="{file_url}" alt="{fig['filename']}" style="max-width:800px;height:auto;" />'
                html_content = html_content.replace(img_pattern, img_html)
    
    # 简单的Markdown到HTML转换（按顺序处理，避免嵌套问题）
    lines = html_content.split('\n')
    html_lines = []
    
    in_code_block = False
    
    for line in lines:
        # 处理代码块
        if line.startswith('```'):
            if not in_code_block:
                html_lines.append('<pre><code>')
                in_code_block = True
            else:
                html_lines.append('</code></pre>')
                in_code_block = False
            continue
            
        if in_code_block:
            html_lines.append(line)
            continue
            
        # 处理标题
        if line.startswith('#### '):
            html_lines.append(f'<h4>{line[5:]}</h4>')
        elif line.startswith('### '):
            html_lines.append(f'<h3>{line[4:]}</h3>')
        elif line.startswith('## '):
            html_lines.append(f'<h2>{line[3:]}</h2>')
        elif line.startswith('# '):
            html_lines.append(f'<h1>{line[2:]}</h1>')
        # 处理粗体
        elif '**' in line:
            line = line.replace('**', '<strong>', 1).replace('**', '</strong>', 1)
            html_lines.append(f'<p>{line}</p>' if line.strip() else '<br>')
        # 处理代码
        elif line.startswith('**图片路径**: `') and line.endswith('`'):
            path_content = line[13:-1]  # 提取路径内容
            html_lines.append(f'<p class="path"><strong>图片路径</strong>: <code>{path_content}</code></p>')
        # 处理分割线
        elif line.strip() == '---':
            html_lines.append('<hr>')
        # 处理列表项
        elif line.startswith('- '):
            html_lines.append(f'<li>{line[2:]}</li>')
        # 处理空行
        elif line.strip() == '':
            html_lines.append('<br>')
        # 处理普通段落
        else:
            if line.strip():
                html_lines.append(f'<p>{line}</p>')
    
    html_content = '\n'.join(html_lines)
    
    # 添加HTML框架
    full_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>神经数据分析完整报告</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', sans-serif; line-height: 1.6; margin: 40px; }}
        h1, h2, h3, h4 {{ color: #333; }}
        img {{ display: block; margin: 20px 0; border: 1px solid #ddd; border-radius: 8px; }}
        code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
        pre {{ background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        hr {{ border: none; border-top: 1px solid #ddd; margin: 30px 0; }}
        .path {{ font-family: monospace; font-size: 0.9em; color: #666; }}
    </style>
</head>
<body>
<p>{html_content}</p>
</body>
</html>"""
    
    # 保存HTML文件
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(full_html)

def get_data_file_description(filename):
    """根据文件名生成数据文件描述"""
    descriptions = {
        'classification_results.npz': '分类结果数据',
        'accuracy_over_time.npz': '时间序列准确率数据',
        'fisher_over_time.npz': 'Fisher信息时间序列数据',
        'neuron_activity_stats.npz': '神经元活动统计数据',
        'neuron_count_analysis.npz': '神经元数量分析数据',
        'rr_neurons_distribution.npz': 'RR神经元分布数据',
        'RR_Neurons_Results.mat': 'RR神经元筛选结果',
        'network_analysis_results.npz': '网络分析结果数据',
        'centrality_analysis.npz': '中心性分析数据',
        'rich_club_analysis.npz': '富人俱乐部分析数据',
        'small_world_analysis.npz': '小世界网络分析数据'
    }
    
    for key, desc in descriptions.items():
        if key in filename:
            return desc
    
    return '分析数据文件'

def generate_analysis_summary(log_file, results_summary):
    """生成分析总结报告"""
    log_message("\n" + "="*80, log_file)
    log_message("分析完成总结报告", log_file)
    log_message("="*80, log_file)
    
    # 成功的分析
    successful = [item for item, success in results_summary.items() if success]
    failed = [item for item, success in results_summary.items() if not success]
    
    log_message(f"成功完成的分析 ({len(successful)}/{len(results_summary)}):", log_file)
    for item in successful:
        log_message(f"  ✓ {item}", log_file)
    
    if failed:
        log_message(f"\n失败的分析 ({len(failed)}/{len(results_summary)}):", log_file)
        for item in failed:
            log_message(f"  ✗ {item}", log_file)
    
    # 生成结果目录信息
    log_message("\n生成的结果目录:", log_file)
    
    # 使用版本感知的路径
    results_dir = cfg.get_results_dir() if hasattr(cfg, 'get_results_dir') else 'results'
    base_dirs = [results_dir]
    
    for base_dir in base_dirs:
        if os.path.exists(base_dir):
            log_message(f"\n{base_dir}/ 目录结构:", log_file)
            for root, dirs, files in os.walk(base_dir):
                level = root.replace(base_dir, '').count(os.sep)
                indent = '  ' * level
                log_message(f"{indent}{os.path.basename(root)}/", log_file)
                subindent = '  ' * (level + 1)
                for file in files:
                    if file.endswith(('.png', '.npz', '.mat', '.csv')):
                        log_message(f"{subindent}{file}", log_file)
    
    log_message("\n" + "="*80, log_file)

def main():
    """主函数：一键运行完整分析流程"""
    print("="*80)
    print("神经数据完整分析流程")
    print("="*80)
    print(f"数据加载版本: {cfg.LOADER_VERSION}")
    results_dir = cfg.get_results_dir() if hasattr(cfg, 'get_results_dir') else 'results'
    print(f"结果保存路径: {results_dir}")
    print("="*80)
    
    # 创建日志文件
    log_file = create_analysis_log()
    log_message("开始完整分析流程", log_file)
    results_dir = cfg.get_results_dir() if hasattr(cfg, 'get_results_dir') else 'results'
    log_message(f"数据加载版本: {cfg.LOADER_VERSION}", log_file)
    log_message(f"结果保存路径: {results_dir}", log_file)
    
    # 分析模块配置
    analysis_modules = [
        ('loaddata', '基础数据处理与RR神经元筛选'),
        ('network', '网络拓扑分析'),
        ('advanced_network', '高级网络分析'),
        ('noise_correlation', '噪声相关性分析'),
        ('degree_centrality', '度中心性与神经信息关系分析'),
        ('manifold', 't-SNE流形学习与降维分析')
    ]
    
    # 执行分析
    total_start_time = time.time()
    results_summary = {}
    
    for module_name, description in analysis_modules:
        log_message(f"\n{'='*60}", log_file)
        success = run_analysis_module(module_name, description, log_file)
        results_summary[description] = success
        
        # 如果基础模块失败，询问是否继续
        if not success and module_name in ['loaddata']:
            response = input(f"\n{description} 失败，这可能影响后续分析。是否继续? (y/N): ")
            if response.lower() != 'y':
                log_message("用户选择停止分析", log_file)
                break
        
        # 模块间休息一下，避免资源冲突
        time.sleep(2)
    
    # 计算总耗时
    total_elapsed = time.time() - total_start_time
    
    # 生成总结报告
    generate_analysis_summary(log_file, results_summary)
    log_message(f"\n总分析时间: {total_elapsed:.1f}秒 ({total_elapsed/60:.1f}分钟)", log_file)
    
    # 生成整合的Markdown报告
    print("\n📝 正在生成整合分析报告...")
    try:
        report_file = generate_markdown_report(log_file, results_summary, total_elapsed)
        html_file = report_file.replace('.md', '.html')
        print(f"✅ Markdown报告已生成: {report_file}")
        print(f"✅ HTML报告已生成: {html_file}")
        print("💡 建议使用HTML版本查看图片，图片显示更稳定")
    except Exception as e:
        print(f"⚠️  整合报告生成失败: {e}")
    
    # 最终提示
    successful_count = sum(results_summary.values())
    total_count = len(results_summary)
    
    print(f"\n{'='*80}")
    print(f"分析流程完成！")
    print(f"成功: {successful_count}/{total_count} 个模块")
    print(f"总耗时: {total_elapsed:.1f}秒 ({total_elapsed/60:.1f}分钟)")
    print(f"详细日志: {log_file}")
    if 'report_file' in locals():
        print(f"Markdown报告: {report_file}")
        print(f"HTML报告: {report_file.replace('.md', '.html')} (推荐用于查看图片)")
    print(f"{'='*80}")
    
    # 如果所有分析都成功，显示主要结果位置
    if successful_count == total_count:
        print("\n🎉 所有分析模块运行成功!")
        print("\n📁 主要结果位置:")
        results_dir = cfg.get_results_dir() if hasattr(cfg, 'get_results_dir') else 'results'
        figures_dir = cfg.get_figures_dir() if hasattr(cfg, 'get_figures_dir') else 'results/figures'
        print(f"├── {figures_dir}/ - 所有可视化图像")
        print(f"├── {results_dir}/ - 数值分析结果(.npz文件)")
        print(f"├── {results_dir}/analysis_logs/ - 运行日志和整合报告")
        print(f"├── {results_dir}/centrality_results/ - 中心性分析结果")
        print(f"├── {results_dir}/advanced_analysis/ - 高级网络分析")
        print(f"└── {results_dir}/noise_correlation/ - 噪声相关分析")
        
        print(f"\n📊 查看整合报告获取完整的可视化结果展示!")
    else:
        print(f"\n⚠️  {total_count - successful_count} 个模块运行失败，请查看日志了解详情")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断分析流程")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n分析流程出现未预期错误: {e}")
        print(f"详细错误信息:\n{traceback.format_exc()}")
        sys.exit(1)