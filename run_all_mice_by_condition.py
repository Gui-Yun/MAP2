# 所有小鼠的按条件网络效率分析
import os
import sys
from network_efficiency_by_condition import run_condition_analysis

def run_all_mice_condition_analysis():
    """为所有小鼠运行按条件的网络效率分析"""
    
    # 小鼠数据路径映射
    mice_paths = {
        'M27': r'C:\Users\76629\OneDrive\brain\Micedata\M27_1008',
        'M30': r'C:\Users\76629\OneDrive\brain\Micedata\M30_0420', 
        'M65': r'C:\Users\76629\OneDrive\brain\Micedata\M65_0816',
        'M74': r'C:\Users\76629\OneDrive\brain\Micedata\M74_0816'
    }
    
    print("=" * 80)
    print("所有小鼠按条件网络效率分析")
    print("=" * 80)
    
    successful_analyses = []
    failed_analyses = []
    
    for mouse_name, data_path in mice_paths.items():
        print(f"\n开始分析{mouse_name}...")
        
        if os.path.exists(data_path):
            try:
                results = run_condition_analysis(data_path, mouse_name)
                if results:
                    successful_analyses.append(mouse_name)
                    print(f"✓ {mouse_name}分析成功完成")
                else:
                    failed_analyses.append(f"{mouse_name} (无有效结果)")
                    print(f"✗ {mouse_name}分析失败：无有效结果")
            except Exception as e:
                failed_analyses.append(f"{mouse_name} (异常: {str(e)[:50]})")
                print(f"✗ {mouse_name}分析失败：{e}")
        else:
            failed_analyses.append(f"{mouse_name} (路径不存在)")
            print(f"✗ {mouse_name}数据路径不存在: {data_path}")
    
    # 打印汇总结果
    print("\n" + "=" * 80)
    print("按条件分析汇总结果")
    print("=" * 80)
    print(f"成功分析: {len(successful_analyses)}只小鼠")
    for mouse in successful_analyses:
        print(f"  ✓ {mouse}")
    
    if failed_analyses:
        print(f"\n失败分析: {len(failed_analyses)}只小鼠")
        for failure in failed_analyses:
            print(f"  ✗ {failure}")
    
    print(f"\n按条件网络效率分析完成！")
    
    # 结果保存位置
    from loaddata import cfg
    results_dir = os.path.join(cfg.get_results_dir(), 'network_efficiency_by_condition')
    print(f"所有结果保存在: {results_dir}")
    
    return successful_analyses, failed_analyses

if __name__ == "__main__":
    successful, failed = run_all_mice_condition_analysis()
    
    print("\n主要输出文件类型:")
    print("- *_condition_comparison.png: 条件对比可视化")
    print("- *_all_conditions_summary.npz: 汇总数据文件") 
    print("- *_condition_*_results.npz: 各条件详细结果")