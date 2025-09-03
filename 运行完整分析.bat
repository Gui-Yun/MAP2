@echo off
chcp 65001 >nul 2>&1
title 神经数据完整分析工具

echo.
echo 🧠 神经数据完整分析工具
echo ========================================
echo 一键运行所有分析模块，自动生成整合报告
echo ========================================

cd /d "%~dp0"

echo 🔍 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误：未找到Python环境
    echo 请确保Python已安装并添加到PATH中
    pause
    exit /b 1
)

echo ✅ Python环境正常
echo.

echo 🚀 启动完整分析流程...
echo ========================================
echo 📋 将依次运行以下模块：
echo   1️⃣  基础数据处理与RR神经元筛选
echo   2️⃣  网络拓扑分析
echo   3️⃣  高级网络分析
echo   4️⃣  噪声相关性分析
echo   5️⃣  度中心性与神经信息关系分析
echo   6️⃣  流形学习与降维分析
echo   7️⃣  CEBRA神经动力学分析
echo ========================================
echo.

echo ⏰ 开始时间：%date% %time%
echo.

python run_complete_analysis.py

echo.
echo ========================================
echo ⏰ 完成时间：%date% %time%
echo ========================================
echo.
echo 📊 分析完成！主要结果：
echo.
echo 📁 可视化结果：
echo    └── results/figures/ - 所有分析图表
echo.
echo 📄 数据结果：
echo    └── results/ - .npz 数据文件
echo.
echo 📝 分析报告：
echo    └── [数据路径]/analysis_logs/ - 详细日志和整合报告
echo.
echo 🎯 建议首先查看整合报告(Markdown文件)获取完整结果展示
echo.
pause