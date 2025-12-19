#!/bin/bash
# 快速运行示例脚本

echo "======================================"
echo "充电桩覆盖收益最大化问题求解示例"
echo "======================================"
echo ""

# 检查Python是否可用
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到python3，请先安装Python"
    exit 1
fi

# 运行示例
echo "运行示例脚本..."
python3 example.py

echo ""
echo "======================================"
echo "示例运行完成！"
echo "======================================"
echo ""
echo "更多使用方法："
echo "  1. 运行所有求解器: python3 main.py --generate --methods all"
echo "  2. 运行特定方法: python3 main.py --generate --methods greedy ga"
echo "  3. 查看帮助: python3 main.py --help"
