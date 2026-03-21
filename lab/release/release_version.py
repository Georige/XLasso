#!/usr/bin/env python
"""
版本发布工具
自动生成版本说明、打tag、保存实验结果快照
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import subprocess
import shutil
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='版本发布工具')
    parser.add_argument('--version', '-v', type=str, required=True,
                        help='版本号，例如：v2.3.1')
    parser.add_argument('--result-dir', '-d', type=str, required=True,
                        help='对应版本的实验结果目录')
    parser.add_argument('--message', '-m', type=str, default=None,
                        help='版本说明')
    return parser.parse_args()

def run_command(cmd, cwd=None):
    """运行命令并返回输出"""
    print(f"⚙️  执行命令: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        print(f"❌ 命令执行失败: {result.stderr}")
        sys.exit(1)
    return result.stdout.strip()

def main():
    args = parse_args()
    version = args.version
    result_dir = args.result_dir
    message = args.message or f"Release {version}"

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    release_dir = os.path.join(root_dir, 'releases', version)
    os.makedirs(release_dir, exist_ok=True)

    print(f"🚀 开始发布版本 {version}")
    print(f"📂 实验结果目录: {result_dir}")
    print(f"📦 发布目录: {release_dir}")

    # 1. 复制实验结果
    print("\n📋 复制实验结果...")
    result_target = os.path.join(release_dir, 'experiment_results')
    shutil.copytree(result_dir, result_target)

    # 2. 生成版本说明
    print("\n📝 生成版本说明...")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    changelog = run_command("git log --oneline -20", cwd=root_dir)

    readme_content = f"""# {version} 版本发布说明

📅 发布时间: {timestamp}

## 版本概述
{message}

## 主要特性
- ✅ 完整支持三大经典场景：AR(1)相关稀疏回归、二分类偏移变量选择、降维打击场景
- ✅ 支持高斯回归和二分类两种任务类型
- ✅ 优化实验框架：按优先级运行算法、实时保存结果、分算法独立存储
- ✅ 完整指标体系：MSE/准确率/AUC/TPR/FDR/F1/选中变量数/运行时间
- ✅ 自动可视化对比工具

## 实验结果
实验结果保存在 `experiment_results/` 目录下，包含：
- 每个算法的原始结果和汇总结果
- 所有对比图表（plots目录）
- 汇总表格

## 代码变更
最近20次提交记录：
```
{changelog}
```

## 使用说明
1. 运行实验：`python run_simulation_experiments.py --help`
2. 可视化结果：`python visualize_experiments.py --result-dir <结果目录>`
3. 查看最新结果：`ls result/comprehensive_experiments/latest/`
"""

    readme_path = os.path.join(release_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    # 3. 打git tag
    print("\n🏷️  打Git标签...")
    run_command(f"git tag -a {version} -m \"{message}\"", cwd=root_dir)

    # 4. 生成压缩包
    print("\n📦 生成版本压缩包...")
    shutil.make_archive(os.path.join(root_dir, 'releases', f'{version}_release'), 'zip', release_dir)

    print(f"\n🎉 版本 {version} 发布完成!")
    print(f"📦 发布目录: {release_dir}")
    print(f"🗜️  压缩包: releases/{version}_release.zip")
    print(f"🏷️  Git标签已创建: {version}")

if __name__ == "__main__":
    main()
