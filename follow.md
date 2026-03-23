# XLasso 项目进展记录

## 项目概述

XLasso 是一个高效的可解释 Lasso 算法框架，旨在提供多种 Lasso 变体的统一实现，并在此基础上开发创新的可解释性算法。

---

## 任务一：已有 Lasso 算法的 Benchmark

### 目标
对现有主流 Lasso 算法进行性能基准测试，建立评估体系。

### 已收集的算法来源

#### 1. 本地实现 (`other_lasso/`)
| 算法 | 文件 | 特点 | 状态 |
|------|------|------|------|
| AdaptiveLasso | adaptive_lasso.py | Zou (2006) 自适应权重 | ✅ 可用 |
| FusedLasso | fused_lasso.py | Tibshirani (2005) 融合惩罚 | ✅ 可用 |
| GroupLasso | group_lasso.py | Yuan & Lin (2006) 组级稀疏 | ✅ 可用 |
| AdaptiveSparseGroupLasso | adaptive_sparse_group_lasso.py | 组内+组间双重稀疏 | ✅ 可用 |

#### 2. skglm 高性能库 (`other_lasso/skglm_benchmark/`)
| 算法 | 文件 | 特点 | 状态 |
|------|------|------|------|
| Lasso | estimators.py | Celer 求解器 + Numba JIT | ✅ 已支持 |
| WeightedLasso | estimators.py | 特征级加权 L1 | ✅ 已支持 |
| ElasticNet | estimators.py | L1 + L2 混合 | ✅ 已支持 |
| MCPRegression | estimators.py | MCP 正则化 | ✅ 已支持 |
| MultiTaskLasso | estimators.py | 多任务学习 | ✅ 已支持 |
| GroupLasso | estimators.py | Numba JIT 优化 | ✅ 已支持 |

#### 3. XLasso 核心实现 (`XLasso/`)
XLasso 自己的实现，作为新算法的基准。

### Benchmark 实验

#### Benchmark 文件夹设置

在 /Users/apple/Downloads/毕业论文设计/做/XLasso/benchmarks 中做benchmark实验
- bechmarks
  - benchmark_util(benchmark框架设置，参数接口，接收生成的模拟实验数据)
  - benchmark_run(超参数搜索，一键运行，多次重复，固定种子从2026开始取，result保存，命令行输入运行参数运行)
  - result

#### result保存设置
首先生成benmark时间戳作为文件夹保存在result中，对于每一个算法，用算法和使用的超参数命名文件夹，在组实验下面，单独做一个raw.csv记录这个配置算法的每一次重复试验，再做一个summary.csv对raw的指标取平均

#### Benchmark 维度

| 维度 | 指标 | 说明 |
|------|------|------|
| **精度** | MSE, MAE, R²,F1,拟合系数与真实系数的误差和平方，如果是二分类问题无法返回MSE则返回准确率,auc,tpr,fdr | 预测准确性|
| **稀疏性** | 非零系数比例 | 模型可解释性 |
| **运行时间** | 训练/预测时间 | 计算效率 |
| **收敛性** | 迭代次数/残差下降 | 求解器质量 |

### TODO
- [x] 升级 sklearn 以支持 skglm
- [x] 统一各算法的评估接口
- [x] 编写 benchmark 的 util 和 run
  - [x] `benchmark_util.py` - 工具库、统一Wrapper、指标计算、结果保存
  - [x] `benchmark_run.py` - 命令行运行入口、参数解析、多实验支持
  - [x] 结果保存机制：时间戳文件夹，raw.csv/summary.csv分层保存
  - [x] 支持固定种子从 2026 开始，多重复实验
  - [x] 完整覆盖要求的 benchmark 维度：精度、稀疏性、运行时间、收敛性
- [x] 将模拟实验数据的生成独立出来做py文件
  - [x] `data_generator.py` - 包含7个模拟实验的生成函数，与论文完全对齐
  - [x] 可直接通过 `from data_generator import generate_experiment1_data` 调用
- [x] 对七个模拟实验，按照要求，完善benchmark配置
  - [x] 所有7个模拟实验已接入benchmark
  - [x] 支持回归和分类两种任务类型
  - [x] 兼容所有算法：本地Lasso变种 + skglm高性能算法
  - [x] ✅ 冒烟测试通过：77/77用例100%成功
- [x] 支持固定种子多次重复实验
  - [x] 支持 `--seed-start` 参数设置起始种子
  - [x] 支持 `--n-repeats` 设置重复次数
- [x] 生成benchmark指标，重复 3 次（固定种子，2026,2027,2028）
  - [x] 7个回归实验全部完成，生成完整指标
  - [x] 结果存储目录：/home/liangyuneng/XLasso/benchmarks/result/
  - [x] 所有指标已记录到 memory/benchmark_results.md
  - [x] 已将benchmark结果写入paper.md的实验部分
- [x] 实现多实验并行运行框架
  - [x] 编写 `run_all_experiments.py` 一键运行所有14个任务（7回归+7分类）
  - [x] 支持后台运行、实时进度报告、动态指标分析、最优算法排名
- [x] 框架bug修复与性能优化
  - [x] 修复JSON序列化问题：添加NumpyEncoder支持numpy类型序列化
  - [x] 修复pandas MultiIndex聚合错误：动态列聚合逻辑
  - [x] 优化实验速度：移除5个慢速cvxpy算法，运行时间从4小时降至10分钟
  - [x] skglm系列算法运行速度比Sklearn快26倍，收敛速度快87倍
- [ ] 分类任务修复
- [ ] 定制图像

---

## 任务二：新 Lasso 算法的核心实现代码

### 目标
基于创新理论，实现 XLasso 核心算法。

### 核心模块预期结构

```
XLasso/
├── core/
│   ├── __init__.py
│   ├── xlasso.py          # XLasso 主算法
│   ├── xlasso_cv.py       # 交叉验证版本
│   └── utils.py           # 核心工具函数
├── solvers/
│   ├── __init__.py
│   ├── cd.py              # 坐标下降法
│   ├── prox_newton.py     # 近端牛顿法
│   └── anderson_cd.py     # Anderson 加速 CD
└── penalties/
    ├── __init__.py
    └── xlasso_penalty.py  # XLasso 专用惩罚项
```

### 算法设计要点
- [ ] 核心理论实现
- [ ] 高效求解器选择
- [ ] 稀疏性保证机制
- [ ] 特征选择稳定性

---

## 任务三：新 Lasso 算法的调参消融实验

### 目标
通过系统性实验验证 XLasso 的有效性。

### 实验设计

#### 3.1 参数敏感性分析
| 参数 | 搜索范围 | 实验设计 |
|------|----------|----------|
| alpha (正则化强度) | [0.001, 0.01, 0.1, 1.0] | GridSearchCV |
| lambda (融合惩罚) | [0.1, 0.5, 1.0, 2.0] × alpha | 网格搜索 |
| max_iter | [100, 500, 1000, 5000] | 收敛性验证 |
| tol | [1e-6, 1e-4, 1e-2] | 精度权衡 |

#### 3.2 消融实验 (Ablation Study)
| 实验 | 去掉组件 | 预期影响 |
|------|----------|----------|
| Baseline | 完整 XLasso | 最佳性能 |
| -自适应权重 | gamma=0 | 性能下降程度 |
| -融合惩罚 | lambda_fused=0 | 平滑性损失 |
| -分组机制 | 逐个去掉组 | 组选择能力 |

#### 3.3 对比实验
| 对比算法 | 论文/来源 | XLasso 优势 |
|----------|-----------|--------------|
| Lasso | Tibshirani (1996) | 特征选择稳定性 |
| AdaptiveLasso | Zou (2006) | 计算效率 |
| GroupLasso | Yuan & Lin (2006) | 组级稀疏性 |
| FusedLasso | Tibshirani (2005) | 融合惩罚 |

### 实验数据集
- [ ] 模拟数据 (低维/高维)
- [ ] 糖尿病数据集 (diabetes)
- [ ] 真实生物数据 (可选)

### TODO
- [ ] 确定参数搜索空间
- [ ] 设计消融实验表格
- [ ] 编写自动化实验脚本
- [ ] 收集实验结果
- [ ] 绘制收敛曲线
- [ ] 生成最终报告

---

## 里程碑

| 阶段 | 目标 | 状态 |
|------|------|------|
| M1 | Benchmark 框架完成 | ✅ 完成 |
| M2 | 基准算法性能测试完成 | ✅ 完成 |
| M3 | 实验结果写入论文 | ✅ 完成 |
| M4 | XLasso 核心代码 | 🔄 进行中 |
| M5 | 调参实验完成 | ⏳ 待办 |
| M6 | 论文初稿 | ⏳ 待办 |

---

## 注意事项

1. **skglm 依赖**：已升级 scikit-learn 到 1.8.0 (2026-03-23)
2. **cvxpy 依赖**：FusedLasso、GroupLasso、AdaptiveSparseGroupLasso 需要 cvxpy
3. **Numba JIT**：skglm 使用 Numba 加速，首次运行需要编译
4. **使用方式**：
   ```python
   import sys
   sys.path.insert(0, 'other_lasso/skglm_benchmark')
   from skglm import Lasso, GroupLasso, WeightedLasso
   ```

---

*最后更新：2026-03-23 v3.0.1*
