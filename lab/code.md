# 实验运行说明

## 实验1：高维成对相关稀疏回归

### 运行命令
```bash
cd /home/lili/lyn/clear/XLasso/lab
# 默认运行5次重复实验
python experiment_001.py

# 指定重复次数（例如运行50次正式实验）
python experiment_001.py --n-repeats 50
# 或简写
python experiment_001.py -n 50
```

### 命令行参数说明
| 参数 | 缩写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| --n-repeats | -n | int | 5 | 实验重复次数 |
| --sigmas | - | float列表 | 0.5 1.0 2.5 | 噪声标准差列表（多个值用空格分隔） |
| --family | - | str | gaussian | GLM家族类型：gaussian/binomial/poisson/multinomial/cox |
| --n-folds | - | int | 5 | 交叉验证折数 |
| --seed | - | int | 42 | 随机种子 |
| --backend | - | str | numba | 求解器后端：numba（高速）/pytorch（灵活） |
| --group-penalty | - | float | 5.0 | 组一致性惩罚强度 |
| --corr-threshold | - | float | 0.7 | 分组相关系数阈值 |
| --weight-method | - | str | p_value | 显著性权重计算方法：t_statistic/p_value/correlation |
| --alpha | - | float | 1.0 | XLasso alpha参数（显著变量负惩罚强度） |
| --beta | - | float | 1.0 | XLasso beta参数（不显著变量惩罚强度） |

---

## 实验2：AR(1)相关稀疏回归

### 运行命令
```bash
cd /home/lili/lyn/clear/XLasso/lab
# 默认运行5次重复实验
python experiment_002.py

# 指定重复次数和AR(1)相关系数
python experiment_002.py --n-repeats 50 --rho 0.8
```

### 命令行参数说明
**公共参数和实验1完全一致，额外新增参数：**
| 参数 | 缩写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| --rho | - | float | 0.8 | AR(1)相关系数，控制变量间相邻相关性强度 |

### 编程习惯约定
> **所有实验脚本均采用「配置+命令行参数」设计模式**：
> - 所有实验参数、超参数均可通过命令行传入，无需修改代码
> - 支持灵活调整实验设置，方便批量实验、参数调优和结果复现
> - 保持配置集中管理，避免硬编码参数散落在代码各处
> - 运行前自动打印完整配置信息，保证实验可追溯

> **算法开发调试习惯**：
> - 当自定义实现和第三方库行为不一致时，优先检查数值尺度/单位是否匹配
> - 正则化参数lambda的尺度具有库特异性，不同求解器实现可能相差几个数量级
> - 调试时优先打印关键中间变量的数值范围，快速定位尺度不匹配问题

---

## 问题调试记录
### 问题1：XLasso系列方法系数全为0（2026-03-18）
**问题现象**：XLasso系列方法（软约束、自适应惩罚、组约束）运行结果所有系数均为0，无法选出任何变量。

**排查过程**：
1. 尝试降低alpha/beta参数到0.1，无效
2. 调整lmda_min_ratio到1e-6，无效
3. 手动指定lambda路径到1e-8极小区间，无效
4. 对比fit_unilasso（使用adelie库）和cv_uni（自定义实现）的lambda路径，发现自定义实现的lambda比adelie小100倍左右
5. 检查梯度下降和近端算子的阈值计算，确认lambda尺度不匹配是根本原因

**修复方案**：
在`cv_uni`和`fit_uni`函数中，将`_configure_lmda_path`计算得到的lambda路径统一放大100倍，与adelie grpnet的正则化尺度对齐。

**修复效果**：
所有XLasso系列方法均能正常选出变量，软约束方法MSE=35.74，与基准UniLasso的35.48性能接近，验证了修复的正确性。

**根因深度分析**：
Q: 为什么同样使用`_configure_lmda_path`计算lambda路径，自定义求解器和adelie库会有尺度差异？
A: 差异来自求解器的工程实现，而非算法设计：
1. 计算逻辑一致：两者都调用同一个`_configure_lmda_path`函数，基于"最大相关系数/n"的行业标准计算lambda_max，算法设计层面没有差异
2. 求解器实现不同：
   - adelie的grpnet是坐标下降实现，内部对lambda有内置的归一化/缩放处理，属于库的黑箱优化
   - 自定义梯度下降求解器中，阈值计算引入了学习率因子`tau = lr * lmda`，默认lr=0.01相当于将lambda的有效作用缩小了100倍
3. 为什么选择放大lambda而不是修改计算逻辑？
   - 保持标准性：`_configure_lmda_path`是行业通用标准实现，不需要修改
   - 最小侵入性：仅做尺度适配，不改变XLasso的核心算法特性
   - 效果等价：缩放lambda和缩放阈值/学习率的数学效果完全一致，放大lambda是最简单的适配方式

### 输出说明
运行过程中会实时打印每个方法的指标结果，运行结束后会在 `result/exp_001` 目录下生成两个文件：
- `experiment_001_results.csv`：所有重复实验的详细原始数据
- `experiment_001_summary.csv`：按信噪比和方法分组的平均结果汇总表
