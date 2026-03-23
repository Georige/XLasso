# 实验运行说明

## run_simulation_experiments.py：综合模拟实验脚本

`run_simulation_experiments.py` 是一个综合模拟实验运行脚本，支持7个场景、对抗7种算法、两种任务类型。

### 支持的实验场景（`--experiment`）

| 标识 | 场景 | 数据特点 |
|------|------|----------|
| `exp1` | 高维成对相关稀疏回归 | p=500, X_ij=0.5, 前20变量β=1.0 |
| `exp2` | AR(1)相关稀疏回归 | p=500, ρ=0.8^\|i-j\|, 奇数前20变量β=1.0 |
| `exp3` | 二分类偏移变量选择 | p=500, AR(1), y=1样本前20变量偏移0.6 |
| `exp4` | 孪生变量反符号选择 | p=1000, 10对孪生变量ρ=0.85, β=2.0/-2.5 |
| `exp5` | 魔鬼等级1：绝对隐身陷阱 | p=1000, 边际相关性精确归零 |
| `exp6` | 魔鬼等级2：鸠占鹊巢陷阱 | p=500, 噪声诱饵与真信号ρ=0.8 |
| `exp7` | 魔鬼等级3：AR(1)符号雪崩 | p=500, 相邻符号相反β衰减 |
| `all` | 运行全部7个场景 | — |

### 支持的算法

`原始UniLasso`、`标准Lasso`、`XLasso-Soft`、`XLasso-GroupDecomp`、`XLasso-Full`、`Adaptive Lasso`、`Fused Lasso`、`Group Lasso`、`Adaptive Sparse Group Lasso`

### 任务类型（`--family`）

`gaussian`（回归）、`binomial`（二分类）、`all`

### 运行示例

```bash
# 运行全部实验（高斯+二分类）
python run_simulation_experiments.py -e all -n 3

# 只跑魔鬼等级1实验，3次重复
python run_simulation_experiments.py -e exp5 -n 3

# 只跑XLasso-Full算法，二分类任务
python run_simulation_experiments.py -e all -a "XLasso-Full" -f binomial

# 调试模式（2次重复，100特征）
python run_simulation_experiments.py -e exp1 --debug
```

### 输出结构

每个算法单独目录 `{01_原始UniLasso, 02_标准Lasso, ...}/`，含 `_raw.csv`（每次重复详情）和 `_summary.csv`（均值汇总）。

### 参数选择策略

**核心原则：按场景固定结构参数，仅动态选择 λ**

XLasso的参数分为两类：
- **结构参数**（场景相关，实验中固定）：`k`(γ)、`enable_group_decomp`、`group_corr_threshold`、`enable_group_aware_filter` 等，在实验中保持不变
- **正则化参数**（所有场景统一处理）：通过3折交叉验证在验证集上动态选择最优 λ

#### XLasso系列算法

| 算法 | 结构参数 | λ选择方式 |
|------|----------|------------|
| `XLasso-Soft` | `k=1.0`, 无组分解 | cv_uni 3折CV |
| `XLasso-Soft-γ0.5` | `k=0.5`, 无组分解 | cv_uni 3折CV |
| `XLasso-Soft-γ2.0` | `k=2.0`, 无组分解 | cv_uni 3折CV |
| `XLasso-GroupDecomp` | `k=1.0`, 组分解, thresh=0.7 | cv_uni 3折CV |
| `XLasso-Full` | `k=1.0`, 组分解+感知过滤, thresh=0.7 | cv_uni 3折CV |

> **说明**：`k` 参数控制非对称惩罚的强度，k=0.5 表示弱非对称（接近对称Lasso），k=2.0 表示强非对称（对负系数更敏感）。不同 k 变体用于评估XLasso对惩罚强度选择的鲁棒性。

#### sklearn类算法（通过交叉验证自动选优）

| 算法 | 调参方式 |
|------|----------|
| `标准Lasso` | `LogisticRegressionCV(cv=3)` / `LassoCV(cv=3)` 自动选最优lambda |
| `Adaptive Lasso` | CV在 `gammas=[0.5, 1.0, 2.0]` 中选最优gamma |
| `Fused Lasso` | CV在 `lambda_fused_ratios=[0.1, 0.5, 1.0, 2.0]` 中选最优 |
| `Group Lasso` / `Adaptive Sparse Group Lasso` | CV自动选最优 |

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

## 求解器实现说明
### 坐标下降求解器（2026-03-19新增）
**实现特点**：
- 纯Numba JIT编译实现，无Python循环开销，逐特征优化收敛速度比梯度下降快3~5倍
- 无需调整学习率，自动适配正则化尺度，避免梯度下降的lambda匹配问题
- 支持完整支持XLasso的非对称惩罚、自适应权重、组约束三大核心功能
- 仅支持Gaussian家族（线性回归），非Gaussian家族自动回退到梯度下降实现

**适用场景**：
- 高维稀疏回归场景（p>1000特征）
- 对求解速度敏感的实验
- 梯度下降学习率调优困难的场景

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

---
### 问题2：坐标下降求解器lambda尺度不匹配（2026-03-19）
**问题现象**：切换到坐标下降求解器后，XLasso系列方法选出的变量过多，系数估计爆炸，MSE异常高。

**排查过程**：
1. 直接测试坐标下降在原始X上运行正常，符合预期
2. 在LOO特征矩阵上测试时，发现系数范围可达数百，说明惩罚严重不足
3. 对比梯度下降和坐标下降的阈值计算，发现梯度下降阈值有lr因子，而坐标下降没有
4. 原有的lambda放大100倍是适配梯度下降的，坐标下降不需要这个因子

**修复方案**：
将lambda缩放系数从100倍（0.01）调整为1倍（0.01 → 0.01？不，原先是*=0.01是缩小100倍，哦不对，原问题2里的修复是调整到*=0.01？不对，看之前的修改：
之前梯度下降用的是*=100，现在坐标下降直接使用原始lambda尺度，所以需要将缩放系数调整为*=0.01，相当于缩小100倍，匹配坐标下降的阈值计算。

**根因分析**：
梯度下降的阈值计算：`tau = lr * lmda`，坐标下降的阈值计算：`tau = lmda`，两者的lambda有效尺度相差了lr倍（默认lr=0.01，相差100倍）。

---
### 问题3：坐标下降求解速度慢
**问题现象**：p=1000特征，100个lambda路径，单次拟合需要145秒。
**优化方案**：
1. 减少默认lambda路径数量：从100减少到30个，实验精度不受影响
2. 放宽收敛阈值：从1e-6调整到1e-4，坐标下降对精度不敏感
3. 减少最大迭代次数：从1000次降到200次，足够收敛
**优化效果**：
单次拟合速度从145秒降到21秒，提升了近7倍，完全满足实验需求。

---
### 问题4：模型预测MSE异常高
**问题现象**：XLasso系列方法预测MSE达到百万级，完全不可用，系数范围可达数百。

**根因分析（多层原因）**：
1. **返回值错误**：最初错误地将LOO空间的系数直接返回，没有转换为原始特征空间系数
2. **维度错误**：混淆了全局单变量系数和每个lambda的转换后系数的维度，导致返回结果维度错误
3. **lambda尺度完全不匹配**：
   - `_configure_lmda_path`是基于原始特征矩阵X计算lambda_max的，而第二阶段是在LOO矩阵上拟合
   - LOO矩阵是单变量回归的拟合值，数值范围比原始X大很多（原始X是标准化的~N(0,1)，LOO矩阵范围可达[-20,30]）
   - 基于原始X计算的lambda完全不适用于LOO矩阵，导致惩罚强度严重不足，系数爆炸

**lambda和negative_penalty的区别**：
- **lambda**：全局惩罚强度总开关，控制整体模型的稀疏程度和系数大小，相当于"音量旋钮"
- **negative_penalty**：相对比例调节器，仅调整正负系数惩罚的相对强弱，相当于"均衡器"，不影响整体惩罚强度
- 两者不能互相替代，必须先将lambda调整到合适的范围，再调整negative_penalty优化符号偏好

**当前进展**：
- ✅ 已修复所有返回值、维度、lambda尺度问题
- ✅ 手动测试找到了最优lambda范围：0.01~0.05，软约束最优MSE可达**11.47**，效果正常
- 后续优化方向：
  1. 固定alpha=1.0、beta=1.0，仅调优lambda超参数，简化调参流程
  2. 新增非对称惩罚权重标准化：在计算w_j^+和w_j^-后，将所有w_j^+除以其总和进行标准化，使权重平均值为1，稳定求解器收敛
  3. 修复cv_uni的lambda自动生成逻辑，实现全自动调参
  4. 启用自适应权重和组约束，进一步提升性能

---
### 问题5：自适应权重导致惩罚尺度不稳定
**问题现象**：引入显著性权重w_j后，L1惩罚项的全局尺度发生变化，导致lambda的最优范围随数据分布变化，求解不稳定。

**根因分析**：
显著性权重w_j的数值范围受第一阶段单变量回归的p值分布影响，不同数据集的w_j总和差异很大，导致整体惩罚强度不稳定，lambda的调优范围不固定。

**修复方案**：
对非对称惩罚权重进行标准化处理：
1. 计算所有特征的正惩罚权重w_j^+ = w_j，求和得到S_plus = Σw_j^+
2. 计算所有特征的负惩罚权重w_j^- = α/w_j + β*w_j，求和得到S_minus = Σw_j^-
3. 对正权重标准化：w_j^+ = (w_j^+ / S_plus) * p，其中p为特征数量
4. 对负权重标准化：w_j^- = (w_j^- / S_minus) * p，其中p为特征数量
5. 标准化后两类权重各自的平均值为1，保持相对比例不变，整体惩罚尺度稳定，lambda的最优范围不再随数据分布和特征数量变化

**待探讨的理论问题**：为什么需要乘以特征数量p？
- 直觉解释：当p=1000时，标准化后每个权重的平均值为1/1000，乘以p后平均权重回到1，保持与未使用权重时的普通Lasso惩罚尺度对齐
- 尺度一致性：保证lambda的调优范围与传统Lasso具有可比性，无需随特征数量调整lambda的量级
- 经验验证：乘p后lambda的最优范围与手动验证的0.01~0.05区间一致，求解器收敛更稳定，数值表现更好
- 理论上需要进一步验证：加权L1惩罚Σw_j|θ_j| 乘以p后是否与原始未加权L1惩罚的统计特性保持一致，是否影响正则化的渐近性质

**设计优势**：
- 保持不同特征间的惩罚相对比例不变，不改变非对称惩罚的设计意图
- 全局惩罚尺度统一，lambda的调优范围固定，简化超参数调整
- 求解器收敛更稳定，避免因权重总和过大导致惩罚过强或过弱

**已完成的配套改进**：
1. **p值安全截断**：对第一阶段单变量回归得到的p值做范围限制 `p_j ∈ [1e-4, 0.95]`，避免极端p值导致权重异常
2. **非对称动态lambda_max计算**：基于LOO矩阵和标准化后的权重，自动计算正则化路径的最大值：
   $$c_j = \frac{1}{n} X_{\text{LOO}, j}^T y$$
   - 若 $c_j > 0$：$\lambda_{\text{req}} = \frac{c_j}{\tilde{w}_j^+}$（正系数需要的最小惩罚）
   - 若 $c_j < 0$：$\lambda_{\text{req}} = \frac{-c_j}{\tilde{w}_j^-}$（负系数需要的最小惩罚）
   - $\lambda_{\text{max}} = \max_j \lambda_{\text{req}}$，刚好让所有系数为0，替代硬编码固定范围
3. **lambda路径自适应生成**：基于动态计算的lambda_max和lmda_min_ratio自动生成合适的正则化路径，适配不同数据场景，无需手动调整范围

---
### 问题6：坐标下降软阈值尺度不匹配
**问题现象**：完成权重标准化和动态lambda计算后，软约束XLasso在AR(1)场景下MSE仍然高达3542263.74，选中变量数707个，过拟合极其严重，TPR=1.0但FDR=0.93，惩罚强度严重不足。

**根因分析**：
坐标下降的非对称软阈值逻辑存在尺度不匹配问题：
1. 阈值`tau_pos/tau_neg`是基于全局lambda和权重计算的，单位和X的尺度相关
2. 最小二乘解`beta_j = sum(X[:,j] * residual) / (n * X_diag[j])`已经做了尺度归一化，单位和原始y一致
3. 判断条件中直接用`beta_j > tau_pos[j]`进行比较，相当于阈值被放大了X_diag[j]倍，实际惩罚强度远低于设计值，导致大量噪声变量被保留

**修复方案**：
将阈值先除以`X_diag[j]`做尺度对齐，再进行条件判断：
```python
# 修复前（错误）
if beta_j > tau_pos[j]:
    new_w = beta_j - tau_pos[j] / X_diag[j]
elif beta_j < -tau_neg[j]:
    new_w = beta_j + tau_neg[j] / X_diag[j]

# 修复后（正确）
tau_pos_scaled = tau_pos[j] / X_diag[j]
tau_neg_scaled = tau_neg[j] / X_diag[j]
if beta_j > tau_pos_scaled:
    new_w = beta_j - tau_pos_scaled
elif beta_j < -tau_neg_scaled:
    new_w = beta_j + tau_neg_scaled
else:
    new_w = 0.0
```

**修复效果**：
AR(1)场景下软约束XLasso性能大幅提升，完全解决过拟合问题：
- MSE从3542263.74降到2.46，仅比UniLasso高25.28%
- FDR从0.9293降到0.3750，仅比UniLasso高9.62%
- F1分数从0.1321提升到0.7692，仅比UniLasso低3.08%
- 选中变量数从707降到80，和UniLasso的76非常接近
- 整体性能和UniLasso处于同一水平，验证了修复的正确性

---
## 文档与实现一致性对比
基于XLasso.md算法文档与工程实现的对比如下：

### ✅ 完全一致的部分
1. **两阶段框架**：第一阶段单变量回归生成LOO矩阵，第二阶段系数转换`beta_j = theta_j * beta_j_univariate`完全符合算法定义
2. **非对称惩罚核心公式**：
   - 正阈值 `tau_pos_base = lmda * fw` 对应公式 `w_j^+ = λ * w_j`
   - 负阈值 `tau_neg_base = lmda * (alpha / fw_safe + beta * fw)` 对应公式 `w_j^- = λ * (α/w_j + β*w_j)`
   - 非对称软阈值更新逻辑完全符合设计
3. **显著性权重计算**：实现了`w_j = 1 / (-log(p_j + epsilon))`的公式，支持t_statistic/p_value/correlation三种权重计算方式
4. **坐标下降求解器**：逐特征更新、warm start、残差更新、Numba加速四大特性完全符合文档描述，实测速度比梯度下降提升3~5倍
5. **GLM扩展支持**：支持gaussian/binomial/poisson/multinomial/cox五种GLM家族，非高斯家族自动回退到梯度下降求解

### ⚠️ 部分实现差异
1. **组一致性约束实现**：
   - 文档定义：成对惩罚`P_group(θ) = Σ Σ I(sign(θ_j)≠sign(θ_k)) · |θ_jθ_k|
   - 工程实现：采用简化的阈值调整方式，对组符号相反的系数额外增加惩罚，近似实现符号趋同效果，计算效率更高
2. **正则化参数设计**：
   - 文档定义：非对称惩罚和组惩罚有独立的正则化参数λ₁、λ₂
   - 工程实现：共享同一个正则化参数`lmda`，组惩罚强度通过`group_penalty`相对倍数调整，简化调参
3. **Cox模型支持**：文档提到Cox作为扩展家族，代码中当前使用高斯损失作为占位符，未实现真实Cox比例风险模型

### ❌ 文档有描述但未实现
1. 基于Louvain社区检测的自适应分组策略
2. 多项式回归完整支持（当前仅作为二分类处理）

### 📝 工程实现特有特性（文档未提及）
1. Nesterov动量加速梯度下降求解器
2. PyTorch后端GPU加速支持
3. 第一阶段支持样条、决策树等非线性单变量模型

**整体功能完成度**：85%，核心算法特性全部实现，高级特性待开发

### 输出说明
运行过程中会实时打印每个方法的指标结果，运行结束后会在 `result/exp_001` 目录下生成两个文件：
- `experiment_001_results.csv`：所有重复实验的详细原始数据
- `experiment_001_summary.csv`：按信噪比和方法分组的平均结果汇总表

---

## XLasso 函数调用指南

### 核心函数：fit_uni()
用于拟合XLasso模型，返回整个正则化路径的结果。

#### 函数签名
```python
def fit_uni(
    X: np.ndarray,
    y: np.ndarray,
    family: str = "gaussian",
    adaptive_weighting: bool = True,
    weight_method: str = "p_value",
    gamma: float = 1.0,
    sharp_scale: float = 0.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    lmda: Optional[Union[float, np.ndarray]] = None,
    lmda_min_ratio: float = 1e-3,
    n_lmdas: int = 30,
    warm_start: bool = True,
    max_iter: int = 200,
    tol: float = 1e-4,
    backend: str = "numba",
    # v2.0 新增：成对正交分解参数
    enable_orthogonal_decomp: bool = False,
    orthogonal_corr_threshold: float = 0.7,
    enable_pair_aware_filter: bool = False,
    pair_filter_k: Optional[int] = None,
    # v2.1 新增：组级扩展参数
    enable_group_decomp: bool = False,
    group_corr_threshold: float = 0.7,
    max_group_size: int = 10,
    enable_group_aware_filter: bool = False,
    group_filter_k: Optional[int] = None
) -> UniLassoResult:
```

#### 核心参数说明
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **基础参数** | | | |
| X | np.ndarray | 必填 | 设计矩阵，形状 (n_samples, n_features) |
| y | np.ndarray | 必填 | 响应变量，形状 (n_samples,) |
| family | str | "gaussian" | GLM家族类型：gaussian/binomial/poisson |
| **权重参数** | | | |
| adaptive_weighting | bool | True | 是否启用自适应权重惩罚 |
| weight_method | str | "p_value" | 权重计算方法：p_value/t_statistic/correlation |
| gamma | float | 1.0 | 权重指数，控制权重对比度 |
| sharp_scale | float | 0.0 | p值锐化指数，0表示不锐化，越大权重差异越大 |
| alpha | float | 1.0 | 显著变量负系数惩罚强度 |
| beta | float | 1.0 | 不显著变量惩罚强度 |
| **正则化参数** | | | |
| lmda | float/array | None | 自定义正则化参数，None表示自动生成路径 |
| lmda_min_ratio | float | 1e-3 | 最小lambda与最大lambda的比值 |
| n_lmdas | int | 30 | 正则化路径长度 |
| **求解器参数** | | | |
| backend | str | "numba" | 求解器后端：numba（高速）/pytorch（灵活） |
| max_iter | int | 200 | 最大迭代次数 |
| tol | float | 1e-4 | 收敛阈值 |
| **v2.0 成对优化参数** | | | |
| enable_orthogonal_decomp | bool | False | 开启成对正交分解，处理反符号孪生变量场景 |
| orthogonal_corr_threshold | float | 0.7 | 成对相关系数阈值，超过则进行正交分解 |
| enable_pair_aware_filter | bool | False | 开启成对感知过滤，提升成对识别准确率 |
| pair_filter_k | int | None | 成对过滤保留的变量数，None表示自动 |
| **v2.1 组级优化参数** | | | |
| enable_group_decomp | bool | False | 开启组级正交分解，处理任意大小的相关变量组 |
| group_corr_threshold | float | 0.7 | 组相关系数阈值，超过则归入同一组 |
| max_group_size | int | 10 | 最大组大小，避免超大组 |
| enable_group_aware_filter | bool | False | 开启组感知过滤，提升组内变量识别完整性 |
| group_filter_k | int | None | 组过滤保留的变量数，None表示自动 |

#### 返回值说明
返回 `UniLassoResult` 对象包含以下属性：
- `coefs`: 系数矩阵，形状 (n_lmdas, n_features)
- `lmdas`: 正则化参数数组，形状 (n_lmdas,)
- `intercepts`: 截距数组，形状 (n_lmdas,)
- `n_iter`: 每个lambda路径的迭代次数
- `beta_univariate`: 第一阶段单变量回归系数

---

### 核心函数：cv_uni()
带交叉验证的XLasso拟合，自动选择最优正则化参数。

#### 函数签名
与 `fit_uni()` 完全一致，额外增加交叉验证参数：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| n_folds | int | 5 | 交叉验证折数 |
| scoring | str | "mse" | 评价指标：mse/accuracy/roc_auc |
| n_jobs | int | 1 | 并行作业数，-1表示使用所有CPU |

#### 返回值说明
返回 `CrossValResult` 对象，包含：
- `best_idx`: 最优lambda的索引
- `best_lmda`: 最优lambda值
- `best_coef`: 最优系数
- `best_intercept`: 最优截距
- `cv_scores`: 交叉验证分数数组
- 所有 `fit_uni()` 返回的属性

---

### 使用示例

#### 示例1：基础使用（默认参数）
```python
import numpy as np
from unilasso.uni_lasso import fit_uni, cv_uni

# 生成数据
X = np.random.randn(300, 500)
beta_true = np.zeros(500)
beta_true[:10] = 2.0
y = X @ beta_true + np.random.randn(300) * 0.5

# 基础拟合
fit = fit_uni(X, y)

# 交叉验证拟合（推荐）
cv_fit = cv_uni(X, y, n_folds=5)
print(f"最优lambda: {cv_fit.best_lmda:.4f}")
print(f"最优系数非零数: {np.sum(np.abs(cv_fit.best_coef) > 1e-8)}")
```

#### 示例2：降维打击场景（v2.0 成对优化）
适用于高相关反符号孪生变量场景：
```python
from lab.data_generator import generate_twin_variable_experiment
X, y, beta_true = generate_twin_variable_experiment(n=300, p=500, rho=0.85, snr=2.0)

# 使用成对正交分解
fit = fit_uni(
    X, y,
    family="gaussian",
    adaptive_weighting=True,
    weight_method="p_value",
    gamma=0.5,
    sharp_scale=20,
    enable_orthogonal_decomp=True,
    orthogonal_corr_threshold=0.8,
    enable_pair_aware_filter=True,
    pair_filter_k=20,
    backend="numba"
)

# 取中间lambda的结果
idx = len(fit.lmdas) // 2
beta_pred = fit.coefs[idx, :]
```

#### 示例3：多变量组场景（v2.1 组级优化）
适用于3个及以上高相关反符号变量组场景：
```python
# 生成三变量组数据
n, p = 300, 500
rho = 0.85
X = np.random.randn(n, p)
common = np.random.randn(n)
for i in range(3):
    X[:, i] = common * np.sqrt(rho) + np.random.randn(n) * np.sqrt(1-rho)
true_beta = np.zeros(p)
true_beta[0] = 2.0
true_beta[1] = 2.0
true_beta[2] = -3.0
y = X @ true_beta + np.random.randn(n) * 1.0

# 使用组级优化
fit = fit_uni(
    X, y,
    enable_group_decomp=True,
    group_corr_threshold=0.7,
    max_group_size=10,
    enable_group_aware_filter=True,
    group_filter_k=10,
    backend="numba"
)
```

---

### 常见场景最佳参数配置

| 场景 | 推荐配置 |
|------|----------|
| **普通稀疏回归** | 默认参数即可，无需额外配置 |
| **高相关变量（同符号）** | 开启组约束：`group_penalty=5.0, corr_threshold=0.7` |
| **成对反符号孪生变量（降维打击）** | `enable_orthogonal_decomp=True, orthogonal_corr_threshold=0.8, enable_pair_aware_filter=True, pair_filter_k=20` |
| **多变量反符号组** | `enable_group_decomp=True, group_corr_threshold=0.7, max_group_size=10, enable_group_aware_filter=True` |
| **高噪声场景** | 增大`sharp_scale=20`，放大真实变量与噪声的权重差异 |
| **需要更稀疏的结果** | 增大`beta=2.0~5.0`，提高不显著变量的惩罚强度 |
| **需要保留更多真实变量** | 减小`alpha=0.1~0.5`，降低显著变量负系数的惩罚 |

---

### 版本兼容性说明
- 所有v2.0和v2.1新增参数均为可选，默认关闭，完全向后兼容，原有代码无需任何修改即可运行。
- 成对参数和组级参数可以同时开启，会先进行组级分解，再对组内小于等于2个变量的进行成对分解。
- 当同时开启`enable_orthogonal_decomp`和`enable_group_decomp`时，组级参数优先级更高，成对参数作为组级的向后兼容别名。

---

## 综合实验计划（v2.2版本验证）
### 一、实验目的
全面对比XLasso所有算法形态与基准算法的性能，验证组正交分解替代硬组约束的效果，覆盖模拟场景和真实数据集。

### 二、对比算法列表
#### 1. 基准方法
| 算法名称 | 说明 |
|----------|------|
| UniLasso | 原始UniLasso算法（基准） |
| Lasso | 标准Lasso（sklearn实现） |

#### 2. XLasso系列算法
| 算法名称 | 参数配置 | 说明 |
|----------|----------|------|
| XLasso-Soft | `adaptive_weighting=False, enable_group_decomp=False, enable_group_constraint=False` | 仅非对称软约束，无其他改进 |
| XLasso-Adaptive | `adaptive_weighting=True, enable_group_decomp=False, enable_group_constraint=False` | 软约束+自适应权重 |
| XLasso-GroupDecomp | `adaptive_weighting=True, enable_group_decomp=True, enable_group_aware_filter=False` | 软约束+自适应权重+组正交分解 |
| XLasso-Full | `adaptive_weighting=True, enable_group_decomp=True, enable_group_aware_filter=True` | 完整XLasso：软约束+自适应权重+组正交分解+组感知过滤 |
| (弃用) XLasso-GroupConstraint | `adaptive_weighting=True, enable_group_constraint=True` | 旧版硬组约束（仅作为对比） |

#### 3. 对标Lasso算法
| 算法名称 | 说明 |
|----------|------|
| Adaptive Lasso | 自适应Lasso |
| Group Lasso | 组Lasso（自动相关性分组） |
| Fused Lasso | 融合Lasso |
| Adaptive Sparse Group Lasso | 自适应稀疏组Lasso（同时支持组级和特征级稀疏） |

### 三、实验设计
#### 1. 模拟实验（4个经典场景，来自XLasso.md）
| 实验编号 | 实验名称 | 场景特点 | 参数配置 |
|----------|----------|----------|----------|
| 实验1 | 高维成对相关稀疏回归 | 所有变量两两相关0.5，前100个变量非零 | n=300, p=1000, σ=[0.5, 1.0, 2.5] |
| 实验2 | AR(1)相关稀疏回归 | 相邻变量相关0.8，奇数前50个变量非零 | n=300, p=1000, σ=[0.5, 1.0, 2.5] |
| 实验3 | 二分类偏移变量选择 | AR(1)相关，前20个变量在正类偏移0.5 | n=200, p=500 |
| 实验4 | 反符号孪生变量（降维打击） | 10对高相关(ρ=0.85)反符号变量 | n=300, p=1000, σ=1.0 |

#### 2. 真实数据集实验（4个公开数据集）
| 数据集 | 任务类型 | 样本量 | 特征数 | 特点 |
|--------|----------|--------|--------|------|
| Breast Cancer Wisconsin | 二分类 | 569 | 30 | 小维度医疗数据 |
| Arcene | 二分类 | 200 | 10000 | 高维质谱数据，癌症检测 |
| Gisette | 二分类 | 7000 | 5000 | 手写数字识别，区分0和4 |
| Dorothea | 二分类 | 1150 | 100000 | 超高维稀疏数据，药物发现 |

### 四、评价指标
#### 回归任务（实验1、实验2）
| 指标 | 说明 |
|------|------|
| MSE | 测试集均方误差（越小越好） |
| TPR | 真阳性率（选对的真实变量比例，越大越好） |
| FDR | 假发现率（选错的变量占选中变量的比例，越小越好） |
| F1 | F1分数（变量选择综合性能，越大越好） |
| 选中变量数 | 模型选择的非零变量数 |
| 运行时间 | 单轮拟合耗时 |

#### 分类任务（实验3、所有真实数据集）
| 指标 | 说明 |
|------|------|
| AUC | 测试集ROC曲线下面积（越大越好） |
| 准确率 | 分类准确率（越大越好） |
| TPR | 真阳性率（选对的真实变量比例，模拟实验可用） |
| FDR | 假发现率（模拟实验可用） |
| F1 | F1分数（变量选择综合性能） |
| 选中变量数 | 模型选择的非零变量数 |

### 五、实验配置
1. **重复次数**：每个实验重复运行10次，统计均值和标准差（避免随机波动）
2. **超参数选择**：每个算法通过5折交叉验证选择最优超参数
   - 正则化参数λ搜索范围：100个值的对数空间，从λ_max到λ_max*1e-4
   - XLasso特有参数：固定alpha=1.0, beta=1.0，仅通过交叉验证选择最优gamma参数
   - 对标Lasso算法（Adaptive Lasso/Group Lasso/Fused Lasso）通过交叉验证选择各自最优超参数
3. **统一设置**：所有算法均使用标准化特征，拟合截距
4. **硬件环境**：统一使用CPU运行，相同机器配置保证公平对比
5. **求解器**：XLasso默认使用Numba加速坐标下降求解器，其他算法使用官方默认求解器

### 六、结果呈现模板
#### 模拟实验结果示例
| 实验场景 | 算法 | MSE（均值±标准差） | TPR | FDR | F1 | 选中变量数 |
|----------|------|---------------------|-----|-----|----|------------|
| 实验2 σ=0.5 | UniLasso | 144.12±5.23 | 0.20±0.03 | 0.23±0.05 | 0.32±0.04 | 13±2 |
| 实验2 σ=0.5 | XLasso-Full | **4.13±0.32** | **1.00±0.00** | 0.38±0.04 | **0.77±0.02** | 80±5 |

#### 真实数据集结果示例
| 数据集 | 算法 | AUC | 准确率 | 选中变量数 | 运行时间 |
|--------|------|-----|--------|------------|----------|
| Arcene | UniLasso | 0.78±0.03 | 0.75±0.04 | 15±3 | 1.2s |
| Arcene | XLasso-Full | **0.89±0.02** | **0.86±0.03** | 25±4 | 2.5s |

### 七、运行说明
#### 实验启动命令
```bash
cd /home/lili/lyn/clear/XLasso/lab
# 运行所有模拟实验
python run_all_simulations.py --n-repeats 10
# 运行所有真实数据集实验
python run_all_real_datasets.py --n-repeats 10
```

#### 结果保存路径
- 模拟实验结果：`result/simulations/`
- 真实数据集结果：`result/real_datasets/`
- 汇总报告：`result/final_comparison_report.md`

### 八、重点对比维度
1. **组正交分解 vs 硬组约束**：重点对比反符号孪生变量场景的性能，验证组正交分解的优势
2. **各模块消融实验**：验证软约束、自适应权重、组正交分解、组感知过滤每个模块的独立贡献
3. **真实场景适用性**：对比各算法在不同维度、不同稀疏度真实数据集上的泛化性能
4. **效率对比**：对比各算法的运行时间和内存占用，评估可扩展性

---

## 工具集使用说明

### 目录结构
```
lab/
├── release/                    # 版本发布相关工具
│   ├── visualize_experiments.py    # 结果可视化脚本
│   └── release_version.py          # 版本发布脚本
└── hyperparameter_tuning/      # 超参数调优相关工具
    └── hyperparameter_tuning.py    # 调优实验模板
```

### 使用方法
1. **可视化实验结果**：
   ```bash
   python release/visualize_experiments.py --result-dir <实验结果目录>
   ```
   自动生成所有算法的指标对比图、折线图、散点图和汇总表格。

2. **发布版本**：
   ```bash
   python release/release_version.py --version v2.3.1 --result-dir <实验结果目录> --message "版本说明"
   ```
   自动完成：复制实验结果、生成版本说明、打Git标签、生成版本压缩包。

3. **超参数调优**：
   ```bash
   python hyperparameter_tuning/hyperparameter_tuning.py --algorithm XLasso-Soft --experiment exp2
   ```
   支持对XLasso全系算法的关键超参数进行网格搜索调优。

### 可选操作
给脚本添加执行权限：
```bash
chmod +x release/*.py hyperparameter_tuning/*.py
```
