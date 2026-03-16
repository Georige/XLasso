# UniLasso: 单变量引导稀疏回归

UniLasso 是一个 Python 包，实现了新颖且可解释的**单变量引导稀疏回归**方法 (https://hdsr.mitpress.mit.edu/pub/3i97j340/release/4)。

本仓库在原始方法基础上扩展了两个关键创新：

1. **分组非负符号软约束** - 对高度相关特征自动分组，鼓励组内特征保持符号一致性，促进组内变量同时选择/剔除
2. **单变量显著性自适应惩罚权重** - 根据单变量回归显著性调整每个特征的Lasso惩罚：显著特征惩罚降低，不显著特征惩罚提高

扩展后的完整算法称为 **XLasso**。

## 特点

- 支持所有广义线性模型：高斯回归、二项逻辑回归、泊松回归、多项分类、Cox比例风险模型
- 支持交叉验证选择最优正则化参数
- 提供拟合和预测功能
- 支持非线性效应：使用B样条或决策树作为单变量模型
- 包含完整的模拟实验框架，系统评估两项创新的效果

## 安装

你可以通过以下命令安装 UniLasso：

```bash
git clone https://github.com/sophial05/uni-lasso.git
cd uni-lasso
pip install -e .
```

## 算法步骤 (`fit_uni`)

`fit_uni` 算法分三步执行：

### 第一步：单变量预估计
对每个特征**单独**拟合广义线性模型：
$$
y \sim \text{GLM}( \beta_0 + \beta_j x_j, \text{family} )
$$
得到每个特征的系数估计 $\hat{\beta}_j$ 及其显著性统计量 $t_j$。

支持非线性单变量模型：
- `linear`: 线性模型（默认）
- `spline`: B样条展开，可以拟合非线性关系 $y \sim f(x_j)$
- `tree`: 决策树回归

### 第二步：自适应惩罚权重构造
根据单变量显著性 $t_j$ 调整每个特征的Lasso惩罚强度：
$$
\lambda_j = \frac{\lambda_0}{|t_j|}
$$
- 显著性越高 $\Rightarrow$ $|t_j|$ 越大 $\Rightarrow$ $\lambda_j$ 越小 $\Rightarrow$ 惩罚越轻
- 显著性越低 $\Rightarrow$ $|t_j|$ 越小 $\Rightarrow$ $\lambda_j$ 越大 $\Rightarrow$ 惩罚越重
- 这自动给重要特征更小惩罚，提高变量选择准确性

### 第三步：分组符号一致性软约束
根据特征间相关系数矩阵自动贪婪分组：
- 任意两个特征，如果 $|\text{cor}(x_j, x_k)| >$ `corr_threshold`，分到同一组
- 对每组添加软约束惩罚：
$$
\text{penalty} = \text{group_penalty} \cdot \sum_{j<k \in g} I(\text{sign}(\beta_j) \neq \text{sign}(\beta_k))
$$
- 当组内高度相关特征符号不一致时增加惩罚，鼓励它们同时入选且保持符号一致

### 最终优化问题
$$
\min_{\beta} \left\{ -\ell(\beta) + \sum_{j=1}^p \lambda_j |\beta_j| + \sum_{g \in \text{groups}} \text{group_penalty} \cdot \sum_{j<k \in g} I(\text{sign}(\beta_j) \neq \text{sign}(\beta_k)) \right\}
$$
其中 $\ell(\beta)$ 是对数似然。

## 快速开始：使用 `fit_uni` / `cv_uni`

```python
import numpy as np
from unilasso.uni_lasso import fit_uni, cv_uni

# 生成示例数据
n, p = 200, 50
X = np.random.randn(n, p)
beta_true = np.zeros(p)
beta_true[:10] = np.random.uniform(0.5, 2.0, 10)
y = X @ beta_true + np.random.randn(n)

# 拟合 XLasso，同时开启两项创新
result = cv_uni(
    X, y,
    family='gaussian',
    adaptive_weighting=True,        # 开启自适应惩罚权重
    enable_group_constraint=True,    # 开启分组符号约束
    corr_threshold=0.7,            # 相关性分组阈值
    group_penalty=5.0              # 分组惩罚强度
)

# 获取交叉验证选出的最佳系数
best_coefs = result.coefs[result.best_idx]
print(best_coefs)

# 在测试集上预测
X_test = np.random.randn(100, p)
y_pred = X_test @ best_coefs + result.intercept[result.best_idx]
```

### 关键参数说明

| 参数 | 说明 | 默认值 |
|-----------|-------------|---------|
| `adaptive_weighting` | 是否开启基于单变量显著性的自适应惩罚加权 | `False` |
| `enable_group_constraint` | 是否开启对相关特征分组符号软约束 | `False` |
| `corr_threshold` | 分组相关性阈值，大于该阈值的特征分到一组 | `0.7` |
| `group_penalty` | 分组符号不一致惩罚强度 | `5.0` |
| `univariate_model` | 单变量模型类型：`linear` 线性, `spline` B样条, `tree` 决策树 | `linear` |

## 运行模拟实验

### 命令行方式

我们提供三个命令行脚本运行不同类别的实验：

```bash
# 运行所有线性高斯实验（每个场景30次重复）
python scripts/run_linear_experiment.py

# 使用较少重复次数快速测试
python scripts/run_linear_experiment.py --n-repeats 5

# 运行所有 GLM 实验（每个分布20次重复）
python scripts/run_glm_experiment.py --n-repeats 5

# 运行所有非线性实验（每个场景20次重复）
python scripts/run_nonlinear_experiment.py --n-repeats 5
```

实验结果（CSV文件和图像）会保存到 `experiments/results/` 目录。

### 交互式 Jupyter 笔记本

如果你想交互式探索并在内联查看所有结果，请打开综合对比笔记本：

```bash
jupyter notebook notebooks/all_experiments_comparison.ipynb
```

这个笔记本依次运行三类实验，直接在笔记本中显示汇总表格和热力图，方便实验探索阶段查看结果。

## 实验设计

我们的实验通过四种配置系统分离出每项创新的贡献：

| 配置 | 分组约束 | 自适应加权 | 说明 |
|---------------|------------------|------------------|-------------|
| 基线 | ❌ | ❌ | 无创新，类似标准Lasso |
| 仅自适应 | ❌ | ✅ | 单独看自适应加权效果 |
| 仅分组 | ✅ | ❌ | 单独看分组约束效果 |
| 完整版 XLasso | ✅ | ✅ | 同时开启两项创新 |

我们在以下场景评估：
- **线性高斯**：7种不同相关性结构场景
- **GLM**：覆盖所有 5 种 GLM 分布
- **非线性**：4种场景，比较 linear / spline / tree 三种单变量模型

## 示例

关于包使用的完整示例，请参见 `examples/` 目录。

## 许可证

本项目使用 MIT 许可证 - 详见 LICENSE 文件。

## 引用

如果你在研究中使用了 UniLasso，请引用原始论文：

```bibtex
@article{chatterjee2025univariate,
  title={Univariate-Guided Sparse Regression},
  author={Chatterjee, Sourav and Hastie, Trevor and Tibshirani, Robert},
  journal={arXiv preprint arXiv:2501.18360},
  year={2025}
}
```

## 联系

如有问题或反馈，请联系 Sophia Lu at sophialu@stanford.edu
