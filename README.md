# CG-Lasso：针对高维共线性数据的条件引导正则化回归

> 专注变量选择能力评估：CG-Lasso (Conditional Guided Lasso) 及全套 Lasso 变体算法基准测试

---

## 安装

```bash
pip install -r requirements.txt
```

依赖列表：
- numpy, pandas, scipy, scikit-learn
- matplotlib, pyyaml, joblib, numba
- colorama
- unilasso

**CG-Lasso** (Conditional Guided Lasso with CV)
```
gamma = 1.0, weight_cap = 10.0
```

一种基于**符号先验驱动**的高维稀疏回归方法，通过 Ridge 回归提取纯净符号信息，结合非负 Lasso 实现精准特征选择。

### 算法流程

```
Step 1: 全局数据标准化
Step 2: RidgeCV 提取纯净符号先验 (signs = sign(beta_ridge))
Step 3: 符号翻转映射至第一象限 (X_flipped = X * signs)
Step 4: 非负 Lasso CV 拟合 (positive=True)
Step 5: 逆变换还原 (beta = theta * signs)
```

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `gamma` | 1.0 | 权重指数，控制惩罚强度衰减速度 |
| `weight_cap` | 10.0 | 权重上限，防止极端权重 |
| `lambda_ridge_list` | [0.1, 1.0, 10.0, 100.0] | Ridge 正则化强度候选 |
| `cv` | 5 | 交叉验证折数 |

---

## 实验框架

本仓库提供完整的**基准测试实验框架**，用于评估各算法在多种场景下的表现。

### 支持的实验场景

| 实验 | 场景描述 | 核心挑战 |
|------|----------|----------|
| Exp1 | 高维成对相关稀疏回归 (n=300, p=500) | 高相关变量组 |
| Exp2 | AR(1) 相关稀疏回归 | 自相关结构 |
| Exp3 | 二分类偏移变量选择 | 类别不平衡 |
| Exp4 | 反符号孪生变量 | 符号相反的相关变量 |
| Exp6 | 鸠占鹊巢陷阱 | 噪声变量伪装成信号 |

### 支持的算法

- `cg_lasso` — Conditional Guided Lasso (本框架主打算法)
- `lasso` — 标准 Lasso
- `relaxed_lasso` — Relaxed Lasso (1-SE rule)
- `adaptive_lasso` — Adaptive Lasso
- `unilasso` — UniLasso

---

## 实验复现

### 运行全部实验

```bash
cd experiments
python factory/sweep.py benchmark --config configs/exp1_afl_cv_sigma.yaml
python factory/sweep.py benchmark --config configs/exp2_afl_cv_sigma.yaml
python factory/sweep.py benchmark --config configs/exp3_afl_cv_sigma.yaml
python factory/sweep.py benchmark --config configs/exp4_afl_cv_sigma.yaml
python factory/sweep.py benchmark --config configs/exp6_afl_cv_sigma.yaml
python factory/realdata.py --config configs/realdata/benchmark_realdata_10fold.yaml
python factory/bafl_ablation.py --config configs/bafl_ablation_example.yaml
```

各实验场景：
| 实验 | 场景 | 数据规模 |
|------|------|----------|
| Exp1 | 高维成对相关稀疏回归 | n=300, p=500 |
| Exp2 | AR(1) 相关稀疏回归 | n=300, p=500 |
| Exp3 | 二分类偏移变量选择 | n=300, p=500 |
| Exp4 | 反符号孪生变量 | n=300, p=1000 |
| Exp6 | 鸠占鹊巢陷阱 | n=300, p=500 |
| 真实数据 | Riboflavin 基因表达 | n=71, p=4088 |
| BAFL Ablation | CG-Lasso 参数消融 (gamma × cap) | AR(1), rho=0.9 |

每个 benchmark 实验包含算法对比：**CG-Lasso** (gamma=1.0, cap=10.0)、AdaptiveLassoCV、LassoCV、UniLassoCV、RelaxedLassoCV、ElasticNetCV

### Python API

```python
from experiments.modules import PFLRegressorCV

# 主打算法：gamma=1, cap=10
model = PFLRegressorCV(
    cv=5,
    lambda_ridge_list=[0.1, 1.0, 10.0, 100.0],
    gamma=1.0,
    weight_cap=10.0,
    random_state=2026
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
selected = model.coef_
```

---

## 项目结构

```
XLasso/
├── experiments/
│   ├── configs/           # 实验配置文件
│   │   ├── exp[1-6]_afl_cv_sigma.yaml
│   │   └── realdata/
│   ├── factory/          # 实验运行器
│   │   ├── run.py        # 单实验运行
│   │   ├── realdata.py   # 真实数据实验
│   │   └── sweep.py       # 参数扫描
│   ├── modules/           # 算法实现
│   │   ├── NLasso/       # NLasso & AFL 实现
│   │   │   └── adaptive_flipped_lasso/
│   │   │       └── pf_lasso.py    # CG-Lasso 主算法
│   │   ├── other_lasso/  # 其他 Lasso 变体
│   │   └── ...
│   └── results/          # 实验结果
└── README.md
```

---

## 真实数据实验结果

**数据集**: Riboflavin (n=71, p=4088)

| 模型 | Test MSE (mean±std) | 模型稀疏度 |
|------|---------------------|-----------|
| **PFLRegressorCV** (gamma=1, cap=10) | 0.3471 ± 0.2152 | 16.9 ± 6.7 |
| RelaxedLasso (1-SE) | 0.2819 ± 0.1375 | 20.9 ± 7.3 |
| LassoCV | 0.2912 ± 0.1529 | 42.2 ± 12.7 |
| AdaptiveLassoCV | 0.4030 ± 0.2459 | 45.1 ± 23.2 |
| UniLassoCV | 52.0017 ± 1.9677 | 15.0 ± 5.7 |

**CG-Lasso 在保持优秀预测性能的同时，实现了最稀疏的模型（仅 16.9 个特征）。**

---

## 引用

如果你在研究中使用了本框架，请引用：

```
@article{xlasso2026,
  title={CG-Lasso：针对高维共线性数据的条件引导正则化回归},
  author={},
  year={2026}
}
```
