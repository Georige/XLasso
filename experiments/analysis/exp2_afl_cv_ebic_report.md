# AFL 参数选择方案对比报告：Fixed vs CV-1SE vs EBIC

## 1. 实验概述

**目的**：对比三种 AFL 参数选择方案在高维稀疏回归中的表现

**数据配置**：
- 相关性：AR(1), rho=0.8
- n_samples: 300
- n_features: 500
- n_nonzero: 20 (真实稀疏度)
- SNR：sigma=0.1 (SNR=10), sigma=0.5 (SNR=2)
- 重复次数：5 次/配置

**三种方案**：
1. **AFL-Fixed**：使用 Stage1 最优固定参数 (lambda_ridge=1.0, gamma=0.5)
2. **AFL-CV-1SE**：使用 1-SE 法则选择 CV 最优参数
3. **AFL-EBIC**：使用 Extended BIC 准则选择参数

---

## 2. EBIC 公式

EBIC (Extended BIC) 定义为：

$$EBIC = n \cdot \ln\left(\frac{RSS}{n}\right) + |S| \cdot \ln(n) + 2\gamma \cdot \ln\binom{p}{|S|}$$

其中：
- 第一项：拟合误差
- 第二项：模型复杂度（特征数量）
- 第三项：高维惩罚项（防止特征组合爆炸）

**特点**：EBIC 极其厌恶假阳性，会在拟合误差和稀疏性之间取得平衡。

---

## 3. 实验结果

### 3.1 sigma=0.1 (SNR=10.0)

| 模型 | n_selected | F1 | FDR | TPR | best_gamma | best_lambda_ridge |
|------|------------|-----|-----|-----|------------|-------------------|
| **AFL-Fixed** | 20.4 | 0.990 | 0.019 | 1.000 | 0.5 (fixed) | 1.0 (fixed) |
| AFL-CV-1SE | 20.4 | 0.991 | 0.017 | 1.000 | 0.70 | 0.96 |
| AFL-EBIC | 21.6 | 0.961 | 0.073 | 1.000 | 1.00 | 2.32 |

### 3.2 sigma=0.5 (SNR=2.0)

| 模型 | n_selected | F1 | FDR | TPR | best_gamma | best_lambda_ridge |
|------|------------|-----|-----|-----|------------|-------------------|
| **AFL-Fixed** | 20.6 | 0.985 | 0.029 | 1.000 | 0.5 (fixed) | 1.0 (fixed) |
| AFL-CV-1SE | 22.4 | 0.946 | 0.101 | 1.000 | 0.68 | 1.24 |
| AFL-EBIC | 22.0 | 0.954 | 0.086 | 1.000 | 1.00 | 2.06 |

### 3.3 全 SNR 平均汇总

| 模型 | n_selected | F1 | FDR | TPR |
|------|------------|-----|-----|-----|
| **AFL-Fixed** | **20.5** | **0.988** | **0.024** | 1.000 |
| AFL-CV-1SE | 21.4 | 0.969 | 0.059 | 1.000 |
| AFL-EBIC | 21.8 | 0.958 | 0.079 | 1.000 |

---

## 4. 参数选择分析

### 4.1 EBIC 参数选择特征

**best_gamma 分布** (10次实验):
```
gamma = 1.0: 10次 (100%)
```
所有 EBIC 实验都一致选择了 gamma=1.0。

**best_lambda_ridge 分布**:
```
0.18:  1次
0.84:  1次
1.04:  1次
1.12:  1次
1.30:  1次
2.80:  2次
3.24:  1次
4.00:  1次
4.62:  1次
```
lambda_ridge 分散度大，从 0.18 到 4.62，说明 EBIC 对 lambda_ridge 不敏感。

### 4.2 为什么 gamma=1.0?

EBIC 选择 gamma=1.0 而非 Fixed 的 0.5，可能原因：

1. **gamma=1.0 时权重计算**：`weights = 1 / (|beta_ridge| + eps)`
   - 比 gamma=0.5 更激进地惩罚小系数
   - 噪声特征权重更大，难以被 Lasso 选中

2. **但实际效果**：gamma=1.0 导致 n_selected 略高 (21.8 vs 20.5)
   - 可能 gamma=0.5 对这个特定数据生成模型更合适

### 4.3 lambda_ridge 分散

EBIC 对 lambda_ridge 的选择分散在很大范围内，说明：
- 不同噪声水平下，最优 lambda_ridge 不同
- EBIC 能够自适应找到合适的值

---

## 5. 三种方案对比

### 5.1 性能排名

| 排名 | 模型 | n_selected | F1 | FDR |
|------|------|------------|-----|-----|
| 1 | **AFL-Fixed** | 20.5 | 0.988 | 0.024 |
| 2 | AFL-CV-1SE | 21.4 | 0.969 | 0.059 |
| 3 | AFL-EBIC | 21.8 | 0.958 | 0.079 |

### 5.2 各方案优缺点

| 方案 | 优点 | 缺点 |
|------|------|------|
| **Fixed** | 最佳性能，可利用先验知识 | 需要预实验确定最优参数 |
| **CV-1SE** | 无需先验，稀疏性好 | 性能略低于 Fixed |
| **EBIC** | 无需验证集，自适应参数 | 性能介于两者之间 |

### 5.3 与 Fixed 的差距

| 方案 | n_selected 差距 | F1 差距 | FDR 差距 |
|------|-----------------|---------|---------|
| CV-1SE | +0.9 (+4%) | -0.019 (-2%) | +0.035 |
| EBIC | +1.3 (+6%) | -0.030 (-3%) | +0.055 |

---

## 6. 结论

1. **Fixed 仍然是最佳**：利用 Stage1 先验知识可获得最优性能

2. **CV-1SE 其次**：
   - 无需先验知识
   - 1-SE 法则有效减少过度选择
   - 性能接近 Fixed

3. **EBIC 第三**：
   - 完全无需验证集
   - 参数选择一致性好（gamma 始终为 1.0）
   - 但性能略低于 CV-1SE

4. **实际建议**：
   - 如果有先验知识：使用 Fixed
   - 如果无先验知识：
     - CV-1SE 适合需要稀疏性的场景
     - EBIC 适合需要参数自适应的场景

---

## 7. 代码修改

### 7.1 EBIC 实现

新增 `AdaptiveFlippedLassoEBIC` 类（位于 `base.py`）：

```python
class AdaptiveFlippedLassoEBIC(BaseAdaptiveFlippedLasso, RegressorMixin):
    """
    使用 EBIC 准则的 AFL
    EBIC = n*ln(RSS/n) + |S|*ln(n) + 2*gamma*ln(C(p,|S|))
    """
    def _compute_ebic(self, y_true, y_pred, n_selected, p):
        # EBIC 计算
        ...

    def fit(self, X, y, ...):
        # 对每个 (gamma, lambda_ridge, alpha) 计算 EBIC
        # 选择 EBIC 最小的参数组合
        ...
```

### 7.2 注册到模块

- `api.py`: 添加 `AdaptiveFlippedLassoClassifierEBIC`
- `__init__.py`: 导出 `AdaptiveFlippedLassoEBIC`, `AdaptiveFlippedLassoClassifierEBIC`
- `run.py`, `sweep.py`: 添加 EBIC 到 ALGO_REGISTRY 和参数处理

---

*报告生成时间: 2026-03-27*
