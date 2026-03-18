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

### 编程习惯约定
> **所有实验脚本均采用「配置+命令行参数」设计模式**：
> - 所有实验参数、超参数均可通过命令行传入，无需修改代码
> - 支持灵活调整实验设置，方便批量实验、参数调优和结果复现
> - 保持配置集中管理，避免硬编码参数散落在代码各处
> - 运行前自动打印完整配置信息，保证实验可追溯

### 输出说明
运行过程中会实时打印每个方法的指标结果，运行结束后会在 `result/exp_001` 目录下生成两个文件：
- `experiment_001_results.csv`：所有重复实验的详细原始数据
- `experiment_001_summary.csv`：按信噪比和方法分组的平均结果汇总表
