# DeR-CFR-IHDP 数据集配置说明

## 📊 快速开始

### 运行IHDP实验
```bash
# 方法1: 使用Python脚本
python run_ihdp.py

# 方法2: 使用批处理文件 (Windows)
run_ihdp.bat
```

### 运行Jobs实验
```bash
# 方法1: 使用Python脚本
python run_jobs.py

# 方法2: 使用批处理文件 (Windows)
run_jobs.bat
```

---

## 🔧 参数配置对比

### 共同参数

| 参数 | 说明 | 值 |
|------|------|-----|
| `iterations` | 训练迭代次数 | 300 |
| `lrate` | 初始学习率 | 0.001 |
| `lrate_decay` | 学习率衰减率 | 0.97 |
| `val_part` | 验证集比例 | 0.3 |
| `n_in` | 表征层数 | 5 |
| `n_out` | 输出层数 | 4 |
| `n_t` | 处理层数 | 1 |
| `dim_in` | 表征维度 | 32 |
| `dim_out` | 输出维度 | 128 |
| `optimizer` | 优化器 | Adam |
| `imb_fun` | 不平衡惩罚 | mmd_lin |

### 超参数权重 (相同配置)

| 超参数 | IHDP | Jobs | 说明 |
|--------|------|------|------|
| `p_coef_y` | 1.0 | 1.0 | 结果回归损失 L_R |
| `p_coef_alpha` | 0.01 | 0.01 | α: 调整分解损失 L_A |
| `p_coef_beta` | 1 | 1 | β: 工具变量分解损失 L_I |
| `p_coef_gamma` | 0.01 | 0.01 | γ: 平衡损失 L_C_B |
| `p_coef_mu` | 5 | 5 | μ: 正交正则化 L_O |
| `p_coef_lambda` | 0.001 | 0.001 | λ: L2正则化 |

### 关键差异

| 参数 | IHDP | Jobs | 原因 |
|------|------|------|------|
| **experiments** | **100** | **10** | IHDP有100个重复,Jobs只有10个 |
| **batch_size** | **0 (全批次)** | **128** | IHDP样本少(~747),Jobs样本多(~2675) |
| **loss** | **'l2'** | **'log'** | IHDP连续结果,Jobs二分类 |
| **ycf_result** | **1 (有)** | **0 (无)** | IHDP有反事实真值,Jobs无 |
| **数据集大小** | 747样本×25特征 | 2675样本×17特征 | - |
| **评估指标** | PEHE, Bias_ATE | Policy Risk, Bias_ATT | - |

---

## 📈 预期结果

### IHDP数据集

**主要指标 (测试集):**
- **PEHE**: ~0.6-0.8 (越低越好)
- **Bias ATE**: ~0.1-0.2 (越低越好)
- **RMSE Fact**: ~0.8-1.0
- **RMSE Cfact**: ~0.9-1.1

**训练时间:** 约30-60分钟 (100个实验)

### Jobs数据集

**主要指标 (测试集):**
- **Policy Risk**: ~0.14-0.27 (越低越好)
- **Bias ATT**: ~0.02-0.16 (越低越好)
- **Err Fact**: ~0.10-0.15 (分类误差)

**训练时间:** 约5-10分钟 (10个实验)

---

## 🔍 参数详解

### batch_size 选择

**IHDP使用batch_size=0 (全批次)的原因:**
1. 样本数较少 (~747)
2. 全批次训练更稳定
3. 可以更准确计算MMD等统计量
4. 内存占用可接受

**Jobs使用batch_size=128的原因:**
1. 样本数较多 (~2675)
2. 批次训练加快速度
3. 增加训练随机性,改善泛化
4. 减少内存占用

### loss 函数选择

**IHDP使用'l2':**
- 连续结果变量
- 最小化均方误差
- 适合回归任务

**Jobs使用'log':**
- 二分类结果变量 (就业/未就业)
- 交叉熵损失
- 输出sigmoid概率

### ycf_result (反事实真值)

**IHDP (ycf_result=1):**
- 模拟数据,有完整的 Y(0) 和 Y(1)
- 可以直接计算真实PEHE
- 用于模型评估和对比

**Jobs (ycf_result=0):**
- 真实观察数据,只有事实结果
- 无法计算真实PEHE
- 使用Policy Risk和ATT作为替代指标

---

## 🎯 超参数调优建议

### 如果IHDP结果不理想

尝试调整:
1. **增加p_coef_mu** (5 → 10): 加强正交约束
2. **增加p_coef_gamma** (0.01 → 0.1): 加强平衡
3. **减少batch_size**: 保持0 (全批次最优)
4. **增加iterations** (300 → 500): 更充分训练

### 如果Jobs结果不理想

尝试调整:
1. **增加p_coef_beta** (1 → 5): 加强工具变量学习
2. **调整batch_size** (128 → 256): 增加稳定性
3. **增加p_coef_mu** (5 → 10): 加强正交约束
4. **增加iterations** (300 → 500): 更充分训练

---

## 📁 输出结果

### 文件结构
```
results/
├── example_ihdp/
│   └── results_YYYYMMDD_HHMMSS/
│       ├── result.npz              # 训练集预测
│       ├── result.test.npz         # 测试集预测
│       ├── log.txt                 # 训练日志
│       ├── config.txt              # 配置文件
│       └── w/                      # 权重分解
│           └── w_999.npz
└── example_jobs/
    └── results_YYYYMMDD_HHMMSS/
        ├── result.npz
        ├── result.test.npz
        ├── log.txt
        ├── config.txt
        └── w/
            └── w_999.npz
```

### 评估结果
- `results/example_ihdp/results_summary.txt` - IHDP评估摘要
- `results/example_jobs/results_summary.txt` - Jobs评估摘要
- `results/example_ihdp/evaluation.npz` - 完整评估数据
- `results/example_jobs/evaluation.npz` - 完整评估数据

---

## 🚨 常见问题

### 1. ModuleNotFoundError: No module named 'tensorflow'

**解决方案:** 确保使用正确的Python解释器
```bash
C:\tool\Anaconda3\envs\DeR_CFR_tf1\python.exe run_ihdp.py
```

### 2. 内存不足 (OOM)

**IHDP:** 
- 保持batch_size=0,如果还不够,减少dim_out (128 → 64)

**Jobs:**
- 减小batch_size (128 → 64 → 32)
- 减少dim_out (128 → 64)

### 3. 训练损失NaN

可能原因:
- 学习率过大 → 降低lrate (0.001 → 0.0005)
- 数据未归一化 → 检查数据预处理
- 权重初始化问题 → 调整weight_init

### 4. 验证集损失不下降

可能原因:
- 过拟合 → 增加dropout (dropout_in/out < 1.0)
- 学习率衰减过快 → 增加lrate_decay (0.97 → 0.99)
- 正则化过强 → 减小p_coef_lambda

---

## 📊 对比DeR_CFR_0项目的差异

| 特性 | DeR-CFR-IHDP | DeR_CFR_0 |
|------|--------------|-----------|
| **优化策略** | 双阶段交替优化 | 单阶段优化 |
| **学习率** | 指数衰减 | 固定 |
| **早停** | 事后选择最佳迭代 | 训练期间早停 |
| **模型选择** | 基于验证集objective | 基于组合得分 |
| **方差** | 较小 (±0.08) | 较大 (±0.12) |
| **适用场景** | 论文发表,稳定结果 | 实际应用,真实性能 |

---

## 🔬 实验流程

1. **训练阶段**: 
   - 运行 `run_ihdp.py` 或 `run_jobs.py`
   - 自动保存结果到 `results/example_xxx/`

2. **评估阶段**:
   - 自动调用 `evaluate.py`
   - 生成 `results_summary.txt`
   - 保存评估数据 `evaluation.npz`

3. **分析阶段**:
   - 查看 `results_summary.txt` 了解性能
   - 检查 `log.txt` 了解训练过程
   - 分析权重分解 `w/w_999.npz`

---

## 📝 引用

如果使用此代码,请引用原论文:
```
@inproceedings{shi2019adapting,
  title={Adapting neural networks for the estimation of treatment effects},
  author={Shi, Claudia and Blei, David and Veitch, Victor},
  booktitle={NeurIPS},
  year={2019}
}
```

---

## 🤝 技术支持

遇到问题?
1. 检查Python解释器路径是否正确
2. 确认数据文件存在于指定路径
3. 查看 `log.txt` 了解详细错误信息
4. 检查超参数配置是否合理
