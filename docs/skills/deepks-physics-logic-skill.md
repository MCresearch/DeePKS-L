# DeePKS-kit 物理逻辑与代码实现对应关系

## 概述

DeePKS-kit 的核心功能是通过**迭代训练**获得一个神经网络模型，该模型用于修正密度泛函理论（DFT）中的泛函，以获得更高精度的量子化学计算结果。

**核心思想**：通过神经网络学习 DFT 与高精度方法（如 CCSD(T)）之间的能量差异，从而以 DFT 的计算成本获得接近高精度方法的结果。

---

## 一、物理背景与核心概念

### 1.1 密度泛函理论修正

**物理问题**：
- DFT 计算速度快但精度有限
- 高精度方法（如 CCSD(T)）精度高但计算成本极高
- 目标：用神经网络学习两者之间的能量修正

**修正方式**：
```
E_total = E_DFT + E_correction(descriptors)
```

其中：
- `E_DFT`：标准 DFT 能量
- `E_correction`：神经网络预测的修正能量
- `descriptors`：投影密度矩阵的本征值（描述子）

### 1.2 描述子构建

**物理量**：投影密度矩阵（Projected Density Matrix, PDM）

**构建过程**：
1. 计算密度矩阵：`ρ = Σ_i |ψ_i⟩⟨ψ_i|`（对占据轨道求和）
2. 投影到局域基组：`D^I = ⟨α^I|ρ|α^I⟩`
3. 对角化得到本征值：`eig(D^I) → {λ_1, λ_2, ..., λ_n}`
4. 本征值作为描述子输入神经网络

**代码位置**：
- `deepks/core/physics/pyscf/scf.py::t_make_pdm()` - 计算投影密度矩阵
- `deepks/core/physics/pyscf/scf.py::t_make_eig()` - 计算本征值描述子

### 1.3 链式法则与导数计算

**物理量的导数关系**：

对于能量修正 `E_c = NN(descriptors)`，需要计算其对各物理量的导数：

1. **力（Force）**：能量对原子坐标的导数
   ```
   F = -∂E/∂R = -∂E/∂λ · ∂λ/∂R
   ```
   - `∂E/∂λ`：神经网络反向传播得到
   - `∂λ/∂R`：从外部读入（预先计算好）

2. **应力（Stress）**：能量对晶胞应变的导数
   ```
   σ = ∂E/∂ε = ∂E/∂λ · ∂λ/∂ε
   ```

3. **其他物理量**：类似的链式法则

**代码位置**：
- `deepks/core/physics/pyscf/scf.py::t_get_corr()` - 计算修正能量和势能
- `deepks/core/physics/pyscf/grad.py::t_grad_corr()` - 计算梯度（力）
- `deepks/io/readers/reader.py` - 读取预计算的导数 `gvx`（∂λ/∂R）

---

## 二、迭代训练流程

### 2.1 整体流程

```
初始化模型（随机或零初始化）
    ↓
[迭代循环开始]
    ↓
第一步：SCF 计算（00.scf/）
    - 使用当前模型进行自洽场计算
    - 计算修正后的能量、力、应力等
    - 与参考值（高精度计算）比较
    - 保存差异作为训练标签
    ↓
第二步：训练（01.train/）
    - 使用 SCF 计算的差异作为标签
    - 训练神经网络学习修正
    - 更新模型参数
    ↓
[下一轮迭代]
```

**代码位置**：
- `deepks/pipelines/iterate/iterate.py::make_iterate()` - 构建迭代工作流
- `deepks/pipelines/iterate/iterate.py::main()` - 执行迭代

### 2.2 初始轮（iter.init）特殊处理

**物理原因**：
- 初始模型参数是随机或零初始化
- 直接使用多种标签（能量+力+应力）会导致训练不稳定
- 需要先用能量标签预训练，使模型达到合理初值

**实现**：
```python
# 初始轮：仅使用能量标签
init_scf:  no_model=True  # 不使用模型，纯 DFT 计算
init_train: energy_factor=1.0, force_factor=0.0  # 仅能量

# 后续轮：使用多种标签
scf:  no_model=False  # 使用当前模型
train: energy_factor=1.0, force_factor=1.0, stress_factor=0.1  # 多种标签
```

**代码位置**：
- `deepks/pipelines/iterate/iterate.py` 第 300-333 行 - 初始轮构建
- `deepks/pipelines/iterate/template.py::make_scf()` - SCF 任务生成
- `deepks/pipelines/iterate/template.py::make_train()` - 训练任务生成

---

## 三、核心组件与代码对应

### 3.1 神经网络模型（CorrNet）

**物理功能**：学习能量修正函数 `E_c = f(descriptors)`

**网络结构**：
```
输入层：描述子（投影密度矩阵本征值）
    ↓
[可选] 嵌入层（Embedding）：对称化处理
    ↓
隐藏层：多层全连接网络（DenseNet）
    - 支持 ResNet 结构
    - 可配置激活函数（tanh, relu, gelu 等）
    ↓
输出层：能量修正值（标量）
```

**关键特性**：
1. **输入归一化**：`(x - shift) / scale`
2. **预拟合层**：线性拟合 `y = w·x + b`
3. **元素常数**：不同元素的能量偏移
4. **输出缩放**：`output / output_scale`

**代码位置**：
- `deepks/core/ml/models/corrnet.py::CorrNet` - 模型定义
- `deepks/core/ml/models/corrnet.py::DenseNet` - 全连接网络
- `deepks/core/ml/models/corrnet.py::TraceEmbedding` - 简单嵌入
- `deepks/core/ml/models/corrnet.py::ThermalEmbedding` - 热力学嵌入

### 3.2 SCF 计算（Self-Consistent Field）

**物理功能**：在给定模型下进行自洽场计算

**计算流程**：
```
1. 初始化密度矩阵 ρ
2. [迭代开始]
3. 计算 Fock 矩阵：F = h + J[ρ] + K[ρ] + V_c[ρ]
   - h：单电子哈密顿量
   - J, K：库仑和交换项
   - V_c：神经网络修正势能
4. 求解本征值问题：F·C = S·C·ε
5. 更新密度矩阵：ρ = Σ_i |C_i⟩⟨C_i|
6. 检查收敛：|ρ_new - ρ_old| < threshold
7. [收敛则退出，否则返回步骤 3]
```

**修正势能计算**：
```python
# 计算描述子
descriptors = eigenvalues(project(ρ))

# 神经网络前向传播
E_c = model(descriptors)

# 反向传播得到势能
V_c = ∂E_c/∂ρ
```

**代码位置**：
- `deepks/core/physics/pyscf/scf.py::RDSCF/UDSCF` - SCF 主类
- `deepks/core/physics/pyscf/scf.py::NetMixin` - 神经网络集成
- `deepks/core/physics/pyscf/scf.py::t_get_corr()` - 修正能量和势能
- `deepks/pipelines/scf/run.py::solve_mol()` - 单分子 SCF
- `deepks/pipelines/scf/run.py::run_scf()` - 批量 SCF

### 3.3 梯度计算（核力）

**物理功能**：计算能量对原子坐标的导数（即原子受力）

**Pulay 力**：由于基组随原子移动产生的额外力
```
F_total = F_Hellmann-Feynman + F_Pulay

F_Pulay = -∂E_c/∂R|_basis
        = -Σ_pq (∂E_c/∂D_pq) · (∂D_pq/∂R)
```

**代码位置**：
- `deepks/core/physics/pyscf/grad.py::Gradients` - 梯度计算主类
- `deepks/core/physics/pyscf/grad.py::t_grad_corr()` - 修正项梯度
- `deepks/core/physics/pyscf/grad.py::t_make_grad_pdm_x()` - PDM 对坐标的导数

### 3.4 训练过程（Training）

**物理功能**：最小化预测值与参考值之间的差异

**损失函数**：
```
L_total = α_E·L_energy + α_F·L_force + α_S·L_stress + ...

其中：
- L_energy = ||E_pred - E_ref||²
- L_force = ||F_pred - F_ref||²
- L_stress = ||σ_pred - σ_ref||²
```

**训练流程**：
```
1. 从数据集采样批次
2. 前向传播：
   - descriptors → model → E_pred
   - 自动微分 → F_pred, σ_pred, ...
3. 计算损失：L = Σ α_i·L_i
4. 反向传播：∂L/∂θ
5. 更新参数：θ ← θ - lr·∂L/∂θ
6. 学习率衰减
7. 保存检查点
```

**代码位置**：
- `deepks/core/ml/train/train.py::train()` - 训练主函数
- `deepks/core/ml/eval/evaluator.py::Evaluator` - 损失计算
- `deepks/core/ml/utils.py::make_loss()` - 损失函数构造
- `deepks/core/ml/utils.py::preprocess()` - 数据预处理

### 3.5 数据读取（Data I/O）

**物理功能**：读取训练数据（描述子、能量、力等）

**数据格式**：
```
system/
├── system.raw          # 元数据：natom, nproj
├── energy.npy          # 能量标签 [nframes]
├── force.npy           # 力标签 [nframes, natom, 3]
├── stress.npy          # 应力标签 [nframes, 6]
├── descriptor.npy      # 描述子 [nframes, natom, ndesc]
├── grad_vx.npy         # ∂λ/∂R [nframes, natom, 3, natom, ndesc]
└── grad_vepsl.npy      # ∂λ/∂ε [nframes, 6, natom, ndesc]
```

**采样策略**：
- 单系统：`Reader` - 从单个系统采样
- 多系统：`GroupReader` - 按概率从多个系统采样
- 分组批次：`group_batch` - 同时从多个系统采样

**代码位置**：
- `deepks/io/readers/reader.py::Reader` - 单系统读取器
- `deepks/io/readers/grouped_reader.py::GroupReader` - 多系统读取器
- `deepks/io/readers/sampling.py` - 采样辅助函数
- `deepks/io/readers/stats.py` - 统计计算

---

## 四、关键物理量与代码映射

| 物理量 | 符号 | 代码变量名 | 文件位置 |
|--------|------|-----------|----------|
| 密度矩阵 | ρ | `dm` | `scf.py` |
| 投影密度矩阵 | D^I | `pdm_shells` | `scf.py::t_make_pdm()` |
| 描述子（本征值） | λ | `eig`, `ceig` | `scf.py::t_make_eig()` |
| 修正能量 | E_c | `ec` | `scf.py::t_get_corr()` |
| 修正势能 | V_c | `vc` | `scf.py::t_get_corr()` |
| 能量标签 | E_ref | `lb_e` | `reader.py` |
| 力标签 | F_ref | `lb_f` | `reader.py` |
| 应力标签 | σ_ref | `lb_s` | `reader.py` |
| 描述子对坐标导数 | ∂λ/∂R | `gvx` | `reader.py` |
| 描述子对应变导数 | ∂λ/∂ε | `gvepsl` | `reader.py` |
| 能量损失 | L_E | `e_loss` | `evaluator.py` |
| 力损失 | L_F | `f_loss` | `evaluator.py` |

---

## 五、迭代训练的物理意义

### 5.1 为什么需要迭代？

**物理原因**：
1. **自洽性**：DFT 是自洽方法，密度矩阵和哈密顿量相互依赖
2. **修正的自洽**：加入神经网络修正后，需要重新自洽
3. **数据一致性**：训练数据应该来自使用当前模型的 SCF 计算

**迭代改进**：
```
第 0 轮：纯 DFT → 与参考值差异大 → 训练初始模型
第 1 轮：DFT + 初始模型 → 差异减小 → 改进模型
第 2 轮：DFT + 改进模型 → 差异更小 → 进一步改进
...
第 N 轮：收敛，差异达到目标精度
```

### 5.2 收敛判据

**物理指标**：
- 能量误差：`MAE(E_pred, E_ref) < threshold`
- 力误差：`MAE(F_pred, F_ref) < threshold`
- 模型变化：`||θ_new - θ_old|| < threshold`

**代码位置**：
- `deepks/core/ml/eval/evaluator.py` - 误差计算
- `deepks/pipelines/iterate/iterate.py` - 迭代控制

---

## 六、代码执行流程示例

### 6.1 完整迭代训练

```bash
# 命令行
deepks iterate \
    --systems systems_train.raw \
    --n-iter 5 \
    --scf-input scf_input.yaml \
    --train-input train_input.yaml
```

**执行流程**：
```
1. deepks/cli/main.py::iter_cli()
   ↓
2. deepks/pipelines/iterate/iterate.py::main()
   ↓
3. make_iterate() 构建工作流
   ├─ iter.init/
   │  ├─ 00.scf/  → make_scf(no_model=True)
   │  └─ 01.train/ → make_train(energy_only)
   ├─ iter.00/
   │  ├─ 00.scf/  → make_scf(with_model)
   │  └─ 01.train/ → make_train(multi_label)
   ├─ iter.01/
   │  └─ ...
   └─ iter.04/
   ↓
4. 执行工作流
   - SCF: deepks/pipelines/scf/run.py::run_scf()
     → deepks/core/physics/pyscf/scf.py::DSCF.kernel()
   - Train: deepks/pipelines/train/train.py::train()
     → deepks/core/ml/train/train.py::train()
```

### 6.2 单次 SCF 计算

```python
from deepks.core.physics.pyscf.scf import DSCF
from pyscf import gto

# 构建分子
mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='ccpvdz')

# 加载模型
model = "model.pth"

# SCF 计算
mf = DSCF(mol, model, xc='PBE')
energy = mf.kernel()

# 计算力
forces = mf.nuc_grad_method().kernel()
```

**执行流程**：
```
1. DSCF.__init__()
   - 加载神经网络模型
   - 准备投影基组
   ↓
2. DSCF.kernel()
   - 初始化密度矩阵
   - [SCF 迭代]
     - get_veff() → 计算有效势能
       - get_veff0() → 标准 DFT 势能
       - get_corr() → 神经网络修正
         - t_make_eig() → 计算描述子
         - model() → 神经网络前向传播
         - autograd → 反向传播得到势能
     - 求解本征值问题
     - 更新密度矩阵
     - 检查收敛
   ↓
3. nuc_grad_method().kernel()
   - grad_elec() → 电子部分梯度
     - grad_elec0() → 标准 DFT 梯度
     - grad_corr() → 神经网络修正梯度
       - t_grad_corr() → Pulay 力
   - grad_nuc() → 核-核排斥梯度
```

### 6.3 单次训练

```python
from deepks.core.ml.train.train import train
from deepks.core.ml.models.corrnet import CorrNet
from deepks.io.readers import GroupReader

# 加载模型
model = CorrNet.load("model.pth")

# 加载数据
reader = GroupReader(["sys1", "sys2"], batch_size=16)

# 训练
train(
    model, reader,
    n_epoch=1000,
    energy_factor=1.0,
    force_factor=1.0,
    start_lr=0.001
)
```

**执行流程**：
```
1. train()
   - 设置优化器（Adam）
   - 设置学习率调度器
   - 创建 Evaluator
   ↓
2. [训练循环]
   for epoch in range(n_epoch):
     for batch in reader:
       ├─ model.train()
       ├─ optimizer.zero_grad()
       ├─ loss = evaluator(model, batch)
       │  ├─ 前向传播：E_pred = model(descriptors)
       │  ├─ 自动微分：F_pred = -∂E_pred/∂R
       │  └─ 计算损失：L = α_E·L_E + α_F·L_F
       ├─ loss.backward()
       └─ optimizer.step()
     ├─ 评估测试集
     ├─ 学习率衰减
     └─ 保存检查点
```

---

## 七、优化建议与物理考虑

### 7.1 当前架构的物理合理性

**优点**：
1. ✅ 清晰的物理层次：Core（物理计算）→ I/O（数据）→ Pipelines（流程）
2. ✅ 模块化设计：SCF、训练、迭代相互独立
3. ✅ 支持多种标签：能量、力、应力、轨道等
4. ✅ 灵活的损失函数：可配置权重和损失类型

**待改进**：
1. ⚠️ 大文件拆分：`template_abacus.py` (1159 行) 过大
2. ⚠️ 类型注解：物理量缺少单位和维度信息
3. ⚠️ 文档：物理公式和代码对应关系不够清晰

### 7.2 物理相关的优化方向

**短期（1-2 周）**：
1. 为物理量添加单位注释
   ```python
   energy: float  # Hartree
   force: np.ndarray  # Hartree/Bohr, shape [natom, 3]
   ```

2. 提取物理常数到配置文件
   ```python
   # constants.py
   HARTREE_TO_EV = 27.211386245988
   BOHR_TO_ANGSTROM = 0.529177210903
   ```

3. 添加物理量验证
   ```python
   def validate_force(force, natom):
       assert force.shape == (natom, 3)
       assert np.isfinite(force).all()
   ```

**中期（1-2 月）**：
1. 实现更多物理后端
   - 完成 ABACUS 集成
   - 支持 CP2K、Quantum ESPRESSO

2. 扩展描述子类型
   - 当前：投影密度矩阵本征值
   - 扩展：SOAP、ACSF、Behler-Parrinello

3. 改进训练策略
   - 课程学习：先简单后复杂
   - 主动学习：选择最有信息量的样本

**长期（3-6 月）**：
1. 分布式训练
   - 数据并行：多 GPU 训练
   - 模型并行：大模型支持

2. 不确定性量化
   - 集成学习：多模型投票
   - 贝叶斯神经网络

3. 可解释性分析
   - 注意力机制：哪些描述子最重要
   - 敏感性分析：物理量对参数的依赖

---

## 八、总结

### 8.1 核心物理流程

```
高精度参考数据（CCSD(T)）
    ↓
[迭代训练循环]
    ↓
SCF 计算（DFT + NN 修正）
    - 输入：原子坐标、基组
    - 计算：自洽场迭代
    - 输出：能量、力、应力等
    - 与参考值比较 → 差异
    ↓
神经网络训练
    - 输入：描述子（PDM 本征值）
    - 标签：SCF 与参考值的差异
    - 优化：最小化预测误差
    - 输出：更新的模型
    ↓
[下一轮迭代，直到收敛]
```

### 8.2 代码-物理对应总结

| 物理概念 | 代码模块 | 关键文件 |
|---------|---------|---------|
| 密度泛函理论 | `core/physics/pyscf/` | `scf.py` |
| 神经网络模型 | `core/ml/models/` | `corrnet.py` |
| 描述子计算 | `core/physics/pyscf/` | `scf.py::t_make_eig()` |
| 能量修正 | `core/physics/pyscf/` | `scf.py::t_get_corr()` |
| 力计算 | `core/physics/pyscf/` | `grad.py` |
| 训练过程 | `core/ml/train/` | `train.py` |
| 损失函数 | `core/ml/eval/` | `evaluator.py` |
| 数据读取 | `io/readers/` | `reader.py` |
| 迭代流程 | `pipelines/iterate/` | `iterate.py` |
| 工作流编排 | `orchestration/workflow/` | `task.py`, `workflow.py` |

### 8.3 使用建议

**对于开发者**：
1. 修改物理计算：关注 `core/physics/`
2. 改进神经网络：关注 `core/ml/models/`
3. 调整训练策略：关注 `core/ml/train/` 和 `core/ml/eval/`
4. 扩展数据格式：关注 `io/`
5. 优化工作流：关注 `orchestration/` 和 `pipelines/`

**对于用户**：
1. 准备数据：参考 `io/readers/` 的数据格式
2. 配置参数：参考 `default.py` 的默认参数
3. 运行迭代：使用 `deepks iterate` 命令
4. 分析结果：使用 `deepks stats` 命令

---

**文档版本**：1.0
**创建日期**：2026-03-18
**作者**：Claude (Sonnet 4.6)
