# DeePKS-kit 输入参数指南

本文档提供 DeePKS-kit 统一输入文件 `input.yaml` 的完整参数说明。

**文档版本**：1.0
**创建日期**：2026-03-18
**作者**：Claude (claude-sonnet-4-6)

---

## 参数结构概览

```yaml
type: iterate
common: {...}
systems: {...}
model: {...}
data: {...}
preprocess: {...}
train: {...}
scf: {...}
iterate: {...}
scheduler: {...}
stats: {...}
```

---

## 1. 顶层参数

### type

- **类型**: string
- **取值范围**: `"train"` | `"test"` | `"scf"` | `"stats"` | `"iterate"`
- **描述**: 计算类型，控制程序执行的主要任务
  - `"train"`: 训练神经网络模型
  - `"test"`: 测试模型性能
  - `"scf"`: 运行自洽场计算
  - `"stats"`: 统计分析结果
  - `"iterate"`: 迭代训练（SCF + 训练循环）
- **默认值**: `"iterate"`

### common.workdir

- **类型**: string
- **取值范围**: 有效的目录路径
- **描述**: 工作目录，所有相对路径的基准目录
- **默认值**: `"."`（当前目录）

### common.seed

- **类型**: int | null
- **取值范围**: 任意整数或 null
- **描述**: 随机种子，用于保证结果可重复性。设为 null 则使用随机种子
- **默认值**: `null`

### common.device

- **类型**: string
- **取值范围**: `"cpu"` | `"cuda"` | `"cuda:0"` | `"cuda:1"` | ...
- **描述**: 计算设备，指定使用 CPU 或 GPU
- **默认值**: `"cpu"`

### common.verbose

- **类型**: int
- **取值范围**: 0-5
- **描述**: 输出详细程度，数值越大输出越详细
  - 0: 仅错误信息
  - 1: 基本信息
  - 2-5: 更详细的调试信息
- **默认值**: `1`

---

## 2. 系统路径参数 (systems)

### systems.train

- **类型**: list[string] | null
- **取值范围**: 系统目录路径列表
- **描述**: 训练数据系统路径列表，每个路径指向一个包含描述子、能量等数据的目录
- **默认值**: `null`（必需参数，用户必须提供）
- **注意**: 对于 `type="train"` 和 `type="iterate"`，此参数为必需

### systems.test

- **类型**: list[string] | null
- **取值范围**: 系统目录路径列表
- **描述**: 测试数据系统路径列表，用于评估模型性能。若为 null 则使用训练集
- **默认值**: `null`

---

## 3. 模型参数 (model)

### model.file

- **类型**: string | null
- **取值范围**: 有效的模型文件路径（.pth 文件）
- **描述**: 预训练模型文件路径，用于加载已有模型继续训练或进行推理
- **默认值**: `null`（从头开始训练）

### model.checkpoint

- **类型**: string
- **取值范围**: 文件路径
- **描述**: 训练过程中保存的检查点文件名
- **默认值**: `"model.pth"`

### model.proj_basis

- **类型**: string | list | null
- **取值范围**: 投影基组文件路径或基组定义列表
- **描述**: 投影基组，用于计算投影密度矩阵描述子。可以是文件路径或直接定义的基组列表
- **默认值**: `null`（必需参数）

### model.elem_table

- **类型**: dict | null
- **取值范围**: 元素符号到能量常数的映射，如 `{"H": -0.5, "O": -75.0}`
- **描述**: 元素能量常数表，用于拟合每个元素的基准能量
- **默认值**: `null`

### model.fit_elem

- **类型**: bool
- **取值范围**: true | false
- **描述**: 是否拟合元素能量常数。若为 true，则在训练过程中自动拟合每个元素的基准能量
- **默认值**: `false`

### model.architecture.hidden_sizes

- **类型**: list[int]
- **取值范围**: 正整数列表
- **描述**: 神经网络隐藏层神经元数列表，定义网络的宽度和深度
- **默认值**: `[100, 100, 100]`（3层，每层100个神经元）

### model.architecture.actv_fn

- **类型**: string
- **取值范围**: `"gelu"` | `"relu"` | `"tanh"` | `"sigmoid"` | `"silu"` | `"softplus"` | `"mygelu"`
- **描述**: 激活函数类型
- **默认值**: `"gelu"`

### model.architecture.use_resnet

- **类型**: bool
- **取值范围**: true | false
- **描述**: 是否使用残差连接（ResNet），有助于训练深层网络
- **默认值**: `true`

### model.architecture.layer_norm

- **类型**: bool | string
- **取值范围**: false | true | `"simple"`
- **描述**: 是否使用层归一化
  - false: 不使用
  - true: 标准层归一化
  - `"simple"`: 简化版层归一化
- **默认值**: `false`

### model.architecture.output_scale

- **类型**: float
- **取值范围**: 正数
- **描述**: 输出缩放因子，用于调整模型输出的数值范围
- **默认值**: `100`

### model.architecture.input_shift

- **类型**: float
- **取值范围**: 任意实数
- **描述**: 输入平移量，用于数据预处理
- **默认值**: `0`

### model.architecture.input_scale

- **类型**: float
- **取值范围**: 正数
- **描述**: 输入缩放因子，用于数据归一化
- **默认值**: `1`

### model.embedding.type

- **类型**: string | null
- **取值范围**: null | `"trace"` | `"thermal"`
- **描述**: 嵌入层类型，用于增强模型表达能力
  - null: 不使用嵌入层
  - `"trace"`: 基于迹的嵌入
  - `"thermal"`: 热力学嵌入
- **默认值**: `null`

### model.embedding.embd_sizes

- **类型**: list[int] | null
- **取值范围**: 正整数列表或 null
- **描述**: 嵌入层维度列表，定义嵌入层的结构
- **默认值**: `null`

### model.embedding.init_beta

- **类型**: float
- **取值范围**: 正数
- **描述**: 嵌入层初始 beta 参数，控制嵌入的初始化
- **默认值**: `5.0`

### model.embedding.momentum

- **类型**: float | null
- **取值范围**: 0-1 或 null
- **描述**: 嵌入层动量参数，用于平滑嵌入更新
- **默认值**: `null`

### model.embedding.max_memory

- **类型**: int
- **取值范围**: 正整数
- **描述**: 嵌入层最大记忆批次数，限制嵌入层的内存使用
- **默认值**: `1000`

---

## 4. 数据参数 (data)

### data.batch_size

- **类型**: int
- **取值范围**: 正整数
- **描述**: 批次大小，每次训练使用的样本数量。较大的批次可以提高训练稳定性，但需要更多内存
- **默认值**: `16`

### data.group_batch

- **类型**: int
- **取值范围**: 正整数
- **描述**: 分组批次数，用于 GroupReader。将多个批次组合在一起处理
- **默认值**: `1`

### data.extra_label

- **类型**: bool
- **取值范围**: true | false
- **描述**: 是否读取额外标签数据（如力、应力等）。若为 false 则仅读取能量和描述子
- **默认值**: `true`

### data.conv_filter

- **类型**: bool
- **取值范围**: true | false
- **描述**: 是否过滤未收敛的 SCF 数据。若为 true，则自动排除收敛标志为 false 的数据
- **默认值**: `true`

### data.conv_name

- **类型**: string
- **取值范围**: 文件名
- **描述**: 收敛标志文件名，用于判断 SCF 是否收敛
- **默认值**: `"conv"`

### data.read_overlap

- **类型**: bool
- **取值范围**: true | false
- **描述**: 是否读取重叠矩阵数据。某些计算需要重叠矩阵信息
- **默认值**: `false`

### data.orb_list

- **类型**: list[string] | null
- **取值范围**: 轨道文件路径列表
- **描述**: ABACUS 轨道文件列表，用于 ABACUS 后端的数据读取
- **默认值**: `null`

### data.alpha_list

- **类型**: list[string] | null
- **取值范围**: alpha 文件路径列表
- **描述**: ABACUS alpha 文件列表，用于投影基组定义
- **默认值**: `null`

### 数据字段名参数

以下参数用于自定义数据文件名，允许用户使用非默认的文件命名：

#### data.e_name
- **类型**: string
- **描述**: 能量文件名
- **默认值**: `"energy"`

#### data.d_name
- **类型**: string
- **描述**: 描述子文件名
- **默认值**: `"descriptor"`

#### data.f_name
- **类型**: string
- **描述**: 力文件名
- **默认值**: `"force"`

#### data.gvx_name
- **类型**: string
- **描述**: 描述子对坐标梯度文件名
- **默认值**: `"grad_vx"`

#### data.s_name
- **类型**: string
- **描述**: 应力文件名
- **默认值**: `"stress"`

#### data.gvepsl_name
- **类型**: string
- **描述**: 描述子对应变梯度文件名
- **默认值**: `"grad_vepsl"`

#### data.o_name
- **类型**: string
- **描述**: 轨道文件名
- **默认值**: `"orbital"`

#### data.op_name
- **类型**: string
- **描述**: 预计算轨道文件名
- **默认值**: `"orbital_precalc"`

#### data.h_name
- **类型**: string
- **描述**: 哈密顿量文件名
- **默认值**: `"hamiltonian"`

#### data.vdp_name
- **类型**: string
- **描述**: v_delta 预计算文件名
- **默认值**: `"v_delta_precalc"`

#### data.vdrp_name
- **类型**: string
- **描述**: v_delta_r 预计算文件名
- **默认值**: `"v_delta_r_precalc"`

#### data.phialpha_name
- **类型**: string
- **描述**: phialpha 文件名
- **默认值**: `"phialpha"`

#### data.gevdm_name
- **类型**: string
- **描述**: 本征值对密度矩阵梯度文件名
- **默认值**: `"grad_eig_dm"`

#### data.hr_name
- **类型**: string
- **描述**: 实空间哈密顿量文件名
- **默认值**: `"hamiltonian_r"`

#### data.h_base_name
- **类型**: string
- **描述**: 基础哈密顿量文件名
- **默认值**: `"hamiltonian_base"`

#### data.h_ref_name
- **类型**: string
- **描述**: 参考哈密顿量文件名
- **默认值**: `"hamiltonian_ref"`

#### data.overlap_name
- **类型**: string
- **描述**: 重叠矩阵文件名
- **默认值**: `"overlap"`

#### data.eg_name
- **类型**: string
- **描述**: 本征值文件名
- **默认值**: `"eig"`

#### data.gveg_name
- **类型**: string
- **描述**: 本征值梯度文件名
- **默认值**: `"grad_eig"`

#### data.gldv_name
- **类型**: string
- **描述**: ldv 梯度文件名
- **默认值**: `"grad_ldv"`

#### data.atom_name
- **类型**: string
- **描述**: 原子信息文件名
- **默认值**: `"atom"`

#### data.box_name
- **类型**: string
- **描述**: 晶胞信息文件名
- **默认值**: `"box"`

---

## 5. 预处理参数 (preprocess)

### preprocess.preshift

- **类型**: bool
- **取值范围**: true | false
- **描述**: 是否对描述子进行平移预处理，将描述子平移到零均值。有助于提高训练稳定性
- **默认值**: `true`

### preprocess.prescale

- **类型**: bool
- **取值范围**: true | false
- **描述**: 是否对描述子进行缩放预处理，将描述子缩放到单位方差。通常与 preshift 配合使用
- **默认值**: `false`

### preprocess.prescale_sqrt

- **类型**: bool
- **取值范围**: true | false
- **描述**: 是否对缩放因子取平方根。用于某些特殊的归一化需求
- **默认值**: `false`

### preprocess.prescale_clip

- **类型**: float
- **取值范围**: 非负数
- **描述**: 缩放裁剪阈值。若大于 0，则对缩放因子进行裁剪，避免极端值。0 表示不裁剪
- **默认值**: `0`

### preprocess.prefit

- **类型**: bool
- **取值范围**: true | false
- **描述**: 是否进行预拟合。使用线性回归对描述子进行预拟合，为神经网络提供更好的初始化
- **默认值**: `true`

### preprocess.prefit_ridge

- **类型**: float
- **取值范围**: 非负数
- **描述**: 预拟合岭回归系数（L2 正则化系数）。较大的值可以提高预拟合的稳定性
- **默认值**: `10.0`

### preprocess.prefit_trainable

- **类型**: bool
- **取值范围**: true | false
- **描述**: 预拟合层是否可训练。若为 true，则预拟合的线性层参数在训练过程中可以更新
- **默认值**: `false`

---

## 6. 训练参数 (train)

### 训练控制

#### train.n_epoch

- **类型**: int
- **取值范围**: 正整数
- **描述**: 训练轮数，即遍历整个训练集的次数
- **默认值**: `5000`

#### train.test_reader

- **类型**: Reader | null
- **取值范围**: Reader 对象或 null
- **描述**: 测试集读取器。若为 null 则使用训练集进行测试
- **默认值**: `null`
- **注意**: 此参数通常由 `systems.test` 自动构建，用户一般不需要直接设置

### 优化器参数

#### train.start_lr

- **类型**: float
- **取值范围**: 正数
- **描述**: 初始学习率，控制参数更新的步长
- **默认值**: `0.001`

#### train.stop_lr

- **类型**: float | null
- **取值范围**: 正数或 null
- **描述**: 终止学习率。若设置，则根据此值自动计算 decay_rate
- **默认值**: `null`

#### train.decay_rate

- **类型**: float
- **取值范围**: 0-1
- **描述**: 学习率衰减率。每 decay_steps 轮后，学习率乘以此值
- **默认值**: `0.96`

#### train.decay_steps

- **类型**: int
- **取值范围**: 正整数
- **描述**: 学习率衰减步数。每经过此轮数，学习率衰减一次
- **默认值**: `100`

#### train.decay_rate_iter

- **类型**: float | null
- **取值范围**: 0-1 或 null
- **描述**: 迭代间学习率衰减率。在迭代训练中，每次迭代后学习率乘以此值
- **默认值**: `null`

#### train.weight_decay

- **类型**: float
- **取值范围**: 非负数
- **描述**: 权重衰减系数（L2 正则化）。用于防止过拟合
- **默认值**: `0.0`

#### train.fix_embedding

- **类型**: bool
- **取值范围**: true | false
- **描述**: 是否固定嵌入层参数。若为 true，则嵌入层参数在训练过程中不更新
- **默认值**: `false`

### 显示参数

#### train.display_epoch

- **类型**: int
- **取值范围**: 正整数
- **描述**: 显示间隔（轮数）。每经过此轮数，输出一次训练信息
- **默认值**: `100`

#### train.display_detail_test

- **类型**: int
- **取值范围**: 0-2
- **描述**: 测试详细程度
  - 0: 仅显示总体统计
  - 1: 显示每个系统的统计
  - 2: 显示详细的样本级信息
- **默认值**: `0`

#### train.display_natom_loss

- **类型**: bool
- **取值范围**: true | false
- **描述**: 是否显示按原子数分类的损失统计。有助于分析不同大小系统的训练情况
- **默认值**: `false`

#### train.save_freq

- **类型**: int | null
- **取值范围**: 正整数或 null
- **描述**: 保存频率（轮数）。每经过此轮数，保存一次模型检查点。null 表示仅在训练结束时保存
- **默认值**: `null`

### 损失函数权重 (train.loss_weights)

#### train.loss_weights.energy

- **类型**: float
- **取值范围**: 非负数
- **描述**: 能量损失权重。控制能量项在总损失中的比重
- **默认值**: `1.0`

#### train.loss_weights.force

- **类型**: float
- **取值范围**: 非负数
- **描述**: 力损失权重。控制力项在总损失中的比重
- **默认值**: `0.0`

#### train.loss_weights.stress

- **类型**: float
- **取值范围**: 非负数
- **描述**: 应力损失权重。控制应力项在总损失中的比重
- **默认值**: `0.0`

#### train.loss_weights.orbital

- **类型**: float
- **取值范围**: 非负数
- **描述**: 轨道损失权重。控制轨道项在总损失中的比重
- **默认值**: `0.0`

#### train.loss_weights.v_delta

- **类型**: float
- **取值范围**: 非负数
- **描述**: v_delta 损失权重。控制 v_delta 项在总损失中的比重
- **默认值**: `0.0`

#### train.loss_weights.v_delta_r

- **类型**: float
- **取值范围**: 非负数
- **描述**: v_delta_r 损失权重。控制 v_delta_r 项在总损失中的比重
- **默认值**: `0.0`

#### train.loss_weights.phi

- **类型**: float
- **取值范围**: 非负数
- **描述**: phi 损失权重。控制 phi 项在总损失中的比重
- **默认值**: `0.0`

#### train.loss_weights.band

- **类型**: float
- **取值范围**: 非负数
- **描述**: 能带损失权重。控制能带项在总损失中的比重
- **默认值**: `0.0`

#### train.loss_weights.density_m

- **类型**: float
- **取值范围**: 非负数
- **描述**: 密度矩阵损失权重。控制密度矩阵项在总损失中的比重
- **默认值**: `0.0`

#### train.loss_weights.density

- **类型**: float
- **取值范围**: 非负数
- **描述**: 密度损失权重。控制密度项在总损失中的比重
- **默认值**: `0.0`

### 损失函数配置 (train.loss_config)

每个损失项可以配置 `cap` 和 `shrink` 参数：

#### train.loss_config.energy

- **类型**: dict
- **子参数**:
  - `cap` (float | null): 损失上限，超过此值的损失被裁剪
  - `shrink` (float | null): 收缩因子，用于平滑损失函数
- **描述**: 能量损失函数配置
- **默认值**: `{"cap": null, "shrink": null}`

#### train.loss_config.force

- **类型**: dict
- **子参数**: 同 energy
- **描述**: 力损失函数配置
- **默认值**: `{"cap": null, "shrink": null}`

#### train.loss_config.stress

- **类型**: dict
- **子参数**: 同 energy
- **描述**: 应力损失函数配置
- **默认值**: `{"cap": null, "shrink": null}`

#### train.loss_config.orbital

- **类型**: dict
- **子参数**: 同 energy
- **描述**: 轨道损失函数配置
- **默认值**: `{"cap": null, "shrink": null}`

#### train.loss_config.v_delta

- **类型**: dict
- **子参数**: 同 energy
- **描述**: v_delta 损失函数配置
- **默认值**: `{"cap": null, "shrink": null}`

#### train.loss_config.v_delta_r

- **类型**: dict
- **子参数**: 同 energy
- **描述**: v_delta_r 损失函数配置
- **默认值**: `{"cap": null, "shrink": null}`

#### train.loss_config.phi

- **类型**: dict
- **子参数**: 同 energy
- **描述**: phi 损失函数配置
- **默认值**: `{"cap": null, "shrink": null}`

#### train.loss_config.band

- **类型**: dict
- **子参数**: 同 energy
- **描述**: 能带损失函数配置
- **默认值**: `{"cap": null, "shrink": null}`

#### train.loss_config.density_m

- **类型**: dict
- **子参数**: 同 energy
- **描述**: 密度矩阵损失函数配置
- **默认值**: `{"cap": null, "shrink": null}`

### 损失归一化

#### train.energy_per_atom

- **类型**: int
- **取值范围**: 0 | 1 | 2
- **描述**: 能量是否按原子数归一化
  - 0: 不归一化
  - 1: 除以原子数
  - 2: 除以原子数的平方根
- **默认值**: `0`

#### train.vd_divide_by_nlocal

- **类型**: bool
- **取值范围**: true | false
- **描述**: v_delta 损失是否除以 nlocal（局部原子数）
- **默认值**: `false`

### 其他训练参数

#### train.grad_penalty

- **类型**: float
- **取值范围**: 非负数
- **描述**: 梯度惩罚系数。用于正则化模型梯度，防止梯度爆炸
- **默认值**: `0.0`

#### train.phi_occ

- **类型**: int
- **取值范围**: 非负整数
- **描述**: phi 占据数。指定计算 phi 损失时使用的占据轨道数
- **默认值**: `0`

#### train.band_occ

- **类型**: int
- **取值范围**: 非负整数
- **描述**: 能带占据数。指定计算能带损失时使用的占据轨道数
- **默认值**: `0`

#### train.density_m_occ

- **类型**: int
- **取值范围**: 非负整数
- **描述**: 密度矩阵占据数。指定计算密度矩阵损失时使用的占据轨道数
- **默认值**: `0`

---

## 7. SCF 参数 (scf)

### scf.scf_soft

- **类型**: string
- **取值范围**: `"pyscf"` | `"abacus"`
- **描述**: SCF 后端选择器，指定使用哪个软件进行自洽场计算
  - `"pyscf"`: 使用 PySCF 后端（适用于分子和小周期体系）
  - `"abacus"`: 使用 ABACUS 后端（适用于大规模周期体系）
- **默认值**: `"pyscf"`
- **注意**: 两个后端互斥，运行时仅使用其中一个

### scf.dump_dir

- **类型**: string
- **取值范围**: 有效的目录路径
- **描述**: SCF 计算结果输出目录
- **默认值**: `"results"`

### scf.dump_fields

- **类型**: list[string]
- **取值范围**: 有效的字段名列表
- **描述**: 需要输出的字段列表，如 `["atom", "e_base", "e_tot", "dm_eig", "conv"]`
- **默认值**: `["atom", "e_base", "e_tot", "dm_eig", "conv"]`

### scf.group_output

- **类型**: bool
- **取值范围**: true | false
- **描述**: 是否分组输出结果。若为 true，则按系统分组保存结果
- **默认值**: `false`

### scf.penalties

- **类型**: list
- **取值范围**: 惩罚项配置列表
- **描述**: SCF 计算中的惩罚项列表，用于约束计算过程
- **默认值**: `[]`（空列表）

### 7.1 PySCF 后端参数 (scf.pyscf)

#### 基本参数

##### scf.pyscf.basis

- **类型**: string
- **取值范围**: PySCF 支持的基组名称，如 `"ccpvdz"`, `"6-31g"`, `"def2-svp"` 等
- **描述**: 基组，定义原子轨道的数学形式
- **默认值**: `"ccpvdz"`

##### scf.pyscf.xc

- **类型**: string
- **取值范围**: 交换相关泛函名称，如 `"HF"`, `"PBE"`, `"B3LYP"` 等
- **描述**: 交换相关泛函。`"HF"` 表示 Hartree-Fock 方法
- **默认值**: `"HF"`

#### SCF 控制参数

##### scf.pyscf.conv_tol

- **类型**: float
- **取值范围**: 正数
- **描述**: 能量收敛阈值。当相邻两次迭代的能量差小于此值时，认为 SCF 收敛
- **默认值**: `1e-7`

##### scf.pyscf.conv_tol_grad

- **类型**: float | null
- **取值范围**: 正数或 null
- **描述**: 梯度收敛阈值。若设置，则同时检查梯度收敛
- **默认值**: `null`

##### scf.pyscf.max_cycle

- **类型**: int
- **取值范围**: 正整数
- **描述**: 最大 SCF 循环数。超过此次数后，即使未收敛也停止计算
- **默认值**: `50`

##### scf.pyscf.diis_space

- **类型**: int
- **取值范围**: 正整数
- **描述**: DIIS（直接反演迭代子空间）空间大小。DIIS 用于加速 SCF 收敛
- **默认值**: `8`

##### scf.pyscf.level_shift

- **类型**: float
- **取值范围**: 非负数
- **描述**: 能级位移。用于改善 SCF 收敛性，特别是对于难收敛的体系
- **默认值**: `0.0`

##### scf.pyscf.conv_check

- **类型**: bool
- **取值范围**: true | false
- **描述**: 是否检查收敛。若为 false，则即使未收敛也继续计算
- **默认值**: `true`

#### 分子参数 (scf.pyscf.mol_args)

##### scf.pyscf.mol_args.charge

- **类型**: int
- **取值范围**: 整数
- **描述**: 分子总电荷。0 表示中性分子，正值表示阳离子，负值表示阴离子
- **默认值**: `0`

##### scf.pyscf.mol_args.spin

- **类型**: int
- **取值范围**: 非负整数
- **描述**: 自旋多重度（2S），其中 S 是总自旋。0 表示单重态，1 表示双重态，等等
- **默认值**: `0`

##### scf.pyscf.mol_args.unit

- **类型**: string
- **取值范围**: `"Angstrom"` | `"Bohr"`
- **描述**: 坐标单位
- **默认值**: `"Angstrom"`

##### scf.pyscf.mol_args.incore_anyway

- **类型**: bool
- **取值范围**: true | false
- **描述**: 是否强制使用内存计算。若为 true，则所有积分都在内存中计算
- **默认值**: `false`

#### 网格参数 (scf.pyscf.grids)

##### scf.pyscf.grids.level

- **类型**: int
- **取值范围**: 0-9
- **描述**: DFT 数值积分网格精度等级。数值越大，网格越密，计算越精确但也越慢
- **默认值**: `3`

### 7.2 ABACUS 后端参数 (scf.abacus)

#### 运行参数

##### scf.abacus.abacus_path

- **类型**: string
- **取值范围**: 有效的可执行文件路径
- **描述**: ABACUS 可执行文件路径
- **默认值**: `"/usr/local/bin/ABACUS.mpi"`

##### scf.abacus.run_cmd

- **类型**: string
- **取值范围**: 有效的命令，如 `"mpirun"`, `"srun"` 等
- **描述**: 运行命令，用于启动 ABACUS 并行计算
- **默认值**: `"mpirun"`

#### 文件参数

##### scf.abacus.orb_files

- **类型**: list[string]
- **取值范围**: 轨道文件路径列表
- **描述**: ABACUS 数值原子轨道文件列表，每个元素对应一个元素的轨道文件
- **默认值**: `["orb"]`

##### scf.abacus.pp_files

- **类型**: list[string]
- **取值范围**: 赝势文件路径列表
- **描述**: 赝势文件列表，每个元素对应一个元素的赝势文件（UPF 格式）
- **默认值**: `["upf"]`

##### scf.abacus.proj_file

- **类型**: list[string]
- **取值范围**: 投影文件路径列表
- **描述**: DeePKS 投影基组文件列表
- **默认值**: `["orb"]`

#### 晶格参数

##### scf.abacus.lattice_constant

- **类型**: float
- **取值范围**: 正数
- **描述**: 晶格常数（单位：Bohr）。晶格矢量会乘以此常数
- **默认值**: `1`

##### scf.abacus.lattice_vector

- **类型**: list[list[float]] | null
- **取值范围**: 3x3 矩阵
- **描述**: 晶格矢量矩阵。若为 null，则从输入文件中读取
- **默认值**: `null`（从输入文件读取）

##### scf.abacus.coord_type

- **类型**: string
- **取值范围**: `"Cartesian"` | `"Direct"`
- **描述**: 坐标类型
  - `"Cartesian"`: 笛卡尔坐标（单位：Angstrom）
  - `"Direct"`: 分数坐标（相对于晶格矢量）
- **默认值**: `"Cartesian"`

#### STRU 文件参数

##### scf.abacus.nspin

- **类型**: int
- **取值范围**: 1 | 2 | 4
- **描述**: 自旋类型
  - 1: 非自旋极化
  - 2: 自旋极化（共线）
  - 4: 非共线自旋
- **默认值**: `1`

##### scf.abacus.symmetry

- **类型**: int
- **取值范围**: 0 | 1
- **描述**: 是否使用对称性
  - 0: 不使用对称性
  - 1: 使用对称性
- **默认值**: `0`

##### scf.abacus.nbands

- **类型**: int | null
- **取值范围**: 正整数或 null
- **描述**: 能带数。若为 null，则自动计算
- **默认值**: `null`

#### INPUT 文件参数

##### scf.abacus.ecutwfc

- **类型**: float
- **取值范围**: 正数
- **描述**: 波函数截断能（单位：Ry）。控制平面波基组的大小
- **默认值**: `50`

##### scf.abacus.scf_thr

- **类型**: float
- **取值范围**: 正数
- **描述**: SCF 收敛阈值（能量，单位：eV）
- **默认值**: `1e-7`

##### scf.abacus.scf_nmax

- **类型**: int
- **取值范围**: 正整数
- **描述**: 最大 SCF 步数
- **默认值**: `50`

##### scf.abacus.dft_functional

- **类型**: string
- **取值范围**: `"pbe"`, `"lda"`, `"hse"` 等
- **描述**: DFT 泛函类型
- **默认值**: `"pbe"`

##### scf.abacus.basis_type

- **类型**: string
- **取值范围**: `"lcao"` | `"pw"`
- **描述**: 基组类型
  - `"lcao"`: 线性组合原子轨道
  - `"pw"`: 平面波
- **默认值**: `"lcao"`

##### scf.abacus.gamma_only

- **类型**: int
- **取值范围**: 0 | 1
- **描述**: 是否仅使用 Gamma 点
  - 0: 使用 k 点网格
  - 1: 仅 Gamma 点
- **默认值**: `1`

##### scf.abacus.k_points

- **类型**: list[int] | null
- **取值范围**: 三个正整数的列表，如 `[4, 4, 4]`
- **描述**: k 点网格。若为 null，则使用 kspacing 或默认值
- **默认值**: `null`

##### scf.abacus.kspacing

- **类型**: float | null
- **取值范围**: 正数或 null
- **描述**: k 点间距（单位：1/Bohr）。用于自动生成 k 点网格
- **默认值**: `null`

##### scf.abacus.smearing_method

- **类型**: string
- **取值范围**: `"gaussian"`, `"fd"`, `"mp"` 等
- **描述**: 展宽方法
  - `"gaussian"`: 高斯展宽
  - `"fd"`: Fermi-Dirac 展宽
  - `"mp"`: Methfessel-Paxton 展宽
- **默认值**: `"gaussian"`

##### scf.abacus.smearing_sigma

- **类型**: float
- **取值范围**: 正数
- **描述**: 展宽参数（单位：Ry）
- **默认值**: `0.02`

##### scf.abacus.mixing_type

- **类型**: string
- **取值范围**: `"pulay"`, `"broyden"`, `"plain"` 等
- **描述**: 电荷密度混合类型
- **默认值**: `"pulay"`

##### scf.abacus.mixing_beta

- **类型**: float
- **取值范围**: 0-1
- **描述**: 混合参数。控制新旧电荷密度的混合比例
- **默认值**: `0.4`

##### scf.abacus.cal_force

- **类型**: int
- **取值范围**: 0 | 1
- **描述**: 是否计算力
  - 0: 不计算
  - 1: 计算
- **默认值**: `0`

##### scf.abacus.cal_stress

- **类型**: int
- **取值范围**: 0 | 1
- **描述**: 是否计算应力
  - 0: 不计算
  - 1: 计算
- **默认值**: `0`

#### DeePKS 相关参数

##### scf.abacus.deepks_bandgap

- **类型**: int
- **取值范围**: 0 | 1
- **描述**: 是否使用 DeePKS 能隙修正
- **默认值**: `0`

##### scf.abacus.deepks_v_delta

- **类型**: int
- **取值范围**: 0 | 1
- **描述**: 是否计算 DeePKS v_delta
- **默认值**: `0`

##### scf.abacus.deepks_out_labels

- **类型**: int
- **取值范围**: 0 | 1
- **描述**: 是否输出 DeePKS 标签数据
- **默认值**: `1`

##### scf.abacus.deepks_scf

- **类型**: int
- **取值范围**: 0 | 1
- **描述**: 是否在 SCF 中使用 DeePKS 修正
- **默认值**: `0`

##### scf.abacus.out_wfc_lcao

- **类型**: int
- **取值范围**: 0 | 1
- **描述**: 是否输出 LCAO 波函数
- **默认值**: `0`

---

## 8. 迭代参数 (iterate)

### iterate.n_iter

- **类型**: int
- **取值范围**: 正整数
- **描述**: 迭代次数。指定 SCF-训练循环的迭代轮数
- **默认值**: `5`

### iterate.share_folder

- **类型**: string
- **取值范围**: 有效的目录名
- **描述**: 共享文件夹名称。用于存放迭代过程中的共享文件（如模型、配置等）
- **默认值**: `"share"`

### iterate.cleanup

- **类型**: bool
- **取值范围**: true | false
- **描述**: 是否清理中间文件。若为 true，则在迭代完成后删除中间计算文件
- **默认值**: `false`

### iterate.strict

- **类型**: bool
- **取值范围**: true | false
- **描述**: 是否严格检查参数。若为 true，则对参数进行严格验证
- **默认值**: `true`

### iterate.init.use_model

- **类型**: bool
- **取值范围**: true | false
- **描述**: 初始迭代是否使用已有模型。若为 true，则从 model_file 加载模型
- **默认值**: `false`

### iterate.init.model_file

- **类型**: string | null
- **取值范围**: 有效的模型文件路径或 null
- **描述**: 初始迭代使用的模型文件路径。仅当 use_model 为 true 时有效
- **默认值**: `null`

### iterate.init.scf

- **类型**: dict
- **取值范围**: SCF 参数字典
- **描述**: 初始迭代的 SCF 参数配置。默认继承 `scf` 的所有参数，用户可覆盖部分参数
- **默认值**: `{}`（继承 `scf` 参数）
- **注意**: 此参数支持参数继承，详见附录

### iterate.init.train

- **类型**: dict
- **取值范围**: 训练参数字典
- **描述**: 初始迭代的训练参数配置。默认继承 `train` 的所有参数，用户可覆盖部分参数
- **默认值**: `{}`（继承 `train` 参数）
- **注意**: 此参数支持参数继承，详见附录

---

## 9. 调度器参数 (scheduler)

### 9.1 SCF 调度器 (scheduler.scf)

#### 任务分组

##### scheduler.scf.sub_size

- **类型**: int
- **取值范围**: 正整数
- **描述**: 每个子任务包含的系统数。将多个系统打包成一个子任务
- **默认值**: `1`

##### scheduler.scf.group_size

- **类型**: int
- **取值范围**: 正整数
- **描述**: 每个作业包含的子任务数。将多个子任务打包成一个作业提交
- **默认值**: `1`

##### scheduler.scf.ingroup_parallel

- **类型**: int
- **取值范围**: 正整数
- **描述**: 作业内并行执行的子任务数。控制作业内的并行度
- **默认值**: `1`

#### 调度器配置 (scheduler.scf.dispatcher)

##### scheduler.scf.dispatcher.context

- **类型**: string
- **取值范围**: `"local"` | `"ssh"`
- **描述**: 执行环境
  - `"local"`: 本地执行
  - `"ssh"`: 通过 SSH 远程执行
- **默认值**: `"local"`

##### scheduler.scf.dispatcher.batch

- **类型**: string
- **取值范围**: `"shell"` | `"slurm"` | `"pbs"` | `"lsf"`
- **描述**: 调度系统类型
  - `"shell"`: 直接在 shell 中执行
  - `"slurm"`: 使用 Slurm 调度器
  - `"pbs"`: 使用 PBS/Torque 调度器
  - `"lsf"`: 使用 LSF 调度器
- **默认值**: `"shell"`

##### scheduler.scf.dispatcher.remote_profile

- **类型**: dict | null
- **取值范围**: SSH 配置字典或 null
- **描述**: SSH 远程连接配置。仅当 context 为 `"ssh"` 时有效
- **默认值**: `null`

#### 资源配置 (scheduler.scf.resources)

##### scheduler.scf.resources.numb_node

- **类型**: int
- **取值范围**: 正整数
- **描述**: 节点数。每个作业使用的计算节点数
- **默认值**: `1`

##### scheduler.scf.resources.cpus_per_task

- **类型**: int
- **取值范围**: 正整数
- **描述**: 每个任务使用的 CPU 核心数
- **默认值**: `8`

##### scheduler.scf.resources.numb_gpu

- **类型**: int
- **取值范围**: 非负整数
- **描述**: GPU 数量。每个作业使用的 GPU 数
- **默认值**: `0`

##### scheduler.scf.resources.mem_limit

- **类型**: int
- **取值范围**: 正整数
- **描述**: 内存限制（单位：GB）
- **默认值**: `8`

##### scheduler.scf.resources.time_limit

- **类型**: string
- **取值范围**: 时间格式字符串，如 `"24:00:00"`, `"1-12:00:00"`
- **描述**: 作业时间限制。格式为 `"HH:MM:SS"` 或 `"D-HH:MM:SS"`
- **默认值**: `"24:00:00"`

##### scheduler.scf.resources.task_per_node

- **类型**: int
- **取值范围**: 正整数
- **描述**: 每个节点运行的任务数
- **默认值**: `1`

##### scheduler.scf.resources.exclusive

- **类型**: bool
- **取值范围**: true | false
- **描述**: 是否独占节点。若为 true，则作业独占整个节点
- **默认值**: `false`

##### scheduler.scf.resources.envs

- **类型**: dict
- **取值范围**: 环境变量字典
- **描述**: 环境变量设置。例如 `{"PYSCF_MAX_MEMORY": 8000, "OMP_NUM_THREADS": 4}`
- **默认值**: `{"PYSCF_MAX_MEMORY": 8000}`

#### 其他

##### scheduler.scf.sub_res

- **类型**: dict | null
- **取值范围**: 资源配置字典或 null
- **描述**: 子任务资源配置。用于并行执行时为每个子任务分配资源
- **默认值**: `null`

##### scheduler.scf.python

- **类型**: string
- **取值范围**: Python 解释器路径
- **描述**: Python 解释器路径。用于执行 Python 脚本
- **默认值**: `"python"`

### 9.2 训练调度器 (scheduler.train)

#### 调度器配置 (scheduler.train.dispatcher)

##### scheduler.train.dispatcher.context

- **类型**: string
- **取值范围**: `"local"` | `"ssh"`
- **描述**: 执行环境（同 SCF）
- **默认值**: `"local"`

##### scheduler.train.dispatcher.batch

- **类型**: string
- **取值范围**: `"shell"` | `"slurm"` | `"pbs"` | `"lsf"`
- **描述**: 调度系统类型（同 SCF）
- **默认值**: `"shell"`

##### scheduler.train.dispatcher.remote_profile

- **类型**: dict | null
- **取值范围**: SSH 配置字典或 null
- **描述**: SSH 远程连接配置（同 SCF）
- **默认值**: `null`

#### 资源配置 (scheduler.train.resources)

##### scheduler.train.resources.cpus_per_task

- **类型**: int
- **取值范围**: 正整数
- **描述**: 每个训练任务使用的 CPU 核心数
- **默认值**: `4`

##### scheduler.train.resources.numb_gpu

- **类型**: int
- **取值范围**: 非负整数
- **描述**: GPU 数量。训练任务使用的 GPU 数
- **默认值**: `0`

##### scheduler.train.resources.mem_limit

- **类型**: int
- **取值范围**: 正整数
- **描述**: 内存限制（单位：GB）
- **默认值**: `8`

##### scheduler.train.resources.time_limit

- **类型**: string
- **取值范围**: 时间格式字符串
- **描述**: 训练作业时间限制
- **默认值**: `"24:00:00"`

##### scheduler.train.resources.envs

- **类型**: dict
- **取值范围**: 环境变量字典
- **描述**: 训练环境变量设置
- **默认值**: `{}`

#### 其他

##### scheduler.train.python

- **类型**: string
- **取值范围**: Python 解释器路径
- **描述**: Python 解释器路径
- **默认值**: `"python"`

### 9.3 初始调度器

#### scheduler.init_scf

- **类型**: dict | null
- **取值范围**: SCF 调度器配置字典或 null
- **描述**: 初始迭代的 SCF 调度器配置。若为 null，则继承 `scheduler.scf` 的配置
- **默认值**: `null`（继承 `scheduler.scf`）
- **注意**: 此参数支持参数继承

#### scheduler.init_train

- **类型**: dict | null
- **取值范围**: 训练调度器配置字典或 null
- **描述**: 初始迭代的训练调度器配置。若为 null，则继承 `scheduler.train` 的配置
- **默认值**: `null`（继承 `scheduler.train`）
- **注意**: 此参数支持参数继承

---

## 10. 统计参数 (stats)

### stats.with_conv

- **类型**: bool
- **取值范围**: true | false
- **描述**: 是否显示收敛统计信息
- **默认值**: `true`

### stats.with_e

- **类型**: bool
- **取值范围**: true | false
- **描述**: 是否显示能量统计信息
- **默认值**: `true`

### stats.with_f

- **类型**: bool
- **取值范围**: true | false
- **描述**: 是否显示力统计信息
- **默认值**: `true`

### stats.with_s

- **类型**: bool
- **取值范围**: true | false
- **描述**: 是否显示应力统计信息
- **默认值**: `true`

### stats.with_o

- **类型**: bool
- **取值范围**: true | false
- **描述**: 是否显示轨道统计信息
- **默认值**: `true`

### stats.e_name

- **类型**: string
- **取值范围**: 文件名
- **描述**: 能量文件名，用于读取能量数据进行统计
- **默认值**: `"e_tot"`

### stats.f_name

- **类型**: string
- **取值范围**: 文件名
- **描述**: 力文件名，用于读取力数据进行统计
- **默认值**: `"f_tot"`

### stats.s_name

- **类型**: string
- **取值范围**: 文件名
- **描述**: 应力文件名，用于读取应力数据进行统计
- **默认值**: `"s_tot"`

### stats.o_name

- **类型**: string
- **取值范围**: 文件名
- **描述**: 轨道文件名，用于读取轨道数据进行统计
- **默认值**: `"o_tot"`

### stats.test_systems

- **类型**: list[string] | null
- **取值范围**: 系统目录路径列表或 null
- **描述**: 测试系统路径列表。若为 null，则仅统计训练系统
- **默认值**: `null`

### stats.test_dump_dir

- **类型**: string | null
- **取值范围**: 有效的目录路径或 null
- **描述**: 测试系统输出目录。若为 null，则使用与训练系统相同的输出目录
- **默认值**: `null`

---

## 附录：参数继承规则

### Init 参数继承

DeePKS-kit 支持参数继承机制，允许初始迭代参数自动继承非初始参数的值，减少配置冗余。

#### 继承关系

1. **训练参数继承**
   - `iterate.init.train` 默认继承 `train` 的所有参数
   - 用户显式设置的参数会覆盖继承值

2. **SCF 参数继承**
   - `iterate.init.scf` 默认继承 `scf` 的所有参数
   - 用户显式设置的参数会覆盖继承值

3. **调度器参数继承**
   - `scheduler.init_scf` 默认继承 `scheduler.scf` 的所有参数
   - `scheduler.init_train` 默认继承 `scheduler.train` 的所有参数

#### 继承规则

- **深度合并**：嵌套字典递归合并，而非简单覆盖
- **显式优先**：用户显式设置的值优先于继承值
- **null 保留**：显式设置为 null 会覆盖继承值

#### 继承示例

```yaml
# 用户配置
train:
  n_epoch: 5000
  start_lr: 0.001
  loss_weights:
    energy: 1.0
    force: 1.0
    stress: 0.5

iterate:
  n_iter: 5
  init:
    train:
      # 仅覆盖需要修改的参数
      loss_weights:
        force: 0.0  # 初始轮不训练力

# 实际生效的配置
iterate:
  init:
    train:
      n_epoch: 5000        # 继承
      start_lr: 0.001      # 继承
      loss_weights:
        energy: 1.0        # 继承
        force: 0.0         # 覆盖
        stress: 0.5        # 继承
```

#### 特殊情况

某些参数仅存在于 init 或非 init 配置中：

- **仅 init 包含**：`iterate.init.use_model`, `iterate.init.model_file`
  - 这些参数不需要继承，仅在初始迭代中使用

- **仅非 init 包含**：某些训练或 SCF 参数可能在初始迭代中不适用
  - 文档中会明确标注

---

## 附录：后端选择

### SCF 后端架构

DeePKS-kit 支持两种 SCF 后端：PySCF 和 ABACUS。两者是并列关系，互斥使用。

#### 后端选择器

通过 `scf.scf_soft` 参数选择后端：

```yaml
scf:
  scf_soft: "pyscf"  # 或 "abacus"
```

#### 后端特点

| 特性 | PySCF | ABACUS |
|------|-------|--------|
| 适用体系 | 分子、小周期体系 | 大规模周期体系 |
| 基组类型 | 高斯基组 | 数值原子轨道、平面波 |
| 并行能力 | 有限 | 强大（MPI） |
| 内存需求 | 较高 | 可控 |
| 安装难度 | 简单（pip） | 需要编译 |

#### 后端配置

##### PySCF 后端示例

```yaml
scf:
  scf_soft: "pyscf"
  dump_dir: "scf_results"
  dump_fields: ["atom", "e_base", "e_tot", "dm_eig", "conv"]

  pyscf:
    basis: "ccpvdz"
    xc: "HF"
    conv_tol: 1e-7
    max_cycle: 50
    mol_args:
      charge: 0
      spin: 0
      unit: "Angstrom"
    grids:
      level: 3
```

##### ABACUS 后端示例

```yaml
scf:
  scf_soft: "abacus"
  dump_dir: "scf_results"
  dump_fields: ["atom", "e_base", "e_tot", "dm_eig", "conv"]

  abacus:
    abacus_path: "/usr/local/bin/ABACUS.mpi"
    run_cmd: "mpirun -np 8"
    orb_files: ["Si_gga_7au_100Ry_2s2p1d.orb"]
    pp_files: ["Si_ONCV_PBE-1.0.upf"]
    proj_file: ["jle_5au_15.orb"]
    ecutwfc: 100
    scf_thr: 1e-7
    scf_nmax: 100
    dft_functional: "pbe"
    basis_type: "lcao"
    kspacing: 0.5
```

#### 后端切换

切换后端只需修改 `scf_soft` 参数和对应的后端配置：

```yaml
# 从 PySCF 切换到 ABACUS
scf:
  scf_soft: "abacus"  # 修改这里
  # 其他通用参数保持不变
  dump_dir: "scf_results"

  # 注释掉 PySCF 配置
  # pyscf:
  #   basis: "ccpvdz"
  #   ...

  # 启用 ABACUS 配置
  abacus:
    abacus_path: "/usr/local/bin/ABACUS.mpi"
    # ...
```

---

## 附录：完整配置示例

### 示例 1：简单训练任务

```yaml
type: train

systems:
  train: ["data/train/sys1", "data/train/sys2"]
  test: ["data/test/sys1"]

model:
  checkpoint: "model.pth"
  proj_basis: "jle.orb"
  architecture:
    hidden_sizes: [100, 100, 100]
    actv_fn: "gelu"
    use_resnet: true

data:
  batch_size: 16

preprocess:
  preshift: true
  prefit: true
  prefit_ridge: 10.0

train:
  n_epoch: 5000
  start_lr: 0.001
  decay_rate: 0.96
  decay_steps: 100
  display_epoch: 100
  loss_weights:
    energy: 1.0
    force: 1.0
```

### 示例 2：PySCF SCF 计算

```yaml
type: scf

systems:
  train: ["data/molecules/H2O", "data/molecules/CH4"]

model:
  file: "trained_model.pth"
  proj_basis: "ccpvdz"

scf:
  scf_soft: "pyscf"
  dump_dir: "scf_results"
  dump_fields: ["atom", "e_base", "e_tot", "dm_eig", "conv"]

  pyscf:
    basis: "ccpvdz"
    xc: "HF"
    conv_tol: 1e-7
    max_cycle: 50
    mol_args:
      charge: 0
      spin: 0
```

### 示例 3：ABACUS 迭代训练

```yaml
type: iterate

common:
  device: "cuda:0"
  verbose: 1

systems:
  train: ["data/Si/train"]
  test: ["data/Si/test"]

model:
  checkpoint: "model.pth"
  proj_basis: "jle_5au_15.orb"
  architecture:
    hidden_sizes: [100, 100, 100]
    actv_fn: "mygelu"
    use_resnet: true
    output_scale: 100
  embedding:
    type: "thermal"
    init_beta: 5.0

data:
  batch_size: 4
  group_batch: 1
  orb_list: ["Si_gga_7au_100Ry_2s2p1d.orb"]
  alpha_list: ["jle_5au_15.orb"]

preprocess:
  preshift: true
  prescale: false
  prefit: true
  prefit_ridge: 10.0

train:
  n_epoch: 5000
  start_lr: 0.0005
  decay_rate: 0.8
  decay_steps: 1000
  display_epoch: 100
  loss_weights:
    energy: 1.0
    force: 1.0
    v_delta_r: 10000

scf:
  scf_soft: "abacus"
  dump_dir: "scf_results"

  abacus:
    abacus_path: "/usr/local/bin/ABACUS.mpi"
    run_cmd: "mpirun -np 8"
    orb_files: ["Si_gga_7au_100Ry_2s2p1d.orb"]
    pp_files: ["Si_ONCV_PBE-1.0.upf"]
    proj_file: ["jle_5au_15.orb"]
    ecutwfc: 100
    scf_thr: 1e-7
    scf_nmax: 100
    dft_functional: "pbe"
    basis_type: "lcao"
    gamma_only: 1

iterate:
  n_iter: 5
  share_folder: "share"
  init:
    train:
      n_epoch: 2000
      loss_weights:
        energy: 1.0
        force: 0.0  # 初始轮仅能量

scheduler:
  scf:
    sub_size: 1
    group_size: 1
    dispatcher:
      context: "local"
      batch: "shell"
    resources:
      cpus_per_task: 8
      mem_limit: 16
      time_limit: "24:00:00"

  train:
    dispatcher:
      context: "local"
      batch: "shell"
    resources:
      cpus_per_task: 4
      numb_gpu: 1
      mem_limit: 16
```

### 示例 4：统计分析

```yaml
type: stats

systems:
  train: ["iter5/scf_results"]
  test: ["iter5/test_results"]

stats:
  with_conv: true
  with_e: true
  with_f: true
  with_s: false
  e_name: "e_tot"
  f_name: "f_tot"
```

---

## 附录：参数快速查找

### 按功能分类

#### 模型架构
- `model.architecture.hidden_sizes`
- `model.architecture.actv_fn`
- `model.architecture.use_resnet`
- `model.architecture.layer_norm`
- `model.architecture.output_scale`

#### 学习率控制
- `train.start_lr`
- `train.stop_lr`
- `train.decay_rate`
- `train.decay_steps`
- `train.decay_rate_iter`

#### 损失函数
- `train.loss_weights.*`
- `train.loss_config.*`
- `train.energy_per_atom`
- `train.grad_penalty`

#### 数据处理
- `data.batch_size`
- `data.group_batch`
- `data.conv_filter`
- `preprocess.preshift`
- `preprocess.prescale`
- `preprocess.prefit`

#### SCF 控制
- `scf.scf_soft`
- `scf.pyscf.conv_tol`
- `scf.pyscf.max_cycle`
- `scf.abacus.scf_thr`
- `scf.abacus.scf_nmax`

#### 计算资源
- `scheduler.*.resources.cpus_per_task`
- `scheduler.*.resources.numb_gpu`
- `scheduler.*.resources.mem_limit`
- `scheduler.*.resources.time_limit`

### 常用参数组合

#### 快速训练（小数据集）
```yaml
data.batch_size: 32
train.n_epoch: 1000
train.start_lr: 0.001
train.display_epoch: 50
```

#### 精细训练（大数据集）
```yaml
data.batch_size: 16
train.n_epoch: 10000
train.start_lr: 0.0005
train.decay_steps: 500
train.display_epoch: 100
```

#### 力场训练
```yaml
train.loss_weights:
  energy: 1.0
  force: 10.0
  stress: 0.1
```

#### 快速 SCF（测试）
```yaml
scf.pyscf.conv_tol: 1e-5
scf.pyscf.max_cycle: 30
```

#### 精确 SCF（生产）
```yaml
scf.pyscf.conv_tol: 1e-9
scf.pyscf.max_cycle: 100
```

---

**文档版本**：1.0
**创建日期**：2026-03-18
**最后更新**：2026-03-18
**作者**：Claude (claude-sonnet-4-6)

**相关文档**：
- [重构计划 V3](INPUT_REFACTOR_PLAN_V3.md)
- [完整参数列表](COMPLETE_PARAMETERS.md)
- [物理逻辑映射](skills/deepks-physics-logic-skill.md)
