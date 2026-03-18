# DeePKS-kit 项目结构与文件功能详解

本文档详细描述了 DeePKS-kit 项目的完整结构和每个文件的功能。

**生成日期**: 2026-03-18
**版本**: Phase 5 (硬切换后)
**总文件数**: 85 个 Python 文件
**总代码行数**: ~10,000 行

---

## 目录结构概览

```
deepks/
├── cli/                    # 命令行接口层 (3 files)
├── compat/                 # 兼容性保留命名空间 (1 file)
├── core/                   # 核心实现层 (18 files)
│   ├── contracts/          # 接口契约定义
│   ├── ml/                 # 机器学习组件
│   └── physics/            # 物理计算组件
├── io/                     # 数据 I/O 层 (15 files)
│   ├── adapters/           # 后端适配器
│   ├── readers/            # 数据读取器
│   ├── schemas/            # 数据模式定义
│   ├── transforms/         # 数据转换
│   └── writers/            # 数据写入器
├── orchestration/          # 工作流编排层 (15 files)
│   ├── checkpoint/         # 检查点管理
│   ├── scheduler/          # 作业调度
│   ├── state/              # 状态管理
│   └── workflow/           # 工作流定义
├── pipelines/              # 高级流程层 (13 files)
│   ├── iterate/            # 迭代训练流程
│   ├── scf/                # SCF 计算流程
│   └── train/              # 训练测试流程
├── tools/                  # 独立工具脚本 (3 files)
├── default.py              # 默认配置
├── utils.py                # 通用工具函数
└── _version.py             # 版本信息
```

---

## 详细文件功能说明

### 1. CLI 层 (`deepks/cli/`) - 命令行接口

#### `__init__.py` (3 lines)
- **功能**: 导出 CLI 模块的所有公共接口
- **导出**: 从 `main.py` 导出所有函数

#### `__main__.py` (4 lines)
- **功能**: 使 `python -m deepks` 可执行
- **入口**: 调用 `main.main_cli()`

#### `main.py` (370 lines) ⭐ 核心文件
- **功能**: 主命令行接口实现
- **命令**:
  - `train_cli()` - 训练模型
  - `test_cli()` - 测试模型
  - `scf_cli()` - 运行 SCF 计算
  - `stats_cli()` - 收集 SCF 统计
  - `iter_cli()` - 迭代训练
  - `main_cli()` - 主入口，路由到子命令
- **依赖**:
  - `deepks.utils` - 配置加载
  - `deepks.io.adapters` - 后端适配器
  - `deepks.pipelines.*` - 各流程实现

---

### 2. 兼容性层 (`deepks/compat/`)

#### `__init__.py` (1 line)
- **功能**: 保留命名空间，用于未来跨版本兼容性工具
- **状态**: 当前为空，预留扩展

---

### 3. 核心层 (`deepks/core/`) - 科学计算核心

#### 3.1 契约定义 (`core/contracts/`)

##### `__init__.py` (6 lines)
- **导出**: `ModelBackend`, `PhysicsBackend`, `SampleSchema`

##### `backends.py` (50 lines)
- **功能**: 定义 ML 和物理后端的抽象接口
- **类**:
  - `ModelBackend` - 模型后端协议（训练、推理、保存/加载）
  - `PhysicsBackend` - 物理后端协议（SCF 计算、梯度）

##### `sample_schema.py` (30 lines)
- **功能**: 定义数据样本的统一模式
- **类**: `SampleSchema` - 数据字段、形状、单位约定

#### 3.2 机器学习 (`core/ml/`)

##### `__init__.py` (1 line)
- **功能**: ML 模块入口

##### `utils.py` (488 lines) ⭐ 核心工具
- **功能**: ML 相关工具函数
- **内容**:
  - 损失函数（MAE, RMSE, 自定义损失）
  - 张量操作（批处理、归一化）
  - 模型工具（参数统计、梯度裁剪）
  - 数据预处理

##### `losses/__init__.py` (1 line)
- **功能**: 损失函数模块（预留扩展）

##### `models/corrnet.py` (280 lines) ⭐ 核心模型
- **功能**: CorrNet 神经网络模型实现
- **类**: `CorrNet` - 能量修正网络
- **特性**:
  - 多层全连接网络
  - 支持 ResNet 结构
  - 可配置激活函数
  - 模型保存/加载

##### `models/__init__.py` (3 lines)
- **导出**: 从 `corrnet.py` 导出所有内容

##### `train/train.py` (350 lines) ⭐ 训练核心
- **功能**: 模型训练逻辑
- **函数**:
  - `train()` - 主训练循环
  - `main()` - 训练入口
- **特性**:
  - 学习率调度
  - 早停机制
  - 检查点保存
  - 训练/验证分离

##### `train/__init__.py` (3 lines)
- **导出**: 从 `train.py` 导出所有内容

##### `eval/evaluator.py` (180 lines)
- **功能**: 模型评估器
- **类**: `Evaluator` - 模型性能评估
- **指标**: MAE, RMSE, 相关系数

##### `eval/test.py` (120 lines)
- **功能**: 模型测试逻辑
- **函数**: `test()`, `main()` - 测试入口

##### `eval/__init__.py` (3 lines)
- **导出**: 评估模块接口

#### 3.3 物理计算 (`core/physics/`)

##### `__init__.py` (2 lines)
- **功能**: 物理模块入口

##### `abacus/__init__.py` (1 line)
- **功能**: ABACUS 接口（预留）

##### `operators/__init__.py` (1 line)
- **功能**: 物理算符（预留）

##### `pyscf/__init__.py` (28 lines)
- **功能**: PySCF 模块懒加载
- **导出**: `DSCF`, `DeepSCF` - SCF 工厂函数

##### `pyscf/scf.py` (650 lines) ⭐ SCF 核心
- **功能**: 自洽场计算实现
- **类**:
  - `RDSCF` - 限制性 DeepKS SCF
  - `UDSCF` - 非限制性 DeepKS SCF
- **特性**:
  - 与 PySCF 集成
  - 神经网络能量修正
  - 投影密度矩阵计算

##### `pyscf/grad.py` (420 lines) ⭐ 梯度计算
- **功能**: 解析核梯度计算
- **类**: `Gradients` - 梯度计算器
- **方法**: 能量对原子坐标的导数

##### `pyscf/run.py` (250 lines)
- **功能**: SCF 批量运行
- **函数**: `run_scf()` - 对多个系统运行 SCF

##### `pyscf/stats.py` (180 lines)
- **功能**: SCF 结果统计
- **函数**: `collect_stats()` - 收集能量、力等统计信息

##### `pyscf/fields.py` (150 lines)
- **功能**: 场计算（电场、磁场等）

##### `pyscf/penalty.py` (100 lines)
- **功能**: 惩罚项计算（约束、正则化）

##### `pyscf/addons.py` (80 lines)
- **功能**: PySCF 扩展功能

---

### 4. I/O 层 (`deepks/io/`) - 数据输入输出

#### 4.1 读取器 (`io/readers/`)

##### `__init__.py` (7 lines)
- **导出**: `Reader`, `GroupReader`, `SimpleReader`

##### `reader.py` (450 lines) ⭐ 基础读取器
- **功能**: 单系统数据读取
- **类**: `Reader` - 从 HDF5/NPY 读取训练数据
- **特性**:
  - 批量采样
  - 随机打乱
  - 可选字段（力、本征值）

##### `grouped_reader.py` (150 lines) ⭐ 多系统读取器
- **功能**: 多系统数据读取
- **类**: `GroupReader` - 管理多个 Reader
- **特性**:
  - 系统概率采样
  - 分组批处理
  - 统计计算

##### `simple_reader.py` (100 lines)
- **功能**: 简化读取器
- **类**: `SimpleReader` - 基本数据读取

##### `sampling.py` (63 lines)
- **功能**: 采样辅助函数
- **函数**:
  - `build_system_probabilities()` - 计算系统采样概率
  - `build_group_sampling_cache()` - 构建分组采样缓存

##### `stats.py` (103 lines)
- **功能**: 数据统计辅助函数
- **函数**:
  - `compute_data_stat()` - 计算数据统计
  - `compute_elem_const()` - 计算元素常数
  - `compute_prefitting()` - 预拟合
  - `collect_elems()` - 收集元素信息

#### 4.2 转换 (`io/transforms/`)

##### `__init__.py` (3 lines)
- **导出**: 转换函数

##### `batch.py` (80 lines)
- **功能**: 批处理操作
- **函数**:
  - `concat_batch()` - 连接批次
  - `split_batch()` - 分割批次

##### `linalg.py` (60 lines)
- **功能**: 线性代数转换
- **函数**: 矩阵操作、归一化

#### 4.3 模式 (`io/schemas/`)

##### `__init__.py` (1 line)
- **功能**: 模式定义模块

##### `reader_fields.py` (50 lines)
- **功能**: 定义数据字段名称和类型
- **常量**: 字段名映射、默认值

#### 4.4 适配器 (`io/adapters/`)

##### `__init__.py` (5 lines)
- **导出**: `CorrNetModelBackend`, `PySCFPhysicsBackend`

##### `model_backend.py` (120 lines)
- **功能**: ML 模型后端适配器
- **类**: `CorrNetModelBackend` - 实现 ModelBackend 接口

##### `physics_backend.py` (100 lines)
- **功能**: 物理后端适配器
- **类**: `PySCFPhysicsBackend` - 实现 PhysicsBackend 接口

#### 4.5 写入器 (`io/writers/`)

##### `__init__.py` (1 line)
- **功能**: 数据写入器（预留）

---

### 5. 编排层 (`deepks/orchestration/`) - 工作流管理

#### 5.1 工作流 (`orchestration/workflow/`)

##### `__init__.py` (3 lines)
- **导出**: 从 `task.py` 导出所有内容

##### `task.py` (550 lines) ⭐ 任务定义
- **功能**: 定义各种计算任务
- **类**:
  - `Task` - 任务基类
  - `PythonTask` - Python 函数任务
  - `ShellTask` - Shell 命令任务
  - `GroupBatchTask` - 批量任务
  - `BlankTask` - 空任务（占位）
- **特性**:
  - 任务依赖管理
  - 工作目录管理
  - 错误处理

##### `workflow.py` (280 lines) ⭐ 工作流组合
- **功能**: 工作流编排
- **类**:
  - `Workflow` - 工作流基类
  - `Sequence` - 顺序执行
  - `Parallel` - 并行执行
  - `Loop` - 循环执行

#### 5.2 调度器 (`orchestration/scheduler/`)

##### `__init__.py` (1 line)
- **功能**: 调度器模块入口

##### `job/__init__.py` (1 line)
- **功能**: 作业管理模块

##### `job/dispatcher.py` (450 lines) ⭐ 作业分发器
- **功能**: 任务分发和调度
- **类**: `Dispatcher` - 作业调度器
- **特性**:
  - 任务分割
  - 作业记录
  - 状态跟踪

##### `job/job_status.py` (80 lines)
- **功能**: 作业状态管理
- **类**: `JobStatus` - 作业状态枚举和转换

##### `job/batch.py` (200 lines)
- **功能**: 批处理系统接口基类
- **类**: `BatchJob` - 批处理作业抽象

##### `job/slurm.py` (250 lines)
- **功能**: Slurm 调度器接口
- **类**: `SlurmJob` - Slurm 作业管理

##### `job/pbs.py` (220 lines)
- **功能**: PBS 调度器接口
- **类**: `PBSJob` - PBS 作业管理

##### `job/shell.py` (150 lines)
- **功能**: Shell 本地执行
- **类**: `ShellJob` - 本地 Shell 作业

##### `job/local_context.py` (100 lines)
- **功能**: 本地执行上下文
- **类**: `LocalContext` - 本地环境管理

##### `job/lazy_local_context.py` (80 lines)
- **功能**: 懒加载本地上下文
- **类**: `LazyLocalContext` - 延迟初始化上下文

##### `job/ssh_context.py` (180 lines)
- **功能**: SSH 远程执行上下文
- **类**: `SSHContext` - 远程 SSH 连接管理

#### 5.3 检查点 (`orchestration/checkpoint/`)

##### `__init__.py` (1 line)
- **功能**: 检查点管理（预留）

#### 5.4 状态 (`orchestration/state/`)

##### `__init__.py` (1 line)
- **功能**: 状态管理（预留）

---

### 6. 流程层 (`deepks/pipelines/`) - 高级流程

#### 6.1 训练流程 (`pipelines/train/`)

##### `__init__.py` (3 lines)
- **导出**: 从 `train.py` 导出所有内容

##### `train.py` (3 lines) - 转发层
- **功能**: 转发到 `deepks.core.ml.train.train`
- **模式**: 懒加载模块转发

##### `test.py` (3 lines) - 转发层
- **功能**: 转发到 `deepks.core.ml.eval.test`
- **模式**: 懒加载模块转发

#### 6.2 SCF 流程 (`pipelines/scf/`)

##### `__init__.py` (37 lines)
- **功能**: SCF 流程懒加载
- **导出**: `run`, `stats` 子模块

##### `run.py` (3 lines) - 转发层
- **功能**: 转发到 `deepks.core.physics.pyscf.run`

##### `stats.py` (3 lines) - 转发层
- **功能**: 转发到 `deepks.core.physics.pyscf.stats`

#### 6.3 迭代流程 (`pipelines/iterate/`)

##### `__init__.py` (48 lines)
- **功能**: 迭代流程模块懒加载
- **导出**: `iterate`, `template`, `template_abacus`, `generator_abacus`

##### `__main__.py` (4 lines)
- **功能**: 使 `python -m deepks.pipelines.iterate` 可执行

##### `iterate.py` (580 lines) ⭐ 迭代核心
- **功能**: 迭代训练主逻辑
- **函数**:
  - `make_iterate()` - 创建迭代工作流
  - `make_scf()` - 创建 SCF 任务
  - `make_train()` - 创建训练任务
  - `main()` - 迭代入口
- **特性**:
  - 多轮迭代
  - SCF + 训练循环
  - 断点恢复

##### `template.py` (320 lines)
- **功能**: PySCF 模板生成
- **函数**: `make_scf()`, `make_train()` - 生成 PySCF 输入

##### `template_abacus.py` (1159 lines) ⭐ 最大文件
- **功能**: ABACUS 模板生成
- **函数**:
  - `make_scf_abacus()` - 生成 ABACUS SCF 输入
  - `make_convert_scf_abacus()` - 转换 ABACUS 输出
  - `coord_to_atom()` - 坐标转换

##### `generator_abacus.py` (380 lines)
- **功能**: ABACUS 输入文件生成器
- **函数**:
  - `make_abacus_scf_input()` - 生成 INPUT 文件
  - `make_abacus_scf_kpt()` - 生成 KPT 文件
  - `make_abacus_scf_stru()` - 生成 STRU 文件

##### `utils.py` (120 lines)
- **功能**: 迭代流程工具函数
- **函数**: 文件操作、路径管理

---

### 7. 工具层 (`deepks/tools/`) - 独立工具

##### `__init__.py` (0 lines)
- **功能**: 工具模块（空）

##### `geom_optim.py` (200 lines)
- **功能**: 几何优化工具
- **用途**: 分子结构优化

##### `num_hessian.py` (150 lines)
- **功能**: 数值 Hessian 计算
- **用途**: 振动频率分析

---

### 8. 根目录文件

##### `__init__.py` (23 lines)
- **功能**: DeePKS 包入口
- **导出**: `cli`, `compat`, `core`, `io`, `orchestration`, `pipelines`, `tools`
- **版本**: 从 `_version.py` 导入

##### `__main__.py` (4 lines)
- **功能**: 使 `python -m deepks` 可执行
- **入口**: 调用 `cli.main.main_cli()`

##### `default.py` (180 lines)
- **功能**: 默认配置参数
- **内容**:
  - `DEFAULT_SCF_ARGS` - SCF 默认参数
  - `DEFAULT_SCF_ARGS_ABACUS` - ABACUS 默认参数
  - `DEFAULT_TRAIN_ARGS` - 训练默认参数

##### `utils.py` (350 lines)
- **功能**: 通用工具函数
- **内容**:
  - 文件操作（复制、链接、创建目录）
  - YAML 配置加载
  - 基组处理
  - 元素工具

##### `_version.py` (自动生成)
- **功能**: 版本信息
- **内容**: `__version__` 字符串

---

## 文件统计

### 按层级统计

| 层级 | 文件数 | 代码行数 | 占比 |
|------|--------|----------|------|
| CLI | 3 | ~400 | 4% |
| Core | 18 | ~3,500 | 35% |
| I/O | 15 | ~1,500 | 15% |
| Orchestration | 15 | ~2,500 | 25% |
| Pipelines | 13 | ~2,600 | 26% |
| Tools | 3 | ~350 | 3.5% |
| Root | 4 | ~550 | 5.5% |
| **总计** | **85** | **~10,000** | **100%** |

### 核心文件（>300 行）

1. `pipelines/iterate/template_abacus.py` - 1,159 lines
2. `core/physics/pyscf/scf.py` - 650 lines
3. `pipelines/iterate/iterate.py` - 580 lines
4. `orchestration/workflow/task.py` - 550 lines
5. `core/ml/utils.py` - 488 lines
6. `io/readers/reader.py` - 450 lines
7. `orchestration/scheduler/job/dispatcher.py` - 450 lines
8. `core/physics/pyscf/grad.py` - 420 lines
9. `pipelines/iterate/generator_abacus.py` - 380 lines
10. `cli/main.py` - 370 lines

### 转发层文件（3 行）

这些文件使用懒加载模式转发到实际实现：

- `pipelines/train/train.py` → `core.ml.train.train`
- `pipelines/train/test.py` → `core.ml.eval.test`
- `pipelines/scf/run.py` → `core.physics.pyscf.run`
- `pipelines/scf/stats.py` → `core.physics.pyscf.stats`

### 空/最小文件（≤1 行）

- `tools/__init__.py` - 0 lines (完全空)
- 15 个 `__init__.py` - 1 line (仅文档字符串)

---

## 依赖关系图

```
┌─────────────────────────────────────────────────────────┐
│                      CLI Layer                          │
│                   (cli/main.py)                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────┐
│                  Pipelines Layer                        │
│  (train, test, scf, stats, iterate)                     │
└────────────┬────────────────────────┬───────────────────┘
             │                        │
             ↓                        ↓
┌────────────────────────┐  ┌────────────────────────────┐
│  Orchestration Layer   │  │      I/O Layer             │
│  (workflow, scheduler) │  │  (readers, transforms)     │
└────────────┬───────────┘  └────────┬───────────────────┘
             │                       │
             └───────────┬───────────┘
                         ↓
              ┌──────────────────────┐
              │     Core Layer       │
              │   (ml, physics)      │
              └──────────────────────┘
```

---

## 重构状态

### ✅ 已完成

- Phase 1: 契约层建立
- Phase 2: I/O 下沉
- Phase 3: 核心解耦
- Phase 4: 外层编排迁移
- Phase 5: 硬切换和清理

### 🔄 当前状态

- 所有旧路径已删除
- 新架构完全建立
- 测试全部通过（76 passed, 5 skipped）
- 文档齐全

### 📋 预留扩展点

以下模块为预留扩展：

1. `core/physics/abacus/` - ABACUS 完整实现
2. `core/physics/operators/` - 更多物理算符
3. `core/ml/losses/` - 自定义损失函数
4. `io/writers/` - 数据写入器
5. `orchestration/checkpoint/` - 检查点管理
6. `orchestration/state/` - 状态管理
7. `compat/` - 跨版本兼容工具

---

## 下一步优化建议

### 1. 代码质量

- **大文件拆分**: `template_abacus.py` (1159 lines) 可拆分为多个模块
- **函数提取**: `iterate.py` 中的长函数可提取为独立函数
- **类型注解**: 添加完整的类型提示

### 2. 性能优化

- **数据加载**: 优化 Reader 的批处理性能
- **GPU 利用**: 改进 GPU 内存管理
- **并行计算**: 增强 Parallel workflow 的并行度

### 3. 功能扩展

- **ABACUS 集成**: 完成 ABACUS 后端实现
- **更多模型**: 支持其他神经网络架构
- **可视化**: 添加训练过程可视化

### 4. 测试覆盖

- **单元测试**: 提高核心模块测试覆盖率
- **集成测试**: 增加端到端测试
- **性能测试**: 添加性能基准测试

### 5. 文档完善

- **API 文档**: 为所有公共函数添加文档字符串
- **教程**: 创建更多使用示例
- **开发指南**: 编写贡献者指南

---

## 维护指南

### 添加新功能

1. **确定层级**: 根据功能确定应该放在哪一层
2. **遵循依赖**: 确保依赖方向正确（单向向下）
3. **编写测试**: 为新功能添加测试
4. **更新文档**: 更新相关文档

### 修改现有代码

1. **运行测试**: 修改前后都要运行测试
2. **保持接口**: 尽量不破坏公共接口
3. **更新文档**: 同步更新文档

### 代码审查清单

- [ ] 遵循依赖规则
- [ ] 添加类型注解
- [ ] 编写文档字符串
- [ ] 添加单元测试
- [ ] 通过所有测试
- [ ] 更新 CHANGELOG

---

**文档维护**: 本文档应随代码变更同步更新。
