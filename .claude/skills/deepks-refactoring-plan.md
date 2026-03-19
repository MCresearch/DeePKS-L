# DeePKS 架构重构完整计划

## 重构目标

将 DeePKS 从混乱的多层架构重构为清晰的物理流程驱动架构，适合科研软件的特点。

## 核心理念

**科研软件的特点**:
- 物理流程是核心，代码结构应直观反映物理过程
- 工作流编排是关键，而非软件工程的分层抽象
- 易于理解和修改比过度工程化更重要

**物理流程示例（Iterate）**:
```
Iterate 迭代训练
├── SCF 过程（自洽场计算，与软件无关）
│   ├── 准备阶段：构建工作目录、生成输入文件
│   ├── 运行阶段：调用后端软件（PySCF/ABACUS）
│   └── 整合阶段：收集结果、统计分析
└── Train 过程
    ├── 训练阶段：模型训练
    └── 测试阶段：模型评估
```

## 目标架构

```
DeePKS-L/
├── main.py                          # 主入口（根目录）
│
├── deepks/
│   ├── io/                          # I/O 层
│   │   ├── input/                   # 输入配置
│   │   ├── readers/                 # 数据读取
│   │   ├── writers/                 # 数据写入
│   │   ├── transforms/              # 数据转换
│   │   └── schemas/                 # 数据模式
│   │
│   ├── workflows/                   # 工作流层（核心）
│   │   ├── scf/                     # SCF 工作流
│   │   │   ├── workflow.py          # 主工作流
│   │   │   ├── prepare.py           # 准备阶段
│   │   │   ├── execute.py           # 执行阶段
│   │   │   └── collect.py           # 收集阶段
│   │   ├── train/                   # 训练工作流
│   │   ├── iterate/                 # 迭代工作流
│   │   └── stats/                   # 统计工作流
│   │
│   ├── core/                        # 核心实现层
│   │   ├── physics/                 # 物理计算
│   │   │   └── backends/            # 后端实现
│   │   │       ├── base.py          # 后端接口
│   │   │       ├── factory.py       # 后端工厂
│   │   │       ├── pyscf/           # PySCF 后端
│   │   │       └── abacus/          # ABACUS 后端
│   │   └── ml/                      # 机器学习
│   │       ├── models/              # 模型定义
│   │       ├── trainers/            # 训练器
│   │       └── evaluators/          # 评估器
│   │
│   └── orchestration/               # 任务调度层
│       ├── workflow/                # 工作流基础设施
│       ├── scheduler/               # 调度器
│       └── checkpoint/              # 断点续算
```

## 关键概念

### 1. 四层架构

1. **主入口层** (`main.py`): 配置加载 + 命令分发
2. **工作流层** (`workflows/`): 物理流程编排
3. **核心实现层** (`core/`): 具体算法实现
4. **调度层** (`orchestration/`): 资源调度、任务执行

### 2. SCF vs PySCF

- **SCF**: 物理过程（自洽场计算），与软件无关
- **PySCF**: 具体实现软件之一
- **ABACUS**: 另一个具体实现软件

**分离方式**:
- `workflows/scf/` → SCF 物理流程
- `core/physics/backends/pyscf/` → PySCF 实现
- `core/physics/backends/abacus/` → ABACUS 实现

### 3. 三阶段模式

每个物理过程分为三个阶段：

1. **准备阶段** (`prepare.py`): 创建目录、生成输入文件
2. **执行阶段** (`execute.py`): 调用调度器执行计算
3. **收集阶段** (`collect.py`): 解析结果、整合数据

### 4. 工作流 vs 调度

- **工作流层** (`workflows/`): 定义做什么（what to do）
- **调度层** (`orchestration/`): 定义怎么做（how to do）

## 详细重构映射

### 阶段 1: 清理转发文件

| 当前文件 | 操作 | 说明 |
|---------|------|------|
| `cli/main.py` | 移动 | → `main.py`（根目录） |
| `cli/__init__.py` | 删除 | 不再需要 |
| `compat/` | 删除 | 空目录 |
| `pipelines/scf/run.py` | 删除 | 纯转发 |
| `pipelines/train/train.py` | 删除 | 纯转发 |
| `io/adapters/` | 删除 | 已被 factory 替代 |

### 阶段 2: 重组后端实现

| 当前文件 | 目标位置 |
|---------|---------|
| `core/physics/factory.py` | `core/physics/backends/factory.py` |
| `core/physics/pyscf/run.py` | `core/physics/backends/pyscf/runner.py` |
| `core/physics/pyscf/stats.py` | `workflows/stats/pyscf_stats.py` |
| `core/physics/abacus/run.py` | 拆分为 3 个文件： |
| | `→ input_generator.py` |
| | `→ runner.py` |
| | `→ parser.py` |

### 阶段 3: 创建工作流层

| 新建文件 | 来源 | 说明 |
|---------|------|------|
| `workflows/scf/workflow.py` | 新建 | SCF 主工作流 |
| `workflows/scf/prepare.py` | 从 `template.py` 提取 | 准备阶段 |
| `workflows/scf/execute.py` | 新建 | 执行阶段 |
| `workflows/scf/collect.py` | 新建 | 收集阶段 |
| `workflows/train/workflow.py` | 新建 | 训练主工作流 |
| `workflows/iterate/workflow.py` | 从 `iterate.py` 重构 | 迭代工作流 |

### 阶段 4: 重构 pipelines/iterate/

**`iterate.py` 拆分**:
- `make_iterate()` → `workflows/iterate/workflow.py`
- `check_share_folder()` → `workflows/iterate/utils.py`
- `check_arg_dict()` → `io/input/validator.py`
- 默认值常量 → `io/input/defaults.py`

**`template.py` 拆分**:
- `make_scf_task()` → `workflows/scf/prepare.py`
- `make_train_task()` → `workflows/train/prepare.py`
- `make_scf()` → `workflows/scf/workflow.py`
- `make_train()` → `workflows/train/workflow.py`

**`template_abacus.py` 拆分**:
- `make_scf_abacus()` → `workflows/scf/workflow.py`
- ABACUS 相关 → `core/physics/backends/abacus/`

**`generator_abacus.py` 移动**:
- 整个文件 → `core/physics/backends/abacus/input_generator.py`

## 重构步骤

### 步骤 1: 创建主入口（1天）

**目标**: 创建 `main.py`，简化 CLI 逻辑

**任务**:
1. 创建 `main.py` 在根目录
2. 实现配置加载 + 命令分发
3. 更新 `io/input/dispatcher.py` 调用工作流
4. 测试基本功能

### 步骤 2: 创建 SCF 工作流（3天）

**目标**: 建立工作流层的模板

**任务**:
1. 创建 `workflows/scf/` 目录结构
2. 实现 `workflow.py`（主流程）
3. 实现 `prepare.py`（准备阶段）
4. 实现 `execute.py`（执行阶段）
5. 实现 `collect.py`（收集阶段）
6. 从 `template.py` 迁移相关逻辑
7. 测试 SCF 工作流

### 步骤 3: 重组后端实现（2天）

**目标**: 统一后端接口

**任务**:
1. 创建 `core/physics/backends/base.py`（接口定义）
2. 移动 `factory.py` 到 `backends/`
3. 重组 PySCF 后端
4. 拆分 ABACUS 后端为 3 个文件
5. 移动 `generator_abacus.py`
6. 测试后端接口

### 步骤 4: 创建训练工作流（2天）

**目标**: 完成训练流程重构

**任务**:
1. 创建 `workflows/train/` 目录
2. 实现训练工作流
3. 从 `core/ml/` 迁移逻辑
4. 测试训练工作流

### 步骤 5: 重构迭代工作流（3天）

**目标**: 完成最复杂的迭代流程

**任务**:
1. 创建 `workflows/iterate/workflow.py`
2. 从 `pipelines/iterate/iterate.py` 迁移逻辑
3. 简化为工作流编排
4. 测试迭代工作流

### 步骤 6: 删除旧文件（1天）

**目标**: 清理代码库

**任务**:
1. 删除 `cli/`
2. 删除 `compat/`
3. 删除 `pipelines/`
4. 删除 `io/adapters/`
5. 验证所有功能正常

### 步骤 7: 更新测试（2天）

**目标**: 确保测试覆盖

**任务**:
1. 更新导入路径
2. 适配新接口
3. 确保所有测试通过

## 关键解耦点

### 1. SCF 概念 vs PySCF 实现

**问题**: `scf` 和 `pyscf` 混用

**解决**:
- `workflows/scf/` → SCF 物理流程（与软件无关）
- `core/physics/backends/pyscf/` → PySCF 具体实现
- `core/physics/backends/abacus/` → ABACUS 具体实现

### 2. 工作流编排 vs 任务执行

**问题**: 逻辑混在一起

**解决**:
- `workflows/` → 编排物理流程（what to do）
- `orchestration/` → 执行任务（how to do）

### 3. 准备/执行/收集 三阶段

**问题**: 所有逻辑在一个函数里

**解决**:
- `prepare.py` → 创建工作目录、生成输入文件
- `execute.py` → 调用调度器执行
- `collect.py` → 解析结果、整合数据

### 4. 后端接口统一

**问题**: PySCF 和 ABACUS 接口不一致

**解决**:
- 定义统一的 `PhysicsBackend` 接口
- 三个方法对应三阶段：
  - `generate_input_files()` → 准备
  - `run_calculation()` → 执行
  - `parse_results()` → 收集

## 代码示例

### 主入口

```python
# main.py
def main():
    # 1. 加载配置
    config = load_and_validate_config()

    # 2. 分发到工作流
    if config['command'] == 'scf':
        from deepks.workflows.scf import run_scf_workflow
        run_scf_workflow(config)
    elif config['command'] == 'iterate':
        from deepks.workflows.iterate import run_iterate_workflow
        run_iterate_workflow(config)
```

### SCF 工作流

```python
# workflows/scf/workflow.py
def run_scf_workflow(config):
    """SCF 工作流 - 体现物理流程."""
    # 1. 准备阶段
    tasks = prepare_scf_tasks(config)

    # 2. 执行阶段
    execute_scf_tasks(tasks, config)

    # 3. 收集阶段
    results = collect_scf_results(tasks, config)

    return results
```

### 后端接口

```python
# core/physics/backends/base.py
class PhysicsBackend(ABC):
    @abstractmethod
    def generate_input_files(self, system, work_dir, config):
        """准备阶段：生成输入文件."""
        pass

    @abstractmethod
    def run_calculation(self, work_dir, config):
        """执行阶段：运行计算."""
        pass

    @abstractmethod
    def parse_results(self, work_dir, config):
        """收集阶段：解析结果."""
        pass
```

## 验收标准

### 功能验收
- [ ] 所有命令正常工作（scf/train/iterate/stats）
- [ ] PySCF 和 ABACUS 后端都能运行
- [ ] 调度器功能正常
- [ ] 所有测试通过

### 架构验收
- [ ] 没有纯转发文件
- [ ] 目录结构反映物理流程
- [ ] SCF 和 PySCF 概念分离
- [ ] 工作流层和调度层职责清晰

### 代码质量验收
- [ ] 每个文件都有实际逻辑
- [ ] 函数职责单一
- [ ] 接口定义清晰
- [ ] 文档完整

## 时间估算

- 高优先级（必须）: 7 天
- 中优先级（重要）: 7 天
- 低优先级（可选）: 7 天

**总计**: 14-21 天

## 风险和缓解

### 风险 1: 破坏现有功能
**缓解**: 增量重构，每步运行测试

### 风险 2: 调度器集成复杂
**缓解**: 先保持 orchestration 不变，只修改调用方式

### 风险 3: 测试更新工作量大
**缓解**: 优先更新核心测试，分阶段更新

## 参考资料

- 当前架构分析: `docs/refactor_phase1_structure_map.md`
- 输入参数文档: `docs/input-parameter.md`
- 测试用例: `tests/`
