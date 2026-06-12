# DeePKS Architecture

当前代码的正式主结构由两条主线组成：

```text
main -> config -> dispatcher -> workflows
```

以及：

```text
ml <- interface -> physics
```

> Note: `deepks/config/` (formerly `deepks/io/input/`) houses the CLI-side
> configuration pipeline (load / normalize / validate / defaults / merge /
> package / dispatch). It was moved out of `io/` because it depends on
> recipe definitions from `interface/`, which is above `io/` in the
> dependency arrow.

## 1. 输入与 workflow 主线

顶层用户入口仍然只有：

- [main.py](/home/ubuntu/work/DeePKS-L/deepks/main.py)

输入处理固定为：

```text
load
-> normalize
-> validate
-> defaults
-> merge
-> package
-> dispatch
```

对应文件：

- [loader.py](/home/ubuntu/work/DeePKS-L/deepks/config/loader.py)
- [normalize.py](/home/ubuntu/work/DeePKS-L/deepks/config/normalize.py)
- [validator.py](/home/ubuntu/work/DeePKS-L/deepks/config/validator.py)
- [defaults.py](/home/ubuntu/work/DeePKS-L/deepks/config/defaults.py)
- [merger.py](/home/ubuntu/work/DeePKS-L/deepks/config/merger.py)
- [packager.py](/home/ubuntu/work/DeePKS-L/deepks/config/packager.py)
- [dispatcher.py](/home/ubuntu/work/DeePKS-L/deepks/config/dispatcher.py)

workflow 只消费已经打包好的：

- `train_param`
- `test_param`
- `scf_param`
- `stats_param`
- `iterate_param`

不再在运行阶段重新做输入结构兼容、别名映射或默认值填充。

当前 `workflows/` 的正式边界也已经固定：

- `train`
  - 只做 train workflow 编排
  - 训练参数翻译与 reader/runtime 准备已下沉到 `interface.train`
- `scf`
  - 只做 `prepare -> execute -> collect` 编排
  - backend 输入生成、执行 task spec、结果解析已下沉到 `physics.backends`
- `stats`
  - 只做 stats workflow 编排
  - stats 参数翻译与统计调用已下沉到 `interface.stats`
- `iterate`
  - 只做 iterate 工作流树组装
  - child snapshot、task-yaml materialization、step payload 解释已下沉到 `workflows.iterate.support`
  - ABACUS iterate sequence 与 task builder 位于 `workflows.iterate.abacus`

## 2. `ml <- interface -> physics`

### `ml`

位置：

- [ml/base.py](/home/ubuntu/work/DeePKS-L/deepks/ml/base.py)
- [ml/models](/home/ubuntu/work/DeePKS-L/deepks/ml/models)
- [ml/train](/home/ubuntu/work/DeePKS-L/deepks/ml/train)

职责：

- 模型定义
- 模型输入上的导数计算
- 通用训练循环
- 通用损失函数
- checkpoint

`ml` 不依赖：

- `physics`
- `interface`

### `physics`

位置：

- [physics/base.py](/home/ubuntu/work/DeePKS-L/deepks/physics/base.py)
- [physics/contracts.py](/home/ubuntu/work/DeePKS-L/deepks/physics/contracts.py)
- [physics/engine.py](/home/ubuntu/work/DeePKS-L/deepks/physics/engine.py)
- [physics/schemes](/home/ubuntu/work/DeePKS-L/deepks/physics/schemes)
- [physics/properties](/home/ubuntu/work/DeePKS-L/deepks/physics/properties)
- [physics/backends](/home/ubuntu/work/DeePKS-L/deepks/physics/backends)

职责：

- 物理量实现
- 物理量 / 模型导数键契约
- 具体网络方案下的物理量恢复
- 物理量共享调度
- SCF/backend 执行

`physics` 不依赖：

- `ml`
- `interface`

### `interface`

位置：

- [interface/adapters](/home/ubuntu/work/DeePKS-L/deepks/interface/adapters)
- [interface/objectives](/home/ubuntu/work/DeePKS-L/deepks/interface/objectives)
- [interface/recipes](/home/ubuntu/work/DeePKS-L/deepks/interface/recipes)
- [interface/reducers.py](/home/ubuntu/work/DeePKS-L/deepks/interface/reducers.py)
- [interface/registry.py](/home/ubuntu/work/DeePKS-L/deepks/interface/registry.py)
- [interface/schemas.py](/home/ubuntu/work/DeePKS-L/deepks/interface/schemas.py)
- [interface/model_builder.py](/home/ubuntu/work/DeePKS-L/deepks/interface/model_builder.py)

> ``TaskBatch`` (the structured per-sample container) lives in
> [io/batch.py](/home/ubuntu/work/DeePKS-L/deepks/io/batch.py) — it is the
> data carrier used by every layer, and keeping it in io avoids any
> upward import from io/ml/physics.

职责：

- 作为唯一组装层同时依赖 `ml` 和 `physics`
- 把 batch、模型、property scheme、objective terms 组装成具体任务
- 选择 recipe 和模型家族
- 通过 recipe schema 显式声明所用 `property_scheme`
- 持有任务专用预处理和评估 runner

## 3. 当前正式抽象

### `ml`

定义于 [ml/base.py](/home/ubuntu/work/DeePKS-L/deepks/ml/base.py)：

- `ModelAdapter`
- `BatchProtocol`

训练主链 [trainer.py](/home/ubuntu/work/DeePKS-L/deepks/ml/train/trainer.py)
只依赖模型、batch 协议和 interface 注入的 objective。

### `interface`

定义于 [interface/base.py](/home/ubuntu/work/DeePKS-L/deepks/interface/base.py)：

- `ObjectiveAdapter`

### `physics`

定义于 [physics/base.py](/home/ubuntu/work/DeePKS-L/deepks/physics/base.py)：

- `RepresentationBuilder`
- `PropertyScheme`
- `BackendRunner`

并由 [contracts.py](/home/ubuntu/work/DeePKS-L/deepks/physics/contracts.py)
提供共享命名契约：

- `PropertyNames`
- `ModelOutputKeys`
- `ModelDerivativeKeys`

其中：

- `physics/properties/*`
  只保留按物理量组织的纯实现函数
- `PropertyScheme`
  定义某个具体网络方案下，如何从模型输出/导数恢复所请求的物理量
- [engine.py](/home/ubuntu/work/DeePKS-L/deepks/physics/engine.py)
  不再隐藏默认 scheme；必须由 interface 显式注入一个 scheme，并按请求组织多个物理量计算

## 4. 当前主训练路径

主路径现在是：

```text
workflow
-> recipe
-> recipe schema property_scheme
-> ObjectiveAdapter
-> model forward / autograd
-> PropertyEngine
-> PropertyScheme
-> physics/properties/*
-> interface objective terms
-> Trainer
```

对应核心文件：

- [corrnet_energy.py](/home/ubuntu/work/DeePKS-L/deepks/interface/recipes/corrnet_energy.py)
- [descriptor_properties.py](/home/ubuntu/work/DeePKS-L/deepks/interface/objectives/descriptor_properties.py)
- [engine.py](/home/ubuntu/work/DeePKS-L/deepks/physics/engine.py)
- [energy_descriptor.py](/home/ubuntu/work/DeePKS-L/deepks/physics/schemes/energy_descriptor.py)
- [properties](/home/ubuntu/work/DeePKS-L/deepks/physics/properties)
- [terms.py](/home/ubuntu/work/DeePKS-L/deepks/interface/objectives/terms.py)
- [trainer.py](/home/ubuntu/work/DeePKS-L/deepks/ml/train/trainer.py)

这条链上：

- `ml` 不知道 force/stress/band/phi 是怎么恢复的
- `physics` 不知道训练循环和 recipe
- `interface` 负责把两边接起来

## 5. iterate 的特殊点

`iterate` 在 package 阶段就提前生成子任务 packed 配置。

也就是说：

- 顶层输入只做一次完整：
  - normalize
  - validate
  - defaults
  - merge
  - package
- `iterate` 生成：
  - `share/train_input.yaml`
  - `share/scf_input.yaml`
  - `share/init_train.yaml`
  - `share/init_scf*.yaml`
- 这些子配置已是 packed config
- 子任务再次执行 `deepks xxx.yaml` 时，会通过内部标记直接走 packed fast-path

当前 iterate 相关实现边界：

- [prepare.py](/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/prepare.py)
  - 只组装 `Iteration` / `Sequence`
- [task_templates.py](/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/support/task_templates.py)
  - 持有通用 iterate task-template 构造
- [snapshots.py](/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/support/snapshots.py)
  - 持有 `share/*.yaml` 快照落盘与资源 materialize
- [task_yaml.py](/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/support/task_yaml.py)
  - 持有 child-task YAML 重写
- [sequence.py](/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/abacus/sequence.py)
  - 持有 ABACUS iterate sequence 组装
- [tasks.py](/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/abacus/tasks.py)
  - 持有 ABACUS iterate task builders（注入 `check_system_names` / `coord_to_atom`）
- [iterate_ops.py](/home/ubuntu/work/DeePKS-L/deepks/physics/backends/abacus/iterate_ops.py)
  - ABACUS 数据解析 / dump 转换（physics 侧 helper，被 workflow tasks 调用）

详见：

- [iterate-call-stack.md](/home/ubuntu/work/DeePKS-L/docs/iterate-call-stack.md)

## 6. 当前状态

当前主路径已经只保留新结构：

- `ml` 负责纯模型与训练循环
- `physics` 负责物理量实现、scheme 恢复逻辑、engine 调度、backend
- `interface` 负责组装 batch / preprocess / eval runner / objective / recipe

当前仍存在的旧语义只剩少量内部实现细节，例如 `CorrNet` embedding 辅助类内部仍使用 `shell_sec` 命名；
这类命名已被限制在 `ml.models.corrnet` 内部实现，不再作为 interface 或 workflow 的公开契约。

## 7. 依赖规则

允许：

- `interface -> ml`
- `interface -> physics`
- `workflows -> interface`
- `workflows -> ml`
- `workflows -> physics`

禁止：

- `ml -> physics`
- `ml -> interface`
- `physics -> ml`
- `physics -> interface`
- 在 workflow 执行阶段继续做输入结构兼容/重命名/默认值设置

## 8. 相关文档

- [iterate-call-stack.md](/home/ubuntu/work/DeePKS-L/docs/iterate-call-stack.md)
- [.claude/skills/deepks-physics-ml-decoupling-refactor-skill.md](/home/ubuntu/work/DeePKS-L/.claude/skills/deepks-physics-ml-decoupling-refactor-skill.md)
- [.claude/skills/deepks-old-architecture-archive-skill.md](/home/ubuntu/work/DeePKS-L/.claude/skills/deepks-old-architecture-archive-skill.md)
