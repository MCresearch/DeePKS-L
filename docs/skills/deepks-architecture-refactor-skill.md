# DeePKS Architecture Refactor Skill (v1)

## 1. 目标
将项目重构为三层结构，并保证可回滚、可验证、可渐进迁移：

1) 外层（Orchestration）：任务调度、资源管理、运行状态与断点恢复。
2) 中层（I/O）：数据读取、预处理、序列化、格式适配。
3) 底层（Core）：网络模型与物理模型解耦，统一输入输出契约。

## 2. 重构原则
- **依赖单向**：`orchestration -> io -> core`，禁止反向依赖。
- **接口先行**：先定义契约（data schema / backend protocol），再搬运实现。
- **兼容优先**：短期保留旧导入路径（shim），先不破坏 CLI 行为。
- **小步提交**：每次变更必须通过测试与最小可运行验证。

## 3. 目标目录（草案）
```text
deepks/
  cli/
  orchestration/
    workflow/
    scheduler/
    checkpoint/
    state/
  io/
    readers/
    writers/
    transforms/
    schemas/
    adapters/
  core/
    contracts/
    ml/
      models/
      losses/
      train/
      eval/
    physics/
      pyscf/
      abacus/
      operators/
  pipelines/
    train/
    scf/
    iterate/
  compat/
```

## 4. 分阶段计划

### Phase 0: 基线冻结（测试与行为）
- 建立当前行为基线：CLI 子命令、关键 API、典型数据流。
- 补齐 smoke 测试，记录 golden outputs（允许数值容差）。

### Phase 1: 契约层
- 新增 `core/contracts`：
  - `ModelBackend`（训练/推理接口）
  - `PhysicsBackend`（物理求解接口）
  - `SampleSchema`（统一字段、shape、unit 约定）
- 不改旧实现，仅增加适配器。

### Phase 2: I/O 下沉
- 将 `reader` / 数据变换迁移到 `io`。
- 把输入文件名与字段映射放到 `io/schemas`。
- 保留 `deepks.model.reader` 兼容导出。

### Phase 3: 核心解耦
- `model/*` 拆到 `core/ml`。
- `scf/*` 拆到 `core/physics`。
- 通过 contracts 连接，不允许直接互相引用实现细节。

### Phase 4: 外层编排迁移
- `task/*`、`iterate/*` 迁移到 `orchestration` + `pipelines`。
- `main.py` 改为调用 pipelines，CLI 参数行为保持不变。

### Phase 5: 清理与收口
- 删除冗余 shim（在 deprecation 周期后）。
- 更新文档、示例、开发者迁移指南。

## 5. 文件映射建议（首批）
- `deepks/task/*` -> `deepks/orchestration/*`
- `deepks/iterate/*` -> `deepks/orchestration/*` + `deepks/pipelines/iterate/*`
- `deepks/model/reader.py` -> `deepks/io/readers/group_reader.py`
- `deepks/model/model.py` -> `deepks/core/ml/models/corrnet.py`
- `deepks/model/evaluator.py` -> `deepks/core/ml/eval/evaluator.py`
- `deepks/scf/*` -> `deepks/core/physics/pyscf/*`

## 6. 风险与控制
- **风险**：路径变更导致 import 断裂。
  - 控制：compat shim + `pytest -k import` 检查。
- **风险**：数值行为偏移。
  - 控制：固定随机种子 + 容差断言 + 小样本 golden。
- **风险**：调度恢复失效。
  - 控制：断点恢复集成测试（RECORD / 失败重启）。

## 7. 每次 PR 的验收清单
- [ ] 单元测试通过
- [ ] 至少 1 个 smoke 流程通过（train/test/scf/iterate 之一）
- [ ] 旧入口兼容（CLI 与旧 import）
- [ ] 文档更新（变更点+迁移说明）

## 8. 执行顺序建议
1) 先测试基线（Phase 0）
2) 再 contracts（Phase 1）
3) 然后 I/O（Phase 2）
4) 再 core 解耦（Phase 3）
5) 最后 orchestration/pipelines（Phase 4）

---
维护约定：任何重构任务必须绑定“测试项 + 回滚点 + 兼容策略”。
