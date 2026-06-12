# DeePKS 测试目录说明

测试目录按“验证范围”组织，而不是按历史来源组织。

## 目录含义

- `data/`
  - 只放纯测试数据、golden 输出、最小输入系统。
  - 禁止放 `test_*.py`。

- `smoke/`
  - 只做最薄的存活性检查。
  - 例如 CLI 帮助页、入口导入、最小命令可调用性。

- `unit/`
  - 只测子功能与局部边界。
  - 目录结构尽量镜像源码架构，例如 `config/`、`workflows/iterate`、`orchestration/scheduler`。

- `integration/`
  - 测完整工作流或多层协作。
  - 历史样例场景统一放在 `integration/scenarios/`。

## 组织规则

- 新的纯数据统一进 `tests/data/`。
- 新的样例驱动工作流测试统一进 `tests/integration/scenarios/`。
- `tests/unit/` 顶层不再新增平铺测试文件，必须落到与源码对应的子目录。
- `tests/data/` 中不允许出现 `test_*.py`。

## 运行方式

- 全量：
  - `pytest -q tests/smoke tests/unit tests/integration`

- 仅 smoke：
  - `pytest -q tests/smoke`

- 仅 unit：
  - `pytest -q tests/unit`

- 仅 integration：
  - `pytest -q tests/integration`

## 可选依赖说明

- `pyscf`、`abacus`、`pyabacus` 为可选依赖。
- 若环境未安装，对应测试会自动 `skip`，不影响其他层级。
- `pyabacus` 相关测试默认关闭；仅在显式设置 `ENABLE_PYABACUS_TESTS=1` 时启用。
