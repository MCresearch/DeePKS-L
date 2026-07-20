# minimal_systems/

用途：放置最小可复现输入系统。

建议内容：
- 最小的 `atom.npy/coord.npy/type.raw`
- 对应的 `energy.npy/force.npy`（如果测试需要）

目标：让 smoke/unit/integration 都能共享一套极小数据，减少重复造数。
