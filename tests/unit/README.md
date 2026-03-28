# unit/

Unit 测试只验证子功能与局部边界。

约束：
- 按源码结构组织，而不是平铺在 `unit/` 根目录。
- 优先放到 `cli/`、`io/input/`、`workflows/*/`、`orchestration/*/`、`physics/*/`、`ml/*/`、`utils/`。
- 只有无法单独归属的历史混合测试，才允许放在 `cross_cutting/`。
