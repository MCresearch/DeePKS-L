"""
整体覆盖：迁移后测试资产完整性检查。

测试列表：
- `test_migrated_unit_files_exist`
- `test_migrated_integration_files_exist`
- `test_legacy_folders_removed`
- `test_legacy_integral_fixtures_exist`
"""

from pathlib import Path


def test_migrated_unit_files_exist():
	"""
	依赖：新测试目录 `tests/unit`。
	测试内容：验证迁移后的历史样例单测文件存在。
	"""
	root = Path(__file__).resolve().parents[2] / "unit"
	expected = [
		"cross_cutting/test_migrated_unittests_samples.py",
	]
	for name in expected:
		assert (root / name).is_file()


def test_migrated_integration_files_exist():
	"""
	依赖：新测试目录 `tests/integration`。
	测试内容：验证迁移后的历史样例集成测试文件存在。
	"""
	root = Path(__file__).resolve().parents[2] / "integration"
	expected = [
		"scenarios/test_migrated_integral_samples.py",
	]
	for name in expected:
		assert (root / name).is_file()


def test_legacy_folders_removed():
	"""
	依赖：仓库测试目录结构。
	测试内容：验证原始 `tests/unittests` 与 `tests/integral` 已移除，避免重复维护。
	"""
	root = Path(__file__).resolve().parents[2]
	assert not (root / "unittests").exists()
	assert not (root / "integral").exists()


def test_legacy_integral_fixtures_exist():
	"""
	依赖：新场景目录 `tests/integration/scenarios/legacy_integral_full`。
	测试内容：验证历史 integral 样例数据已迁移到 integration scenarios。
	"""
	root = Path(__file__).resolve().parent / "legacy_integral_full"
	assert root.is_dir()
	assert (root / "train" / "train_input.yaml").is_file()
	assert (root / "test" / "test_input.yaml").is_file()
	assert (root / "scf" / "scf_input.yaml").is_file()
	assert (root / "stats" / "stats_input.yaml").is_file()
	assert (root / "iterate" / "abacus_local" / "iter_input.yaml").is_file()
