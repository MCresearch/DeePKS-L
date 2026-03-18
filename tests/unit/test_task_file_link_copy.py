"""
整体覆盖：任务预处理阶段的文件链接/复制语义。

测试列表：
- `test_task_preprocess_link_and_copy`
"""

from pathlib import Path

from deepks.orchestration.workflow.task import BlankTask


def _write(p: Path, txt: str):
	p.parent.mkdir(parents=True, exist_ok=True)
	p.write_text(txt, encoding="utf-8")


def test_task_preprocess_link_and_copy(tmp_path):
	"""
	依赖：`deepks.orchestration.workflow.task.BlankTask`、本地文件系统（tmp_path）。
	测试内容：验证 `prev/share/abs` 三类来源在 link/copy 模式下均按预期落盘。
	"""
	prev = tmp_path / "prev"
	share = tmp_path / "share"
	absdir = tmp_path / "abs"
	work = tmp_path / "work"

	_write(prev / "a.txt", "from-prev")
	_write(share / "b.txt", "from-share")
	_write(absdir / "c.txt", "from-abs")
	_write(absdir / "d.txt", "from-abs-copy")

	task = BlankTask(
		workdir=work,
		prev_folder=prev,
		share_folder=share,
		link_prev_files=[("a.txt", "in/pa.txt")],
		copy_share_files=[("b.txt", "in/sb.txt")],
		link_abs_files=[(str(absdir / "c.txt"), "in/ac.txt")],
		copy_abs_files=[(str(absdir / "d.txt"), "in/ad.txt")],
	)
	task.run()

	pa = work / "in" / "pa.txt"
	sb = work / "in" / "sb.txt"
	ac = work / "in" / "ac.txt"
	ad = work / "in" / "ad.txt"

	assert pa.exists() and pa.is_symlink()
	assert sb.exists() and not sb.is_symlink()
	assert ac.exists() and ac.is_symlink()
	assert ad.exists() and not ad.is_symlink()

	assert pa.read_text(encoding="utf-8") == "from-prev"
	assert sb.read_text(encoding="utf-8") == "from-share"
	assert ac.read_text(encoding="utf-8") == "from-abs"
	assert ad.read_text(encoding="utf-8") == "from-abs-copy"


