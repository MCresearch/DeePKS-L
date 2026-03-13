"""
整体覆盖：`Iteration.restart()` 的断点恢复链路。

测试列表：
- `test_iteration_restart_from_record`
"""

from pathlib import Path

from deepks.task.task import PythonTask
from deepks.task.workflow import Iteration, Sequence


def _mark(name):
    with open("mark.log", "a", encoding="utf-8") as fp:
        fp.write(name + "\n")


def test_iteration_restart_from_record(tmp_path, monkeypatch):
    """
    依赖：`deepks.task.workflow.Iteration/Sequence` 与 `PythonTask`。
    测试内容：模拟已有 RECORD 的中断场景，验证重启后仅补齐未完成步骤。
    """
    monkeypatch.chdir(tmp_path)

    per_iter = Sequence(
        [
            PythonTask(_mark, call_args=["a"], workdir="a"),
            PythonTask(_mark, call_args=["b"], workdir="b"),
        ],
        workdir=".",
    )
    wf = Iteration(per_iter, 2, workdir=".", record_file="RECORD")

    # 模拟中断后已完成到 tag=(0,0)
    Path("RECORD").write_text("0 0\n", encoding="utf-8")

    wf.restart()

    # (0,0) 不应重复，剩余应执行 (0,1),(1,0),(1,1)
    c0b = Path("iter.00") / "b" / "mark.log"
    c1a = Path("iter.01") / "a" / "mark.log"
    c1b = Path("iter.01") / "b" / "mark.log"
    assert c0b.exists()
    assert c1a.exists()
    assert c1b.exists()
