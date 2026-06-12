"""
整体覆盖：`Sequence` 任务链构建与执行行为。

测试列表：
- `test_sequence_chain_and_run`
"""

from pathlib import Path

from deepks.orchestration.workflow.task import PythonTask
from deepks.orchestration.workflow.workflow import Sequence


def _append_line(name):
    with open("events.log", "a", encoding="utf-8") as fp:
        fp.write(name + "\n")


def test_sequence_chain_and_run(tmp_path, monkeypatch):
    """
    依赖：`deepks.orchestration.workflow.workflow.Sequence`、`deepks.orchestration.workflow.task.PythonTask`。
    测试内容：验证任务链自动连接，且执行后在各子目录产生预期输出。
    """
    monkeypatch.chdir(tmp_path)
    t1 = PythonTask(_append_line, call_args=["task1"], workdir="s1")
    t2 = PythonTask(_append_line, call_args=["task2"], workdir="s2")

    seq = Sequence([t1, t2], workdir="wf")

    # 链接关系在 Sequence.postmod_hook 中建立
    assert seq.child_tasks[1].prev_task is not None

    seq.run()

    log = Path("wf") / "s1" / "events.log"
    assert log.exists()
    assert log.read_text(encoding="utf-8").strip() == "task1"

    log2 = Path("wf") / "s2" / "events.log"
    assert log2.exists()
    assert log2.read_text(encoding="utf-8").strip() == "task2"
