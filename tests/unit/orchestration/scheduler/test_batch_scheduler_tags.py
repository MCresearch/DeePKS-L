"""Regression tests for batch-script per-task completion tags."""

from deepks.orchestration.scheduler.job.batch import Batch


class _DummyContext:
    remote_root = "/tmp/work"
    job_uuid = "dummy-job"


class _DummyBatch(Batch):
    def check_status(self):
        raise NotImplementedError

    def default_resources(self, res):
        return {} if res is None else res

    def sub_script_head(self, res):
        return ""

    def sub_script_cmd(self, cmd, arg, res):
        return f"{cmd} {arg}".strip()

    def exec_sub_script(self, script_str):
        raise NotImplementedError


def test_sub_script_uses_unique_finished_tags_for_each_task():
    batch = _DummyBatch(_DummyContext(), uuid_names=False)

    script = batch.sub_script(
        job_dirs=["systems", "systems"],
        cmds=[["echo first"], ["echo second"]],
        outlog="log",
        errlog="err",
    )

    assert "tag_0_0_finished" in script
    assert "tag_0_1_finished" in script
    assert "tag_0_finished" not in script
