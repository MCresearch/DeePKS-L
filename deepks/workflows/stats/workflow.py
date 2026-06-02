"""Stats workflow orchestration."""

from deepks.workflows.stats.runtime import run_stats


def run_stats_workflow(config):
    """Run stats collection/reporting using the physics stats adapter."""
    runtime = config.get("runtime") if isinstance(config.get("runtime"), dict) else {}
    runtime_io = runtime.get("io") if isinstance(runtime.get("io"), dict) else {}
    log_file = runtime_io.get("stats_log", runtime_io.get("log_file", "log.stats"))
    return run_stats(config, log_file)
