"""
整体覆盖：调度层离线可验证能力（任务分组、状态机、脚本构建）。

测试列表：
- `test_split_tasks_total_and_sizes`
- `test_jobrecord_state_machine`
- `test_slurm_script_building`
- `test_slurm_parallel_substep_script`
- `test_pbs_script_building`
- `test_shell_script_building_and_mpi_cmd`
"""

from pathlib import Path

from deepks.task.job.dispatcher import JobRecord, _split_tasks
from deepks.task.job.local_context import LocalContext, LocalSession
from deepks.task.job.pbs import PBS
from deepks.task.job.shell import Shell
from deepks.task.job.slurm import Slurm


def _mk_context(tmp_path):
	local_root = tmp_path / "local"
	remote_root = tmp_path / "remote"
	local_root.mkdir(parents=True, exist_ok=True)
	remote_root.mkdir(parents=True, exist_ok=True)
	session = LocalSession({"work_path": str(remote_root)})
	return LocalContext(str(local_root), session, job_uuid="job-ut")


def test_split_tasks_total_and_sizes():
	"""依赖：`_split_tasks`。测试内容：验证分组后元素不丢失、分组数量正确。"""
	tasks = list(range(10))
	chunks = _split_tasks(tasks, group_size=3)
	flat = [x for c in chunks for x in c]
	assert sorted(flat) == tasks
	assert len(chunks) == 4


def test_jobrecord_state_machine(tmp_path):
	"""依赖：`JobRecord`。测试内容：验证提交、失败计数、完成计数和持久化加载。"""
	task_chunks = [
		[
			{"dir": "a", "cmds": ["echo a"], "_label": "A"},
			{"dir": "b", "cmds": ["echo b"], "_label": "B"},
		],
		[{"dir": "c", "cmds": ["echo c"], "_label": "C"}],
	]
	jr = JobRecord(str(tmp_path), task_chunks, fname="jr_ut.json")

	assert jr.get_total_tasks() == 3
	assert jr.get_completed_tasks() == 0

	h1 = list(jr.record.keys())[0]
	jr.record_remote_context(h1, "L", "R", "UUID-1")
	assert jr.check_submitted(h1)
	assert jr.get_uuid(h1) == "UUID-1"

	jr.increase_nfail(h1)
	assert jr.check_nfail(h1) == 1

	jr.record_finish(h1)
	assert jr.check_finished(h1)
	assert jr.get_completed_tasks() == 2

	jr.dump()
	jr2 = JobRecord(str(tmp_path), task_chunks, fname="jr_ut.json")
	assert jr2.get_completed_tasks() == 2


def test_slurm_script_building(tmp_path):
	"""依赖：`Slurm.sub_script`。测试内容：验证关键 `#SBATCH` 头与环境变量拼接。"""
	ctx = _mk_context(tmp_path)
	slurm = Slurm(ctx, uuid_names=False)

	script = slurm.sub_script(
		job_dirs=["t0"],
		cmds=[["python run.py"]],
		res={
			"numb_node": 1,
			"task_per_node": 2,
			"cpus_per_task": 4,
			"time_limit": "00:10:00",
			"mem_limit": 8,
			"partition": "cpu",
			"with_mpi": False,
			"envs": {"OMP_NUM_THREADS": "4"},
		},
	)
	assert "#SBATCH -N 1" in script
	assert "#SBATCH --ntasks-per-node=2" in script
	assert "#SBATCH --cpus-per-task=4" in script
	assert "#SBATCH --partition=cpu" in script
	assert "export OMP_NUM_THREADS=4" in script
	assert "touch tag_finished" in script


def test_slurm_parallel_substep_script(tmp_path):
	"""依赖：`Slurm.sub_script`。测试内容：验证并行子任务脚本片段（pids/wait/srun）。"""
	ctx = _mk_context(tmp_path)
	slurm = Slurm(ctx, uuid_names=False)
	script = slurm.sub_script(
		job_dirs=["t0", "t1"],
		cmds=[["python a.py"], ["python b.py"]],
		para_deg=2,
		para_res=[
			{"numb_node": 1, "cpus_per_task": 2, "task_per_node": 4},
			{"numb_node": 1, "cpus_per_task": 2, "task_per_node": 4},
		],
		res={"with_mpi": False},
	)
	assert 'pids=""; FAIL=0' in script
	assert "srun" in script
	assert "for p in $pids; do wait $p" in script


def test_pbs_script_building(tmp_path):
	"""依赖：`PBS.sub_script`。测试内容：验证 `#PBS` 资源字段与工作目录语句。"""
	ctx = _mk_context(tmp_path)
	pbs = PBS(ctx, uuid_names=False)
	script = pbs.sub_script(
		job_dirs=["t0"],
		cmds=[["python run.py"]],
		res={
			"numb_node": 1,
			"task_per_node": 8,
			"numb_gpu": 1,
			"time_limit": "00:30:00",
			"partition": "batch",
			"with_mpi": False,
		},
	)
	assert "#PBS -l nodes=1:ppn=8:gpus=1" in script
	assert "#PBS -l walltime=00:30:00" in script
	assert "#PBS -q batch" in script
	assert "cd  $PBS_O_WORKDIR" in script
	assert "touch tag_finished" in script


def test_shell_script_building_and_mpi_cmd(tmp_path):
	"""依赖：`Shell.sub_script`。测试内容：验证 shell 环境配置与 `mpirun` 命令拼接。"""
	ctx = _mk_context(tmp_path)
	sh = Shell(ctx, uuid_names=False)
	script = sh.sub_script(
		job_dirs=["t0"],
		cmds=[["python run.py"]],
		res={
			"task_per_node": 4,
			"with_mpi": True,
			"envs": {"A": "1"},
			"module_list": ["gcc/12"],
			"module_unload_list": ["old"],
			"source_list": ["~/.bashrc"],
		},
	)
	assert "#!/bin/bash" in script
	assert "export A=1" in script
	assert "module unload old" in script
	assert "module load gcc/12" in script
	assert "source ~/.bashrc" in script
	assert "mpirun -n 4 python run.py" in script
	assert "touch tag_finished" in script


