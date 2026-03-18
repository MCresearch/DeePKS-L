"""
整体覆盖：ABACUS 侧最小流程编排（不执行外部 ABACUS 二进制）。

测试列表：
- `test_make_scf_abacus_sequence_structure`
"""

from pathlib import Path

import numpy as np

from deepks.pipelines.iterate.template_abacus import make_scf_abacus
from deepks.orchestration.workflow.workflow import Sequence


def test_make_scf_abacus_sequence_structure(tmp_path, monkeypatch):
	"""
	依赖：`deepks.pipelines.iterate.template_abacus.make_scf_abacus`。
	测试内容：本地构建 ABACUS SCF workflow，验证返回 `Sequence` 且包含预处理/运行/后处理三个步骤。
	"""
	monkeypatch.chdir(tmp_path)

	# share 目录与必要文件（会被 check_share_folder 消费）
	share = tmp_path / "share"
	share.mkdir()
	for fn in ["H.orb", "O.orb", "H.upf", "O.upf", "jle.orb"]:
		(tmp_path / fn).write_text("placeholder", encoding="utf-8")

	# 系统目录（构建阶段仅检查路径，不做真实运行）
	trn = tmp_path / "systems" / "group.00"
	tst = tmp_path / "systems" / "group.01"
	trn.mkdir(parents=True)
	tst.mkdir(parents=True)

	# make_run_scf_abacus 构建阶段会读取 atom.npy 或 coord/type 信息
	coord = np.array([[[0.0, 0.0, 0.0]]])
	np.save(trn / "coord.npy", coord)
	np.save(tst / "coord.npy", coord)
	(trn / "type_map.raw").write_text("H\n", encoding="utf-8")
	(tst / "type_map.raw").write_text("H\n", encoding="utf-8")
	np.savetxt(trn / "type.raw", np.array([1], dtype=int), fmt="%d")
	np.savetxt(tst / "type.raw", np.array([1], dtype=int), fmt="%d")

	wf = make_scf_abacus(
		systems_train=[str(trn)],
		systems_test=[str(tst)],
		cleanup=False,
		resources={"task_per_node": 1},
		dispatcher={"context": "local", "batch": "shell"},
		no_model=True,
		group_size=1,
		workdir="00.scf",
		share_folder="share",
		model_file=None,
		orb_files=[str(tmp_path / "H.orb"), str(tmp_path / "O.orb")],
		pp_files=[str(tmp_path / "H.upf"), str(tmp_path / "O.upf")],
		proj_file=[str(tmp_path / "jle.orb")],
		run_cmd="mpirun",
		abacus_path="/bin/echo",
	)

	assert isinstance(wf, Sequence)
	# pre / run / post
	assert len(wf.child_tasks) == 3


