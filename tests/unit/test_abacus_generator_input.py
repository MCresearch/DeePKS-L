"""
整体覆盖：ABACUS 接口输入文件生成器。

测试列表：
- `test_make_abacus_kpt_default_and_custom`
- `test_make_abacus_input_contains_key_parameters`
- `test_make_abacus_stru_with_lcao_and_descriptor`
"""

import numpy as np

from deepks.pipelines.iterate.generator_abacus import (
	make_abacus_scf_input,
	make_abacus_scf_kpt,
	make_abacus_scf_stru,
)


def test_make_abacus_kpt_default_and_custom():
	"""
	依赖：`deepks.pipelines.iterate.generator_abacus.make_abacus_scf_kpt`。
	测试内容：验证默认 KPT 与自定义 k_points 生成内容。
	"""
	kpt_default = make_abacus_scf_kpt({})
	assert "K_POINTS" in kpt_default
	assert "Gamma" in kpt_default
	assert "1 1 1 0 0 0" in kpt_default

	kpt_custom = make_abacus_scf_kpt({"k_points": [2, 2, 1, 0, 0, 0]})
	assert "2 2 1 0 0 0" in kpt_custom


def test_make_abacus_input_contains_key_parameters():
	"""
	依赖：`deepks.pipelines.iterate.generator_abacus.make_abacus_scf_input`。
	测试内容：验证 INPUT 关键参数（基组、泛函、deepks 参数）被正确写入。
	"""
	fp = {
		"ecutwfc": 60,
		"scf_thr": 1e-7,
		"scf_nmax": 80,
		"basis_type": "lcao",
		"dft_functional": "pbe",
		"gamma_only": 1,
		"mixing_type": "pulay",
		"mixing_beta": 0.4,
		"nspin": 1,
		"smearing_method": "gaussian",
		"smearing_sigma": 0.02,
		"k_points": None,
		"cal_force": 1,
		"cal_stress": 0,
		"deepks_out_labels": 1,
		"deepks_scf": 1,
		"deepks_bandgap": 0,
		"deepks_v_delta": 0,
		"model_file": "model.ptg",
		"out_wfc_lcao": 0,
	}
	text = make_abacus_scf_input(fp)
	assert "INPUT_PARAMETERS" in text
	assert "basis_type lcao" in text
	assert "dft_functional pbe" in text
	assert "deepks_scf 1" in text
	assert "deepks_model model.ptg" in text


def test_make_abacus_stru_with_lcao_and_descriptor():
	"""
	依赖：`deepks.pipelines.iterate.generator_abacus.make_abacus_scf_stru`。
	测试内容：验证 STRU 中赝势、轨道文件、坐标与 NUMERICAL_DESCRIPTOR 段生成正确。
	"""
	sys_data = {
		"atom_names": ["H", "O"],
		"atom_numbs": [2, 1],
		"cells": [np.eye(3)],
		"coords": [
			np.array(
				[
					[0.0, 0.0, 0.0],
					[0.0, 0.0, 1.0],
					[0.0, 0.0, 2.0],
				]
			)
		],
	}
	pp_files = ["H_ONCV.upf", "O_ONCV.upf"]
	params = {
		"lattice_constant": 1,
		"coord_type": "Cartesian",
		"basis_type": "lcao",
		"orb_files": ["H_gga.orb", "O_gga.orb"],
		"deepks_scf": 1,
		"deepks_out_labels": 1,
		"proj_file": ["jle.orb"],
	}

	text = make_abacus_scf_stru(sys_data, pp_files, params)
	assert "ATOMIC_SPECIES" in text
	assert "H 1.00 H_ONCV.upf" in text
	assert "O 1.00 O_ONCV.upf" in text
	assert "NUMERICAL_ORBITAL" in text
	assert "H_gga.orb" in text and "O_gga.orb" in text
	assert "NUMERICAL_DESCRIPTOR" in text
	assert "jle.orb" in text


