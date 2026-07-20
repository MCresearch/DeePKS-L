# Iterate Call Stack

在每次重构代码后都需要相应更新此文件。  
当前内容已按 2026-04-16 的代码状态同步。

示例输入：

- [iter_input.yaml](/home/ubuntu/work/DeePKS-L/tests/integration/scenarios/integral_full/iterate/abacus_local/iter_input.yaml)

示例命令：

```text
deepks tests/integration/scenarios/integral_full/iterate/abacus_local/iter_input.yaml
```

## 1. 顶层 CLI 调用栈

```text
main(/home/ubuntu/work/DeePKS-L/deepks/main.py)
-> _configure_line_buffered_stdio(/home/ubuntu/work/DeePKS-L/deepks/main.py)
-> argparse.ArgumentParser.parse_args(/home/ubuntu/work/DeePKS-L/deepks/main.py)
-> load_runtime_config(/home/ubuntu/work/DeePKS-L/deepks/config/__init__.py)
   -> load_config(/home/ubuntu/work/DeePKS-L/deepks/config/loader.py)
      -> load_yaml(/home/ubuntu/work/DeePKS-L/deepks/io/utils.py)
   -> normalize_config(/home/ubuntu/work/DeePKS-L/deepks/config/normalize.py)
   -> validate_config(/home/ubuntu/work/DeePKS-L/deepks/config/validator.py)
   -> get_default_config(/home/ubuntu/work/DeePKS-L/deepks/config/defaults.py)
   -> merge_configs(/home/ubuntu/work/DeePKS-L/deepks/config/merger.py)
   -> package_config(/home/ubuntu/work/DeePKS-L/deepks/config/packager.py)
-> dispatch_command(/home/ubuntu/work/DeePKS-L/deepks/config/dispatcher.py)
   -> deepcopy(runtime_config["iterate_param"])(/home/ubuntu/work/DeePKS-L/deepks/config/dispatcher.py)
   -> _get_workflow_handler(/home/ubuntu/work/DeePKS-L/deepks/config/dispatcher.py)
   -> run_iterate_workflow(/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/workflow.py)
```

## 2. `run_iterate_workflow()` 调用栈

```text
run_iterate_workflow(/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/workflow.py)
-> prepare_iterate(/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/prepare.py)
-> os.path.exists(record_file)
-> iteration_workflow.run(/home/ubuntu/work/DeePKS-L/deepks/orchestration/workflow/workflow.py)
   or
   iteration_workflow.restart(/home/ubuntu/work/DeePKS-L/deepks/orchestration/workflow/workflow.py)
-> return {"final_model", "n_iterations", "workdir"}
```

## 3. `prepare_iterate()` 调用栈

```text
prepare_iterate(/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/prepare.py)
-> read packed iterate parameters:
   -> iterate_param["runtime"]
   -> iterate_param["data"]
   -> iterate_param["physics"]
   -> iterate_param["ml"]
   -> iterate_param["iterate"]
   -> iterate_param["iterate"]["tasks"]["main"]["scf"]
   -> iterate_param["iterate"]["tasks"]["main"]["train"]
-> prepare_iterate_snapshots(/home/ubuntu/work/DeePKS-L/deepks/interface/iterate/snapshots.py)
   -> if pyscf proj_basis is provided:
      -> load_basis(/home/ubuntu/work/DeePKS-L/deepks/physics/backends/pyscf/basis.py)
      -> save_basis(/home/ubuntu/work/DeePKS-L/deepks/physics/backends/pyscf/basis.py)
   -> _save_task_snapshot(..., "share/scf_abacus.yaml" or "share/scf_input.yaml")
   -> _save_task_snapshot(..., "share/train_input.yaml")
   -> optional _save_task_snapshot(..., "share/init_scf_abacus.yaml" or "share/init_scf.yaml")
   -> optional _save_task_snapshot(..., "share/init_train.yaml")
-> create_scf_step(/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/scf_step.py)
-> create_train_step(/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/train_step.py)
-> Iteration(/home/ubuntu/work/DeePKS-L/deepks/orchestration/workflow/workflow.py)
-> if iterate.use_init:
   -> create_scf_step(...)
   -> create_train_step(...)
   -> Sequence([scf_init, train_init], workdir="iter.init")
   -> iteration_workflow.prepend(init_iter)
   -> iteration_workflow.set_init_folder(...)
-> elif initial_model:
   -> iteration_workflow.set_init_folder(share_path)
-> return iteration_workflow, workdir, record_file
```

## 4. `iterate` 工作流树

`iterate.use_init: true` 时：

```text
Iteration(workdir=".")
-> Sequence(workdir="iter.init")
   -> Sequence(workdir="00.scf")
      -> PythonTask(convert_data, workdir=".")
      -> GroupBatchTask(...) or DPDispatcherTask(..., workdir=".")
      -> PythonTask(stat_data, workdir=".")
      -> optional ShellTask(cleanup, workdir=".")
   -> Sequence(workdir="01.train")
      -> BatchTask("deepks train_input.yaml", workdir=".")
      -> optional ShellTask(cleanup, workdir=".")
-> Sequence(workdir="iter.00")
   -> Sequence(workdir="00.scf")
   -> Sequence(workdir="01.train")
-> Sequence(workdir="iter.01")
   -> Sequence(workdir="00.scf")
   -> Sequence(workdir="01.train")
-> ...
```

`iterate.use_init: false` 时：

```text
Iteration(workdir=".")
-> Sequence(workdir="iter.00")
   -> Sequence(workdir="00.scf")
   -> Sequence(workdir="01.train")
-> Sequence(workdir="iter.01")
   -> Sequence(workdir="00.scf")
   -> Sequence(workdir="01.train")
-> ...
```

## 5. `iteration_workflow.run()` 调用栈

```text
Workflow.run(/home/ubuntu/work/DeePKS-L/deepks/orchestration/workflow/workflow.py)
-> for child in self.child_tasks
   -> task.run(curr_tag)
      -> if child is Workflow:
         -> Workflow.run(parent_tag=curr_tag)
      -> if child is AbstructTask:
         -> AbstructTask.run(/home/ubuntu/work/DeePKS-L/deepks/orchestration/workflow/task.py)
            -> preprocess()
            -> os.chdir(self.workdir)
            -> execute()
            -> os.chdir(self.olddir)
            -> postprocess()
-> write_record(curr_tag)
```

## 6. `iter.init/00.scf` 的构建调用栈

```text
prepare_iterate(/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/prepare.py)
-> create_scf_step(/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/scf_step.py)
   -> read packed scf payload:
      -> scf_param["physics"]["backend"]["input"]
      -> scf_param["runtime"]["scf"]["command"]
      -> scf_param["runtime"]["scf"]["execute"]
   -> build_abacus_iterate_scf_kwargs(/home/ubuntu/work/DeePKS-L/deepks/interface/iterate/task_params.py)
   -> make_scf_abacus(/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/abacus/sequence.py)
      -> make_convert_scf_abacus(/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/abacus/sequence.py)
         -> PythonTask(convert_data, ...)
      -> make_run_scf_abacus(/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/abacus/sequence.py)
         -> GroupBatchTask(...) or DPDispatcherTask(...)
      -> make_stat_scf_abacus(/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/abacus/sequence.py)
         -> PythonTask(... collect/stat ..., ...)
      -> optional make_cleanup(/home/ubuntu/work/DeePKS-L/deepks/interface/iterate/task_templates.py)
      -> Sequence([pre_scf_abacus, run_scf_abacus, post_scf_abacus, cleanup?], workdir="00.scf")
```

## 7. `iter.init/00.scf` 运行时调用栈

```text
Workflow.run(...)
-> Sequence(workdir="iter.init").run(tag=(0,))
   -> Sequence(workdir="iter.init/00.scf").run(tag=(0,0))
      -> PythonTask(convert_data).run(tag=(0,0,0))
         -> PythonTask.execute(...)
            -> convert_data(/home/ubuntu/work/DeePKS-L/deepks/physics/backends/abacus/iterate_ops.py)
               -> if model is needed:
                  -> load_runtime_model(/home/ubuntu/work/DeePKS-L/deepks/ml/model_io.py)
                  -> model.compile_save(...)
               -> coord_to_atom(...) if atom.npy missing
               -> make_abacus_scf_stru(...)
               -> make_abacus_scf_input(...)
               -> make_abacus_scf_kpt(...)
      -> GroupBatchTask.run(tag=(0,0,1))
         -> GroupBatchTask.execute(...)
            -> Dispatcher.run_jobs(...)
      -> PythonTask(stat_scf_abacus).run(tag=(0,0,2))
      -> optional ShellTask(cleanup).run(tag=(0,0,3))
```

## 8. `iter.init/00.scf` 中单个 frame 子任务的调用栈

```text
make_run_scf_abacus(/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/abacus/sequence.py)
-> GroupBatchTask(batch_tasks=[BatchTask(frame_0), BatchTask(frame_1), ...])
-> GroupBatchTask.execute(/home/ubuntu/work/DeePKS-L/deepks/orchestration/workflow/task.py)
   -> BatchTask.make_dict(base=self.workdir) for frame_i
   -> Dispatcher.run_jobs([task_dict_0, task_dict_1, ...], ...)
      -> submit_jobs(...)
      -> all_finished(...)
         -> batch.check_status()
         -> context.download(...)
```

单个 frame 的命令来源链：

```text
create_scf_step(...)
-> make_scf_abacus(...)
-> make_run_scf_abacus(...)
-> BatchTask(cmds=[f"cd <system>/ABACUS/<frame> && {run_cmd} -n {task_per_node} {abacus_path} ..."])
-> Dispatcher.run_jobs(...)
```

## 9. `iter.init/01.train` 的构建调用栈

```text
prepare_iterate(/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/prepare.py)
-> create_train_step(/home/ubuntu/work/DeePKS-L/deepks/workflows/iterate/train_step.py)
   -> make_train(/home/ubuntu/work/DeePKS-L/deepks/interface/iterate/task_templates.py)
      -> make_run_train(/home/ubuntu/work/DeePKS-L/deepks/interface/iterate/task_templates.py)
         -> make_train_task(/home/ubuntu/work/DeePKS-L/deepks/interface/iterate/task_templates.py)
            -> build_task_yaml(/home/ubuntu/work/DeePKS-L/deepks/interface/iterate/task_yaml.py)
            -> BatchTask("deepks train_input.yaml", ...)
```

## 10. `iter.init/01.train` 运行时调用栈

```text
Workflow.run(...)
-> Sequence(workdir="iter.init").run(tag=(0,))
   -> Sequence(workdir="iter.init/01.train").run(tag=(0,1))
      -> BatchTask("deepks train_input.yaml").run(tag=(0,1,0))
         -> BatchTask.preprocess(...)
            -> write_files["train_input.yaml"] -> train_input.yaml
         -> BatchTask.execute(...)
            -> Dispatcher.run_jobs(...)
               -> batch.submit(cmds=["deepks train_input.yaml"], ...)
```

`deepks train_input.yaml` 这个子命令的实际调用栈：

```text
main(/home/ubuntu/work/DeePKS-L/deepks/main.py)
-> load_runtime_config(/home/ubuntu/work/DeePKS-L/deepks/config/__init__.py)
   -> load_config(...)
   -> detect INTERNAL_PACKED_MARKER and return packed payload directly
-> dispatch_command(/home/ubuntu/work/DeePKS-L/deepks/config/dispatcher.py)
-> run_train_workflow(/home/ubuntu/work/DeePKS-L/deepks/workflows/train/workflow.py)
   -> prepare_train_runtime(/home/ubuntu/work/DeePKS-L/deepks/interface/train/prepare.py)
   -> run_training_stage(/home/ubuntu/work/DeePKS-L/deepks/interface/train/run.py)
      -> get_recipe(/home/ubuntu/work/DeePKS-L/deepks/interface/registry.py)
      -> recipe.create_or_load_model(/home/ubuntu/work/DeePKS-L/deepks/interface/recipes/corrnet_energy.py)
      -> recipe.preprocess_training_data(...)
      -> recipe.train_model(...)
         -> Trainer.train(/home/ubuntu/work/DeePKS-L/deepks/ml/train/trainer.py)
            -> ObjectiveAdapter.compute_losses(...)
               -> model.forward_with_derivatives(...)
               -> PropertyEngine.get_many(...)
               -> interface objective terms.compute_loss(...)
```

## 11. `iter.00/00.scf` 与后续 SCF 轮

与 `iter.init/00.scf` 同构，差别只在模型来源：

- `use_init: true`
  - `iter.00/00.scf` 默认使用 `iter.init/01.train/model.pth`
- `use_init: false` 且存在初始模型
  - `iter.00/00.scf` 使用外部初始模型
- `use_init: false` 且无初始模型
  - `iter.00/00.scf` 从无模型 SCF 开始

## 12. `iter.00/01.train` 与后续 train 轮

与 `iter.init/01.train` 同构，差别只在：

- `restart` 源文件不同
- 所用 packed 子任务快照来自 `iterate.tasks.main.train`

## 13. 一次完整 iterate 运算的总栈

```text
main
-> load_runtime_config
   -> load_config
   -> normalize_config
   -> validate_config
   -> get_default_config
   -> merge_configs
   -> package_config
-> dispatch_command
   -> run_iterate_workflow
      -> prepare_iterate
      -> Workflow.run
         -> optional iter.init/00.scf
            -> convert_data
            -> run_scf_abacus or run_pyscf
            -> stat_scf
         -> optional iter.init/01.train
            -> deepks train_input.yaml
               -> packed config fast-path
               -> run_train_workflow
                  -> Trainer.train
                     -> ObjectiveAdapter.compute_losses
                        -> model forward / autograd
                        -> PropertyEngine.get_many
                        -> interface objective terms.compute_loss
         -> iter.00/00.scf
         -> iter.00/01.train
         -> iter.01/00.scf
         -> iter.01/01.train
         -> ...
```
