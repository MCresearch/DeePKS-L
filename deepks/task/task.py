import os
from pathlib import Path
from deepks.utils import link_file, copy_file, create_dir
from deepks.utils import check_list
from deepks.utils import get_abs_path

import sys
import subprocess as sp
from copy import deepcopy
from contextlib import nullcontext, redirect_stdout, redirect_stderr
from deepks.task.job.dispatcher import Dispatcher


__all__ = ["BlankTask", "PythonTask", "ShellTask", "BatchTask", "GroupBatchTask", "DPDispatcherTask"]


class AbstructStep(object):
    '''
    An abstract class for a single step in the pipeline.
    '''
    def __init__(self, workdir):
        self.workdir = Path(workdir)
        
    def __repr__(self):
        return f'<{type(self).__module__}.{type(self).__name__} with workdir: {self.workdir}>'
        
    def run(self, *args, **kwargs):
        raise NotImplementedError
    
    def prepend_workdir(self, path):
        self.workdir = path / self.workdir
        
    def append_workdir(self, path):
        self.workdir = self.workdir / path


class AbstructTask(AbstructStep):
    '''
    An abstract class for a task in the pipeline.
    The key methods are preprocess, execute, and postprocess, corresponding to the three stages of a task.
    Parameters:
    - workdir: the working directory (relative) of the task.
    - backup: whether to create the working directory at parent directory.
    - *prev*: the previous task or folder, used for linking or copying files.
    - *share*: the shared folder, used for linking or copying files.
    - *abs*: the absolute path, used for linking or copying files.
    '''
    def __init__(self, workdir='.', backup=False, prev_task=None,
                 prev_folder=None, link_prev_files=None, copy_prev_files=None, 
                 share_folder=None, link_share_files=None, copy_share_files=None,
                 link_abs_files=None, copy_abs_files=None):
        # workdir has to be relative in order to be chained
        # prev_task is dereferenced to folder dynamically.
        # folders are absolute.
        super().__init__(workdir)
        self.backup = backup
        assert prev_task is None or prev_folder is None
        self.prev_task = prev_task
        self.prev_folder = get_abs_path(prev_folder)
        self.share_folder = get_abs_path(share_folder)
        self.link_prev_files = check_list(link_prev_files)
        self.copy_prev_files = check_list(copy_prev_files)
        self.link_share_files = check_list(link_share_files)
        self.copy_share_files = check_list(copy_share_files)
        self.link_abs_files = check_list(link_abs_files)
        self.copy_abs_files = check_list(copy_abs_files)
        
    def preprocess(self):
        '''
        Create the working directory and link or copy files from previous tasks or shared folders.
        '''
        create_dir(self.workdir, self.backup)
        if self.prev_folder is None and (self.link_prev_files or self.copy_prev_files):
            self.prev_folder = self.prev_task.workdir
        for f in self.link_prev_files:
            (fsrc, fdst) = (f, f) if isinstance(f, str) else f
            link_file(self.prev_folder / fsrc, self.workdir / fdst)
        for f in self.copy_prev_files:
            (fsrc, fdst) = (f, f) if isinstance(f, str) else f
            copy_file(self.prev_folder / fsrc, self.workdir / fdst)
        for f in self.link_share_files:
            (fsrc, fdst) = (f, f) if isinstance(f, str) else f
            link_file(self.share_folder / fsrc, self.workdir / fdst)
        for f in self.copy_share_files:
            (fsrc, fdst) = (f, f) if isinstance(f, str) else f
            copy_file(self.share_folder / fsrc, self.workdir / fdst)
        for f in self.link_abs_files:
            (fsrc, fdst) = (f, os.path.basename(f)) if isinstance(f, str) else f
            link_file(fsrc, self.workdir / fdst, use_abs=True)
        for f in self.copy_abs_files:
            (fsrc, fdst) = (f, os.path.basename(f)) if isinstance(f, str) else f
            copy_file(fsrc, self.workdir / fdst)
    
    def execute(self):
        raise NotImplementedError

    def postprocess(self):
        pass
        
    def run(self, *args, **kwargs):
        self.preprocess()
        self.olddir = os.getcwd()
        os.chdir(self.workdir)
        self.execute()
        os.chdir(self.olddir)
        self.postprocess()
    
    def set_prev_task(self, task):
        assert isinstance(task, AbstructTask)
        self.prev_folder = None
        self.prev_task = task

    def set_prev_folder(self, path):
        self.prev_task = None
        self.prev_folder = path


class BlankTask(AbstructTask):
    '''
    A blank task that does nothing.
    '''
    def execute(self):
        pass


class PythonTask(AbstructTask):
    '''
    A task that runs a Python callable.
    Parameters:
    - pycallable: the Python callable to run.
    - call_args: the positional arguments to pass to the callable.
    - call_kwargs: the keyword arguments to pass to the callable.
    - outlog: the file to redirect stdout to.
    - errlog: the file to redirect stderr to.
    '''
    def __init__(self, pycallable, 
                 call_args=None, call_kwargs=None, 
                 outlog=None, errlog=None,
                 **task_args):
        super().__init__(**task_args)
        self.pycallable = pycallable
        self.call_args = call_args if call_args is not None else []
        self.call_kwargs = call_kwargs if call_kwargs is not None else {}
        self.outlog = outlog
        self.errlog = errlog
    
    def execute(self):
        with (open(self.outlog, 'w', 1) if self.outlog is not None 
                else nullcontext(sys.stdout)) as fo, \
             redirect_stdout(fo), \
             (open(self.errlog, 'w', 1) if self.errlog is not None 
                else nullcontext(sys.stderr)) as fe, \
             redirect_stderr(fe):
            self.pycallable(*self.call_args, **self.call_kwargs)


class ShellTask(AbstructTask):
    '''
    A task that runs a shell command.
    Parameters:
    - cmd: the shell command to run.
    - env: the environment variables to set.
    - outlog: the file to redirect stdout to.
    - errlog: the file to redirect stderr to.
    '''
    def __init__(self, cmd, env=None,
                 outlog=None, errlog=None,
                 **task_args):
        super().__init__(**task_args)
        self.cmd = cmd
        self.env = env
        self.outlog = outlog
        self.errlog = errlog

    def execute(self):
        with (open(self.outlog, 'w', 1) if self.outlog is not None 
                else nullcontext()) as fo, \
             (open(self.errlog, 'w', 1) if self.errlog is not None 
                else nullcontext()) as fe:
            sp.run(self.cmd, env=self.env, shell=True, stdout=fo, stderr=fe)


class BatchTask(AbstructTask):
    '''
    A task that runs a batch of commands. Supports different dispatchers.
    Parameters:
    - cmds: the list of shell commands to run.
    - dispatcher: the dispatcher to use.
    - resources: the resources to request.
    - outlog: the file to redirect stdout to.
    - errlog: the file to redirect stderr to.
    - *_files: the files to forward or backward.
    '''
    def __init__(self, cmds, 
                 dispatcher=None, resources=None, 
                 outlog='log', errlog='err', 
                 forward_files=None, backward_files=None,
                 **task_args):
        super().__init__(**task_args)
        self.cmds = check_list(cmds)
        if dispatcher is None:
            dispatcher = Dispatcher()
        elif isinstance(dispatcher, dict):
            dispatcher = Dispatcher(**dispatcher)
        assert isinstance(dispatcher, Dispatcher)
        self.dispatcher = dispatcher
        self.resources = resources
        self.outlog = outlog
        self.errlog = errlog
        self.forward_files = check_list(forward_files)
        self.backward_files = check_list(backward_files)
    
    def execute(self):
        tdict = self.make_dict(base=self.workdir)
        self.dispatcher.run_jobs([tdict], group_size=1, work_path='.', 
                                 resources=self.resources, forward_task_deref=True,
                                 outlog=self.outlog, errlog=self.errlog)

    def make_dict(self, base='.'):
        return {'dir': str(self.workdir.relative_to(base)),
                'cmds': self.cmds,
                "resources": self.resources,
                'forward_files': self.forward_files,
                'backward_files': self.backward_files}


class GroupBatchTask(AbstructTask):
    '''
    A task that groups a batch of tasks. Supports different dispatchers.
    Parameters:
    - batch_tasks: the list of tasks to group.
    - group_size: the size of each group.
    - ingroup_parallel: the degree of parallelism within each group.
    - [others]: the same as BatchTask.
    '''
    # after grouping up, the following individual setting would be ignored:
    # dispatcher, outlog, errlog
    # only grouped one setting in this task would be effective
    def __init__(self, batch_tasks, group_size=1, ingroup_parallel=1,
                 dispatcher=None, resources=None, 
                 outlog='log', errlog='err', 
                 forward_files=None, backward_files=None,
                 **task_args):
        super().__init__(**task_args)            
        self.batch_tasks = [deepcopy(task) for task in batch_tasks]
        for task in self.batch_tasks:
            assert isinstance(task, BatchTask), f'given task is instance of {task.__class__}'
            assert not task.workdir.is_absolute()
            task.prepend_workdir(self.workdir)
            if task.prev_folder is None:
                task.prev_folder = self.prev_folder
        self.group_size = group_size
        self.para_deg = ingroup_parallel
        if dispatcher is None:
            dispatcher = Dispatcher()
        elif isinstance(dispatcher, dict):
            dispatcher = Dispatcher(**dispatcher)
        assert isinstance(dispatcher, Dispatcher)
        self.dispatcher = dispatcher
        self.resources = resources
        self.outlog = outlog
        self.errlog = errlog
        self.forward_files = check_list(forward_files)
        self.backward_files = check_list(backward_files)

    def execute(self):
        tdicts = [t.make_dict(base=self.workdir) for t in self.batch_tasks]
        self.dispatcher.run_jobs(tdicts, group_size=self.group_size, para_deg=self.para_deg,
                                 work_path='.', resources=self.resources, 
                                 forward_task_deref=False,
                                 forward_common_files=self.forward_files,
                                 backward_common_files=self.backward_files,
                                 outlog=self.outlog, errlog=self.errlog)
    
    def preprocess(self):
        # if (self.workdir / 'fin.record').exists():
        #     return
        super().preprocess()
        for t in self.batch_tasks:
            t.preprocess()

    def prepend_workdir(self, path):
        super().prepend_workdir(path)
        for t in self.batch_tasks:
            t.prepend_workdir(path)
    
    def set_prev_task(self, task):
        super().set_prev_task(task)
        for t in self.batch_tasks:
            t.set_prev_task(task)
    
    def set_prev_folder(self, path):
        super().set_prev_folder(path)
        for t in self.batch_tasks:
            t.set_prev_folder(path)


class DPDispatcherTask(AbstructTask):
    '''
    A task that dispatches a batch of tasks to Bohrium(a calculation platform).
    '''
    # after grouping up, the following individual setting would be ignored:
    # dispatcher, outlog, errlog
    # only grouped one setting in this task would be effective
    def __init__(self, task_list, work_base, 
                 group_size=1, 
                 resources=None, machine=None,
                 outlog='log', errlog='err', 
                 forward_files=None, backward_files=None,
                 **task_args):
        super().__init__(**task_args)            
        self.group_size = group_size
        self.task_list = task_list
        self.machine = machine
        self.resources = resources
        self.work_base = work_base
        self.outlog = outlog
        self.errlog = errlog
        self.forward_files = check_list(forward_files)
        self.backward_files = check_list(backward_files)

    def execute(self):
        from dpdispatcher import Machine, Resources, Submission
        submission=Submission(
            work_base=self.work_base,
            machine=Machine.load_from_dict(self.machine),
            resources=Resources.load_from_dict(self.resources),
            task_list=self.task_list,
            forward_common_files=self.forward_files,
            backward_common_files=self.backward_files)
        submission.run_submission()