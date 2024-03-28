from pathlib import Path
import shutil
from typing import Optional
import tempfile
from time import time

from abc import ABC, abstractmethod


class BaseTask(ABC):

    def __init__(self, track_stats: bool = False, *args, **kwargs):
        """
        Base class for all tasks.
        Args:
            track_stats (bool): whether to track the task stats. Default: False.
        """
        self.track_stats = track_stats
        self.stats = []

    @abstractmethod
    def is_available(self):
        """
        Checks if the dependency requirement is fulfilled to use the task. As a guidance, all child
        """
        raise NotImplementedError(
            "The is_available method needs to be implemented in the child class."
        )

    def check_availability(self):
        """
        Checks if the dependency requirement is fulfilled to use the task. As a guidance, all child
        """
        assert (
            self.is_available()
        ), f"The task ({self.__class__.__name__}) is not currently available due to the dependency requirement doesn't meet."
        f"Please double check if the all dependencies are installed successfully. If you still encounte this error message"
        f"after installing all dependencies, please post an issue at https://github.com/xiaoruiDong/RDMC/issues/."

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Runs the task.
        """
        raise NotImplementedError(
            "The run method needs to be implemented in the child class."
        )

    def update_work_and_save_dir(
        self,
        work_dir: Optional[str] = None,
        save_dir: Optional[str] = None,
    ):
        self.save_dir = Path(save_dir).absolute() if save_dir is not None else None
        self.work_dir = Path(work_dir).absolute() if work_dir is not None else None

        if self.save_dir is None and self.work_dir is None:
            self.work_dir = Path(tempfile.mkdtemp())

        elif self.save_dir is not None and self.work_dir is None:
            self.work_dir = self.save_dir

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        if self.work_dir and not self.work_dir.is_dir():
            self.work_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, *args, **kwargs):

        self.update_work_and_save_dir(
            kwargs.pop("work_dir", None),
            kwargs.pop("save_dir", None),
        )

        time_start = time()

        results = self.run(*args, **kwargs)

        if self.track_stats:
            time_end = time()
            self.update_stats(
                time_end - time_start,
                results,
                *args,
                **kwargs,
            )

        self.copy_work_dir_to_save_dir()

        return results

    def copy_work_dir_to_save_dir(self):

        if self.save_dir is not None and (
            self.work_dir.resolve() != self.save_dir.resolve()
        ):
            shutil.copytree(self.work_dir, self.save_dir, dirs_exist_ok=True)
            shutil.rmtree(self.work_dir)

    def update_stats(self, exe_time: float, results, *args, **kwargs):
        """
        Update the task stats. The default behavior is recording the execution time of the task.
        """
        stats = {"time": exe_time}
        self.stats.append(stats)
