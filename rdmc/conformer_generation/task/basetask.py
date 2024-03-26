from pathlib import Path
from typing import Optional
import tempfile
from time import time

from abc import ABC, abstractmethod


class BaseTask(ABC):

    def __init__(self, track_stats: bool = False):
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
            self.work_dir = tempfile.mkdtemp()
        elif self.save_dir is not None and self.work_dir is None:
            self.work_dir = self.save_dir

    def __call__(self, *args, **kwargs):
        time_start = time()

        self.update_work_and_save_dir(
            kwargs.get("work_dir"),
            kwargs.get("save_dir"),
        )

        results = self.run(*args, **kwargs)

        if self.track_stats:
            time_end = time()
            stats = {"time": time_end - time_start}
            self.stats.append(stats)

        return results
