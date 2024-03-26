from abc import ABC, abstractmethod


class BaseTask(ABC):

    def __init__(self, track_stats: bool = False):
        """
        Base class for all tasks.
        Args:
            track_stats (bool): whether to track the task stats. Default: False.
        """
        self.track_stats = track_stats

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
