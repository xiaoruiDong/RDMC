#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This is the module for abstracting the conformer generation task"""

import time
from typing import Optional

from rdmc.conformer_generation.utils import _software_available


class Task(object):

    # A list of external software required by the task
    request_external_software = []

    def __init__(self,
                 track_stats: bool = False,
                 save_dir: Optional[str] = None,
                 work_dir: Optional[str] = None,
                 iter: int = 0,
                 **kwargs,
                 ):
        """
        Initialize the task.

        Args:
            track_stats (bool, optional): Whether to track the statistics of the task.
                                          Defaults to False.
            save_dir (str, optional): The directory to save the data.
            work_dir (str, optional): The directory to store the intermediate data.
        """
        self.track_stats = track_stats
        self.save_dir = save_dir
        self.work_dir = work_dir
        self.iter = iter

        # if both save_dir and work_dir are None, the task will be run in a tempdir
        if self.save_dir is None and self.work_dir is None:
            self.run_in_tempdir = True
        else:
            self.run_in_tempdir = False

        if self.request_external_software:
            self.check_external_software()

        self.task_prep(**kwargs)

    # check if the external software is available
    def check_external_software(self):
        """
        Check if the external software is available.
        """
        if not all(_software_available[s] for s in self.request_external_software):
            raise RuntimeError(
                    f"The software requirement "
                    f"({', '.join(self.request_external_software)}) "
                    f"is not met. Please install the software and try again.")

    def task_prep(self, **kwargs):
        """
        Prepare the task.
        """
        return True

    def timer(func):
        """
        Timer decorator for recording the time of a function.

        Args:
            func (function): The function to be decorated.

        Returns:
            function: The decorated function.
        """
        def wrapper(self, *args, **kwargs):
            time_start = time.time()
            result = func(self, *args, **kwargs)
            time_end = time.time()
            self._last_run_time = time_end - time_start
            return result
        return wrapper

    @property
    def last_run_time(self):
        """
        The time of the last run of the task
        """
        try:
            return self._last_run_time
        except AttributeError:
            raise RuntimeError("The task has not been run yet.")

    @property
    def n_subtasks(self):
        """
        The number of subtasks.
        """
        try:
            return self._n_subtasks
        except AttributeError:
            return 1

    @n_subtasks.setter
    def n_subtasks(self, n: int):
        """
        Set the number of subtasks.

        Args:
            n (int): The number of subtasks.
        """
        self._n_subtasks = n

    @property
    def n_success(self):
        """
        The number of successful subtasks.
        """
        try:
            return self._n_success
        except AttributeError:
            return 0

    @n_success.setter
    def n_success(self, n: int):
        """
        Set the number of successful subtasks.

        Args:
            n (int): The number of successful subtasks.
        """
        self._n_success = n

    @property
    def percent_success(self):
        """
        The percentage of successful subtasks.

        Returns:
            float: The percentage of successful subtasks.
        """
        return self.n_success / self.n_subtasks * 100

    def prepare_stats(self,):
        """
        Prepare the common statistics of the task. Ideally, this function should
        not be modified. Adding more statistics should be done in the
        `prepare_extra_stats` function.

        Returns:
            dict: The common statistics of the task.
        """
        stats = {"iter": self.iter,
                 "time": self.last_run_time,
                 "n_success": self.n_success,
                 "percent_success": self.percent_success}
        return stats

    def prepare_extra_stats(self):
        """
        Prepare the extra statistics of the task. Developers can add more statistics
        for a specific task in this function.

        Returns:
            dict: The extra statistics of the task.
        """
        return {}

    def update_stats(self):
        """
        Update the statistics of the task.
        """
        stats = self.prepare_stats()
        stats.update(self.prepare_extra_stats())
        self.stats.append(stats)

    def save_data(self):
        """
        Save the data of the task.
        """
        raise NotImplementedError

    @timer
    def run(self,
            test: bool = False,
            **kwargs):
        """
        The main task. This function should be implemented by the developer.
        Please note that the `run_timer_and_counter` decorator is added to this
        function to record the time of the task and the iteration number. The function
        should return the result of the task.
        """
        if test:
            return True
        raise NotImplementedError

    def pre_run(self,
                **kwargs):
        """
        The function to be executed before the task is run.
        """

    def post_run(self,
                 **kwargs):
        """
        The function to be executed after the task is run.
        """

    def __call__(self, **kwargs):
        """
        Run the task.
        """
        self.pre_run(**kwargs)
        self.last_result = self.run(**kwargs)
        self.post_run(**kwargs)

        if self.save_dir:
            self.save_data()

        if self.track_stats:
            self.update_stats()

        return self.last_result
