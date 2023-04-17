#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""This is the module for abstracting the conformer generation task"""

import time


class Task(object):

    # A list of external software required by the task
    request_external_software = []

    def __init__(self,
                 track_stats=False,
                 save_dir=None,
                 work_dir=None,
                 *args,
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

        if self.request_external_software:
            self.check_external_software()

        self.task_prep(*args, **kwargs)

    # check if the external software is available
    def check_external_software(self):
        """
        Check if the external software is available.
        """
        return True

    def task_prep(self, *args, **kwargs):
        """
        Prepare the task.
        """
        return True

    @property
    def last_run_time(self):
        """
        The time of the last run of the task
        """
        try:
            return self._last_run_time
        except AttributeError:
            raise RuntimeError("The task has not been run yet.")

    def run_timer_and_counter(func):
        """
        Timer decorator for recording the time of a function.
        """
        def wrapper(self, *args, **kwargs):
            time_start = time.time()
            result = func(self, *args, **kwargs)
            time_end = time.time()
            self._last_run_time = time_end - time_start
            self.iter += 1
            return result
        return wrapper

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
        except:
            RuntimeError()


    @n_success.setter
    def n_success(self, n: int):
        """
        Set the number of successful subtasks.

        Args:
            n (int): The number of successful subtasks.
        """
        self._n_success = n

    def count_success(self):
        """
        Count the number of successful subtasks.
        """
        raise NotImplementedError

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
        self.count_success()

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

    @run_timer_and_counter
    def task(self,
             test: bool = False,
             *args,
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

    def __call__(self, *args, **kwargs):
        """
        Run the task.
        """

        result = self.task(*args, **kwargs)

        if self.save_dir:
            self.save_data()

        if self.track_stats:
            self.update_stats()

        return result