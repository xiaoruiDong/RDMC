#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modules for computing metrics to decide when to stop generating conformers
"""

import numpy as np
from typing import List, Optional

R = 0.0019872  # kcal/(K*mol)


class SCGMetric:
    """
    A class to calculate and track the given metric ("entropy", "partition function", or "total conformers") for a molecule over time.

    Args:
        metric (str, optional): Metric to be calculated. Options are ``"entropy"``, ``"partition function"``, or ``"total conformers"``.
                                Defaults to ``"entropy"``.
        window (int, optional): Window size to compute the change in metric (doesn't work when the metric is "total conformers").
                                Defaults to ``5``.
        threshold (float, optional): Threshold for the change in metric to decide when to stop generating conformers.
                                     Defaults to ``0.01``.
        T (float, optional): Temperature for entropy or partition function calculations. Defaults to ``298`` K.
    """

    def __init__(self,
                 metric: Optional[str] = "entropy",
                 window: Optional[int] = 5,
                 threshold: Optional[float] = 0.01,
                 T: Optional[float] = 298,
                 ):
        """
        Generate an SCGMetric instance.

        Args:
            metric (str, optional): Metric to be calculated. Options are ``"entropy"``, ``"partition function"``, or ``"total conformers"``.
                                    Defaults to ``"entropy"``.
            window (int, optional): Window size to compute the change in metric (doesn't work when the metric is "total conformers").
                                    Defaults to ``5``.
            threshold (float, optional): Threshold for the change in metric to decide when to stop generating conformers.
                                        Defaults to ``0.01``.
            T (float, optional): Temperature for entropy or partition function calculations. Defaults to ``298`` K.
        """
        self.metric = metric
        self.window = window
        self.threshold = threshold
        self.T = T
        self.metric_history = []

    def calculate_metric(self,
                         mol_data: List[dict]):
        """
        Calculate the metric for a given molecule. The calculated value will be appended to the ``metric_history`` list.

        Args:
            mol_data (List[dict]): A list of dictionaries with molecule conformers.
        """
        if self.metric == "entropy":
            metric_val = self.calculate_entropy(mol_data)

        elif self.metric == "partition function":
            metric_val = self.calculate_partition_function(mol_data)

        elif self.metric == "total conformers":
            metric_val = len(mol_data)

        else:
            raise NotImplementedError(f"Metric {self.metric} is not supported.")

        self.metric_history.append(metric_val)

    def check_metric(self):
        """
        Check if the change in metric is below the threshold.

        Returns:
            bool: ``True`` if the change in metric is below the threshold, ``False`` otherwise.
        """
        if self.metric == "total conformers":
            return False
        else:
            min_metric = np.min(self.metric_history[-self.window:])
            max_metric = np.max(self.metric_history[-self.window:])
            change = (max_metric - min_metric) / np.clip(
                min_metric, a_min=1e-10, a_max=None
            )
            return True if change <= self.threshold else False

    def calculate_entropy(self,
                          mol_data: List[dict]):
        """
        Calculate the entropy of a molecule.

        Args:
            mol_data (List[dict]): A list of dictionaries with molecule conformers.
        """
        energies = np.array([c["energy"] for c in mol_data])
        energies = energies - energies.min()
        _prob = np.exp(-energies / (R * self.T))
        prob = _prob / _prob.sum()
        entropy = -R * np.sum(prob * np.log(prob))
        return entropy

    def calculate_partition_function(self,
                                     mol_data: List[dict]):
        """
        Calculate the partition function of a molecule.

        Args:
            mol_data (List[dict]): A list of dictionaries with molecule conformers.
        """
        energies = np.array([c["energy"] for c in mol_data])
        energies = energies - energies.min()
        prob = np.exp(-energies / (R * self.T))
        partition_fn = 1 + prob.sum()
        return partition_fn
