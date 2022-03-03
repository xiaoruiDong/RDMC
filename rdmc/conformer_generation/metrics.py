#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Modules for computing metrics to decide when to stop generating conformers
"""

import numpy as np
R = 0.0019872  # kcal/(K*mol)


class SCGMetric:
    def __init__(self, metric="entropy", window=5, threshold=0.01, T=298):
        self.metric = metric
        self.window = window
        self.threshold = threshold
        self.T = T
        self.metric_history = []

    def calculate_metric(self, mol_data):

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

        min_metric = np.min(self.metric_history[-self.window:])
        max_metric = np.max(self.metric_history[-self.window:])
        change = (max_metric - min_metric) / np.clip(min_metric, a_min=1e-10, a_max=None)
        return True if change <= self.threshold else False

    def calculate_entropy(self, mol_data):

        energies = np.array([c["energy"] for c in mol_data])
        energies = energies - energies.min()
        _prob = np.exp(-energies / (R * self.T))
        prob = _prob / _prob.sum()
        entropy = -R * np.sum(prob * np.log(prob))
        return entropy

    def calculate_partition_function(self, mol_data):

        energies = np.array([c["energy"] for c in mol_data])
        energies = energies - energies.min()
        prob = np.exp(-energies / (R * self.T))
        partition_fn = 1 + prob.sum()
        return partition_fn
