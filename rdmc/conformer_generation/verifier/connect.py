#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdmc import RDKitMol
from rdmc.conformer_generation.task import MolTask


class ConnectivityVerifier(MolTask):
    """
    The connectivity verifier check if the coordinates of the molecules represents the same graph
    as its owning molecule. This is only used for non-TS molecules. The connectivity verifier
    assumes the molecule and the coordinates has the same atom mapping, and neglect the bond order
    difference when comparing the connectivity (to avoid the problem related to aromatic perception
    / bond conjugation).

    Args:
        backend(str, optional): The backend used for xyz perception. Defaults to 'openbabel'. 'jensen'
                                is also available.
    """

    def task_prep(self,
                  backend: str = 'openbabel',
                  **kwargs,
                  ):
        """
        Set the backend.

        Args:
            backend(str, optional): The backend used for xyz perception. Defaults
                                    to 'openbabel'. 'jensen' is also available.
        """
        self.backend = backend
        super().task_prep(**kwargs)

    @MolTask.timer
    def run(self,
            mol: 'RDKitMol',
            **kwargs):
        """
        """
        # No need to check connectivity for uni-atom molecules
        if mol.GetNumAtoms() == 1:
            return

        for subtask_id in self.run_ids:
            # Check if the connectivity is correct
            if not mol.HasSameConnectivity(confId=subtask_id,
                                           backend=self.backend,
                                           **kwargs):
                mol.keep_ids[subtask_id] = False

        return mol
