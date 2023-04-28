#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from rdmc.conformer_generation.optimizer.base import IOOptimizer
from rdmc.conformer_generation.task import ORCABaseTask


class ORCAOptimizer(ORCABaseTask, IOOptimizer):
    """
    The class to optimize geometries using the algorithm built in ORCA.
    You have to have the Orca package installed to run this optimizer.

    Args:
        method (str, optional): The method available in ORCA to be used for optimization.
                                If you want to use XTB methods, you need to put the xtb binary
                                into the Orca directory. Defaults to XTB2.
        nprocs (int, optional): The number of processors to use. Defaults to 1.
        memory (int, optional): The memory to use in GB. Defaults to 1.
    """

    subtask_dir_name = 'orca_opt'
    files = {'input_file': 'orca_opt.inp',
             'log_file': 'orca_opt.log',
             'opt_file': 'orca_opt.opt'}
    keep_files = ['orca_opt.inp', 'orca_opt.log', 'orca_opt.opt']
    create_mol_flag = True
    init_attrs = {'energies': np.nan, 'frequencies': None}
    calc_type = 'opt'

    def save_data(self, **kwargs):
        """
        Save the data.
        """
        super(ORCABaseTask, self).save_data(**kwargs)  # from IOOptimizer

    def post_run(self, **kwargs):
        """
        Setting the success information, also set the energy to the
        conformers. Remove temporary directory if necessary.
        """
        super(ORCABaseTask, self).post_run(**kwargs)  # from IOOptimizer

    def analyze_subtask_result(self,
                               mol: 'RDKitMol',
                               subtask_id: int,
                               subtask_result: tuple,
                               **kwargs):
        """
        Analyze the subtask result. This method will parse the number of optimization
        cycles and the energy from the ORCA output file and set them to the molecule.
        """
        log = self.logparser(self.paths['log_file'][subtask_id])
        # 1. Parse coordinates
        if log.success:
            mol.SetPositions(log.all_geometries[-1],
                             id=subtask_id)
            mol.GetConformer(subtask_id).SetIntProp('n_opt_cycles',
                                                    log.optstatus.shape[0] - 1)
            # Not using more preferred
            # mol.SetPositions(log.converged_geometries[-1], id=subtask_id)
            # ORCA treats the following as converged job as well while not fulfilling
            # all of the convergence criteria. This is currently treated as non-converged
            # by the ORCALog parser. Therefore, we need to catch this exception.
            #
            #                         .--------------------.
            #   ----------------------|Geometry convergence|-------------------------
            #   Item                value                   Tolerance       Converged
            #   ---------------------------------------------------------------------
            #   Energy change      -0.0000139792            0.0000050000      NO
            #   RMS gradient        0.0000407031            0.0001000000      YES
            #   MAX gradient        0.0000648215            0.0003000000      YES
            #   RMS step            0.0000999215            0.0020000000      YES
            #   MAX step            0.0001589253            0.0040000000      YES
            #   ........................................................
            #   Max(Bonds)      0.0001      Max(Angles)    0.00
            #   Max(Dihed)        0.00      Max(Improp)    0.00
            #   ---------------------------------------------------------------------
            #
            #    Everything but the energy has converged. However, the energy
            #    appears to be close enough to convergence to make sure that the
            #    final evaluation at the new geometry represents the equilibrium energy.
            #    Convergence will therefore be signaled now
            #
            # todo: 1. wait until updates in cclib. or
            # todo: 2. in ORCALog force the last geometry to be converged if the following is detected.
            # todo: 3. evaluate whether to put analyze subtask result in the iooptimizer, so that
            # todo:    no individual definition is needed in gaussian/orca/qchem optimizer.
            #   ***********************HURRAY********************
            #   ***        THE OPTIMIZATION HAS CONVERGED     ***
            #   *************************************************
        else:  # not log.success
            mol.keep_ids[subtask_id] = False
            print(f'Error in optimizing the geometry of conformer {subtask_id} in {self.label}')
            return
        # 2. Parse Energy
        try:
            mol.energies[subtask_id] = log.get_scf_energies(relative=False)[-1].item()
        except AttributeError:
            # Unable to parse the energies of the xTB methods from the log file.
            # use .opt file instead.
            mol.energies[subtask_id] = \
                    parse_energy_from_opt_file(self.paths['opt_file'][subtask_id])
        mol.frequencies[subtask_id] = log.freqs


def parse_energy_from_opt_file(path: str):
    """
    Parse the energy from the opt file.

    Args:
        path (str): The path to the opt file (e.g., orca_opt.opt).
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line == '$energies':
            # E.g.,
            # $energies
            # 4
            # -4.1738410246499997
            # -4.1750652237499999
            # -4.1751853899600002
            # -4.1752182364700001
            n_iter = int(lines[i + 1].strip())
            return float(lines[i + n_iter + 1].strip()) * 627.509  # hartree to kcal/mol
