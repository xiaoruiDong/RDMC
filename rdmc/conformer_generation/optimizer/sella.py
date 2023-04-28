#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from contextlib import redirect_stdout
import io
import traceback

from ase.calculators.orca import ORCA
import numpy as np

from rdmc import RDKitMol
from rdmc.conformer_generation.optimizer.base import IOOptimizer
from rdmc.conformer_generation.utils import register_software

with register_software('sella'):
    from sella import Sella

with register_software('xtb-python'):
    from xtb.ase.calculator import XTB
    from xtb.utils import get_method


class SellaOptimizer(IOOptimizer):
    """
    The class to optimize TS geometries using the Sella algorithm.
    It uses XTB as the backend calculator, ASE as the interface, and Sella module from the Sella repo.

    Args:
        method (str, optional): The method in XTB used to optimize the geometry. Options are 'GFN1-xTB' and 'GFN2-xTB'. Defaults to "GFN2-xTB".
        fmax (float, optional): The force threshold used in the optimization. Defaults to 1e-3.
        steps (int, optional): Max number of steps allowed in the optimization. Defaults to 1000.
    """

    request_external_software = ['sella', 'xtb-python']
    subtask_dir_name = 'sella_opt'
    files = {'traj_file': 'sella_opt.traj',
             'log_file': 'sella_opt.log',
             'common_name': 'sella_opt'}
    keep_files = ['sella_opt.traj', 'sella_opt.log']
    create_mol_flag = True
    init_attrs = {'energies': np.nan, 'frequencies': None}

    def task_prep(self,
                  method: str = "GFN2-xTB",
                  fmax: float = 1e-3,
                  steps: int = 1000,):
        """
        Set up the Sella optimizer.
        """
        self.is_xtb_calc = True if get_method(method) is not None else False
        self.method = get_method(method) or method
        self.fmax = fmax
        self.steps = steps

    def write_input_file(self, **kwargs):
        """
        No input file is needed.
        """
        return

    def mol_to_atoms(self,
                     mol: 'RDKitMol',
                     subtask_id: int,):
        """
        Convert an RDKitMol object to an ASE atoms object, and
        set up the calculator.
        """
        atoms = mol.ToAtoms(confId=subtask_id)
        atoms.set_initial_magnetic_moments(
                    [atom.GetNumRadicalElectrons() + 1
                     for atom in mol.GetAtoms()])
        atoms.set_initial_charges(
                    [atom.GetFormalCharge()
                     for atom in mol.GetAtoms()])
        if self.is_xtb_calc:
            atoms.calc = XTB(method=self.method)
        else:
            atoms.calc = ORCA(label=self.paths['common_name'][subtask_id],
                              orcasimpleinput=self.method)
        return atoms

    def runner(self,
               mol: 'RDKitMol',
               subtask_id: int,
               **kwargs):
        """
        Run the Sella optimization.
        """
        # 1. Convert the mol to atoms
        atoms = self.mol_to_atoms(mol=mol, subtask_id=subtask_id)
        # 2. Run the Sella optimization
        with io.StringIO() as buf, redirect_stdout(buf):
            opt = Sella(atoms,
                        logfile=self.paths['log_file'][subtask_id],
                        trajectory=self.paths['traj_file'][subtask_id],
                        )
            opt.run(self.fmax, self.steps)
        return opt

    def analyze_subtask_result(self,
                               mol: 'RDKitMol',
                               subtask_id: int,
                               subtask_result: tuple,
                               **kwargs):
        """
        Analyze the subtask result. This method will parse the number of optimization
        cycles and the energy from the Sella output file and set them to the molecule.

        Note, n_opt_cycles and energy parsing hasn't been tested yet.
        """
        opt = subtask_result  # better readability
        mol.SetPositions(opt.atoms.positions, id=subtask_id)
        try:
            mol.GetConformer(subtask_id).SetIntProp('n_opt_cycles', opt.nsteps)
            mol.energies[subtask_id] = opt.atoms.get_potential_energy()
        except Exception as exc:
                print(f'Sella optimization finished but result parsing failed:\n{exc}')
                traceback.print_exc()
