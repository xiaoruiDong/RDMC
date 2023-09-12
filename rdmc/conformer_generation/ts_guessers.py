#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modules for providing transition state initial guess geometries.
"""
# RDKit import first to avoid some import or runtime issues
# TODO: Details to be added.
from rdkit import Chem

import os
import subprocess
import tempfile
from time import time
from typing import Optional

# Use PyTorch and PyTorch-Geometric for ML methods
import numpy as np
# import torch
# from torch_geometric.data import Batch

# Use [pygsm-gaussian](https://pypi.org/project/pygsm-gaussian/) for DE-GSM calculation
from rdmc.external.inpwriter import write_gaussian_gsm
# Uses XTB binary for the RMSD-PP method
from rdmc.external.xtb_tools.opt import run_xtb_calc

# Import TS-EGNN
_ts_egnn_avail = True
try:
    from ts_ml.trainers.ts_egnn_trainer import LitTSModule
    from ts_ml.dataloaders.ts_egnn_loader import TSDataset

    class EvalTSDataset(TSDataset):
        def __init__(self, config):
            self.mols = []
            self.no_shuffle_mols = True  # randomize which is reactant/product
            self.no_mol_prep = False  # prep as if starting from SMILES
            self.set_similar_mols = False  # use species (r/p) which is more similar to TS as starting mol
            self.product_loss = False
            self.prod_feat = config["prod_feat"]  # whether product features include distance or adjacency
except ImportError:
    _ts_egnn_avail = False
    print("No TS-EGNN installation detected. Skipping import...")

# Import TS_GCN
_ts_gcn_avail = True
try:
    from ts_ml.trainers.ts_gcn_trainer import LitTSModule as LitTSGCNModule
    from ts_ml.dataloaders.ts_gcn_loader import TSGCNDataset

    class EvalTSGCNDataset(TSGCNDataset):
        def __init__(self, config):
            self.no_shuffle_mols = True  # randomize which is reactant/product
            self.no_mol_prep = False  # prep as if starting from SMILES
except ImportError:
    _ts_gcn_avail = False
    print("No TS-GCN installation detected. Skipping import...")

# Use ASE for the AutoNEB method
_ase_avail = True
try:
    from ase import Atoms
    from ase.autoneb import AutoNEB
    from ase.calculators.calculator import CalculationFailed, Calculator
    from ase.optimize import QuasiNewton
except BaseException:
    _ase_avail = False
    print("No ASE installation detected. Skipping import...")
# Use xtb ase calculator defined in the xtb-python module
try:
    from xtb.ase.calculator import XTB
except BaseException:
    XTB = "xtb-python not installed"  # Defined to provide informative error message.
    print("XTB cannot be used for AutoNEBGuesser as its ASE interface imported incorrectly. Skipping import...")


class TSInitialGuesser:
    """
    The abstract class for TS initial Guesser.

    Args:
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """
    _avail_ = True

    def __init__(self,
                 track_stats: Optional[bool] = False,
                 ):
        """
        Initialize the TS initial guesser.

        Args:
            track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
        """
        assert self._avail, f"The dependency requirement needs to be fulfilled to use {self.__class__.__name__}. Please install the relevant dependencies and try again.."
        self.track_stats = track_stats
        self.n_success = None
        self.percent_success = None
        self.stats = []

    def generate_ts_guesses(self,
                            mols: list,
                            save_dir: Optional[str] = None,
                            ) -> 'RDKitMol':
        """
        The key function used to generate TS guesses. It varies by the actual classes and need to implemented inside each class.
        The function should at least take mols and save_dir as input arguments. The returned value should be a RDKitMol with TS
        geometries.

        Args:
            mols (list): A list of reactant and product pairs.
            save_dir (Optional[str], optional): The path to save the results. Defaults to ``None`` for not saving.

        Returns:
            RDKitMol: The TS molecule in ``RDKitMol`` with 3D conformer saved with the molecule.

        Raises:
            NotImplementedError: This method needs to be implemented in the subclass.
        """
        raise NotImplementedError

    def save_guesses(self,
                     save_dir: str,
                     rp_combos: list,
                     ts_mol: 'RDKitMol'):
        """
        Save the generated guesses into the given ``save_dir``.

        Args:
            save_dir (str): The path to the directory to save the results.
            rp_combos (list): A list of reactant and product complex pairs used to generate transition states.
            ts_mol (RDKitMol): The TS molecule in RDKitMol with 3D conformer saved with the molecule.
        """

        # Save reactants and products into SDF format
        r_path = os.path.join(save_dir, "reactant_confs.sdf")
        p_path = os.path.join(save_dir, "product_confs.sdf")
        try:
            r_writer = Chem.rdmolfiles.SDWriter(r_path)
            p_writer = Chem.rdmolfiles.SDWriter(p_path)

            for r, p in rp_combos:

                if r.GetProp("Identity") == "reactant":
                    reactant = r
                    product = p
                elif r.GetProp("Identity") == "product":
                    reactant = p
                    product = r

                reactant, product = reactant.ToRWMol(), product.ToRWMol()
                reactant.SetProp("_Name", f"{Chem.MolToSmiles(reactant)}")
                product.SetProp("_Name", f"{Chem.MolToSmiles(product)}")
                r_writer.write(reactant)
                p_writer.write(product)

        except Exception:
            raise
        finally:
            r_writer.close()
            p_writer.close()

        # save TS initial guesses
        ts_path = os.path.join(save_dir, "ts_initial_guess_confs.sdf")
        try:
            ts_writer = Chem.rdmolfiles.SDWriter(ts_path)
            for i in range(ts_mol.GetNumConformers()):
                ts_writer.write(ts_mol, confId=i)
        except Exception:
            raise
        finally:
            ts_writer.close()

    def __call__(self,
                 mols: list,
                 multiplicity: Optional[int] = None,
                 save_dir: Optional[str] = None,
                 ):
        """
        The workflow to generate TS initial guesses.

        Args:
            mols (list): A list of molecules.
            multiplicity (int, optional): The spin multiplicity of the reaction. Defaults to ``None`` for not setting.
            save_dir (str, optional): The path to save results. Defaults to ``None`` for not saving.

        Returns:
            RDKitMol: The TS molecule in RDKitMol with 3D conformer saved with the molecule.
        """
        time_start = time()
        ts_mol_data = self.generate_ts_guesses(mols, multiplicity, save_dir)

        if self.track_stats:
            time_end = time()
            stats = {"time": time_end - time_start}
            self.stats.append(stats)

        return ts_mol_data


class TSEGNNGuesser(TSInitialGuesser):
    """
    The class for generating TS guesses using the TS-EGNN model.

    Args:
        trained_model_dir (str): The path to the directory storing the trained TS-EGNN model.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """
    _avail = _ts_egnn_avail

    def __init__(self,
                 trained_model_dir: str,
                 track_stats: Optional[bool] = False):
        """
        Initialize the TS-EGNN guesser.

        Args:
            trained_model_dir (str): The path to the directory storing the trained TS-EGNN model.
            track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
        """
        super(TSEGNNGuesser, self).__init__(track_stats)

        # Load the TS-EGNN model
        checkpoint_path = os.path.join(trained_model_dir, "best_model.ckpt")
        model_checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model_checkpoint["hyper_parameters"]["config"]["training"] = False  # we're not training
        self.module = LitTSModule(model_checkpoint["hyper_parameters"]["config"])
        self.module.load_state_dict(state_dict=model_checkpoint["state_dict"])

        # Setup TS-EGNN configuration
        self.config = self.module.config
        self.module.model.eval()
        self.test_dataset = EvalTSDataset(self.config)

    def generate_ts_guesses(self,
                            mols: list,
                            multiplicity: Optional[int] = None,
                            save_dir: Optional[str] = None):
        """
        Generate TS guesser.

        Args:
            mols (list): A list of reactant and product pairs.
            multiplicity (int, optional): The spin multiplicity of the reaction. Defaults to ``None``.
            save_dir (Optional[str], optional): The path to save the results. Defaults to ``None``.

        Returns:
            RDKitMol: The TS molecule in ``RDKitMol`` with 3D conformer saved with the molecule.
        """
        # Generate the input for the TS-EGNN model
        rp_inputs = [(x[0].ToRWMol(), None, x[1].ToRWMol()) for x in mols]  # reactant, None (for TS), product
        rp_data = [self.test_dataset.process_mols(m, no_ts=True) for m in rp_inputs]
        batch_data = Batch.from_data_list(rp_data)

        # Use TS-EGNN to make initial guesses
        predicted_ts_coords = self.module.model(batch_data)[:, :3].cpu().detach().numpy()
        predicted_ts_coords = np.array_split(predicted_ts_coords, len(rp_inputs))

        # Copy data to mol
        ts_mol = mols[0][0].Copy(quickCopy=True)
        ts_mol.EmbedMultipleNullConfs(len(rp_inputs))
        [ts_mol.GetConformer(i).SetPositions(np.array(predicted_ts_coords[i], dtype=float))
         for i in range(len(rp_inputs))]

        if save_dir:
            self.save_guesses(save_dir, mols, ts_mol.ToRWMol())

        return ts_mol


class TSGCNGuesser(TSInitialGuesser):
    """
    The class for generating TS guesses using the TS-GCN model.

    Args:
        trained_model_dir (str): The path to the directory storing the trained TS-GCN model.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """
    _avail = _ts_gcn_avail

    def __init__(self,
                 trained_model_dir: str,
                 track_stats: Optional[bool] = False):
        """
        Initialize the TS-EGNN guesser.

        Args:
            trained_model_dir (str): The path to the directory storing the trained TS-GCN model.
            track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
        """
        super(TSGCNGuesser, self).__init__(track_stats)

        # Load the TS-GCN model
        self.module = LitTSGCNModule.load_from_checkpoint(
            checkpoint_path=os.path.join(trained_model_dir, "best_model.ckpt"),
            strict=False,  # TODO: make sure d_init can be properly loaded
        )

        # Set the configuration of TS-GCN
        self.config = self.module.config
        self.module.model.eval()
        self.config["shuffle_mols"] = False
        self.config["prep_mols"] = False  # ts_generator class takes care of prep
        self.test_dataset = EvalTSGCNDataset(self.config)

    def generate_ts_guesses(self,
                            mols: list,
                            multiplicity: Optional[int] = None,
                            save_dir: Optional[str] = None):
        """
        Generate TS guesser.

        Args:
            mols (list): A list of reactant and product pairs.
            multiplicity (int, optional): The spin multiplicity of the reaction. Defaults to ``None``.
            save_dir (Optional[str], optional): The path to save the results. Defaults to ``None``.

        Returns:
            RDKitMol: The TS molecule in RDKitMol with 3D conformer saved with the molecule.
        """
        # Prepare the input for the TS-GCN model
        rp_inputs = [(x[0].ToRWMol(), None, x[1].ToRWMol()) for x in mols]
        rp_data = [self.test_dataset.process_mols(m, no_ts=True) for m in rp_inputs]
        batch_data = Batch.from_data_list(rp_data)

        # Use TS-GCN to make initial guesses
        _ = self.module.model(batch_data)
        predicted_ts_coords = torch.vstack([c[:m[0].GetNumAtoms()] for c, m in zip(batch_data.coords, batch_data.mols)])
        predicted_ts_coords = np.array_split(predicted_ts_coords.cpu().detach().numpy(), len(rp_inputs))

        # Copy data to mol
        ts_mol = mols[0][0].Copy(quickCopy=True)
        ts_mol.EmbedMultipleNullConfs(len(rp_inputs))
        [ts_mol.GetConformer(i).SetPositions(np.array(predicted_ts_coords[i], dtype=float))
         for i in range(len(rp_inputs))]

        if save_dir:
            self.save_guesses(save_dir, mols, ts_mol.ToRWMol())

        return ts_mol


class RMSDPPGuesser(TSInitialGuesser):
    """
    The class for generating TS guesses using the RMSD-PP method.

    Args:
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """
    _avail = True

    def __init__(self,
                 track_stats: Optional[bool] = False):
        """
        Initialize the RMSD-PP initial guesser.

        Args:
            track_stats (bool, optional): Whether to track the status. Defaults to False.
        """
        super(RMSDPPGuesser, self).__init__(track_stats)

    def generate_ts_guesses(self,
                            mols,
                            multiplicity: Optional[int] = None,
                            save_dir: Optional[str] = None):
        """
        Generate TS guesser.

        Args:
            mols (list): A list of reactant and product pairs.
            multiplicity (int, optional): The spin multiplicity of the reaction. Defaults to None.
            save_dir (Optional[str], optional): The path to save the results. Defaults to None.

        Returns:
            RDKitMol: The TS molecule in RDKitMol with 3D conformer saved with the molecule.
        """
        ts_guesses, used_rp_combos = [], []
        multiplicity = multiplicity or 1
        for r_mol, p_mol in mols:
            _, ts_guess = run_xtb_calc((r_mol, p_mol), return_optmol=True, job="--path", uhf=multiplicity - 1)
            if ts_guess:
                ts_guesses.append(ts_guess)
                used_rp_combos.append((r_mol, p_mol))

        if len(ts_guesses) == 0:
            # TODO: Need to think about catching this in the upper level
            return None

        # Copy data to mol
        ts_mol = mols[0][0].Copy(quickCopy=True)
        [ts_mol.AddConformer(t.GetConformer().ToConformer(), assignId=True)
         for t in ts_guesses]

        if save_dir:
            self.save_guesses(save_dir, used_rp_combos, ts_mol.ToRWMol())

        return ts_mol


class AutoNEBGuesser(TSInitialGuesser):
    """
    The class for generating TS guesses using the AutoNEB method.

    Args:
        optimizer (ase.calculator.calculator.Calculator): ASE calculator. Defaults to the XTB implementation ``xtb.ase.calculator.XTB``.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """
    _avail = _ase_avail

    def __init__(self,
                 optimizer: 'Calculator' = XTB,
                 track_stats: Optional[bool] = False):
        """
        Initialize the AutoNEB TS initial guesser.

        Args:
            optimizer (ase.calculator.calculator.Calculator): ASE calculator. Defaults to the XTB implementation ``xtb.ase.calculator.XTB``.
            track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
        """
        super(AutoNEBGuesser, self).__init__(track_stats)
        self.optimizer = optimizer

    @property
    def attach_calculators(self):
        """
        Set the calculator for each image.
        """
        def fun(images):
            for i in range(len(images)):
                images[i].set_calculator(self.optimizer())
        return fun

    @property
    def optimizer(self):
        """
        Optimizer used by the AutoNEB method.
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self,
                  optimizer: 'Calculator'):
        try:
            assert isinstance(optimizer, Calculator), f"Invalid optimizer used ('{optimizer}'). Please use ASE calculators."
        except NameError:
            print("ASE.Calculator was not correctly imported, thus AutoNEBGuesser can not be used.")
        self._optimizer = optimizer

    def generate_ts_guesses(self,
                            mols,
                            multiplicity: Optional[int] = None,
                            save_dir: Optional[str] = None):
        """
        Generate TS guesser.

        Args:
            mols (list): A list of reactant and product pairs.
            multiplicity (int, optional): The spin multiplicity of the reaction. Defaults to ``None``.
            save_dir (Optional[str], optional): The path to save the results. Defaults to ``None``.

        Returns:
            RDKitMol: The TS molecule in RDKitMol with 3D conformer saved with the molecule.
        """

        ts_guesses, used_rp_combos = [], []
        for i, (r_mol, p_mol) in enumerate(mols):

            # TODO: Need to clean the logic here, `ts_conf_dir` is used no matter `save_dir` being true
            if save_dir:
                ts_conf_dir = os.path.join(save_dir, f"neb_conf{i}")
                if not os.path.exists(ts_conf_dir):
                    os.makedirs(ts_conf_dir)

            r_traj = os.path.join(ts_conf_dir, "ts000.traj")
            p_traj = os.path.join(ts_conf_dir, "ts001.traj")

            r_coords = r_mol.GetConformer().GetPositions()
            r_numbers = r_mol.GetAtomicNumbers()
            r_atoms = Atoms(positions=r_coords, numbers=r_numbers)
            r_atoms.set_calculator(self.optimizer())
            qn = QuasiNewton(r_atoms, trajectory=r_traj, logfile=None)
            qn.run(fmax=0.05)

            p_coords = p_mol.GetConformer().GetPositions()
            p_numbers = p_mol.GetAtomicNumbers()
            p_atoms = Atoms(positions=p_coords, numbers=p_numbers)
            p_atoms.set_calculator(self.optimizer())
            qn = QuasiNewton(p_atoms, trajectory=p_traj, logfile=None)
            qn.run(fmax=0.05)

            # need to change dirs bc autoneb path settings are messed up
            cwd = os.getcwd()

            try:
                os.chdir(ts_conf_dir)

                autoneb = AutoNEB(self.attach_calculators,
                                  prefix='ts',
                                  optimizer='BFGS',
                                  n_simul=3,
                                  n_max=7,
                                  fmax=0.05,
                                  k=0.5,
                                  parallel=False,
                                  maxsteps=[50, 1000])

                autoneb.run()
                os.chdir(cwd)

                used_rp_combos.append((r_mol, p_mol))
                ts_guess_idx = np.argmax(autoneb.get_energies())
                ts_guesses.append(autoneb.all_images[ts_guess_idx].positions)

            except (CalculationFailed, AssertionError) as e:
                os.chdir(cwd)

        if len(ts_guesses) == 0:
            return None

        # copy data to mol
        ts_mol = mols[0][0].Copy(quickCopy=True)
        ts_mol.EmbedMultipleNullConfs(len(ts_guesses))
        [ts_mol.GetConformer(i).SetPositions(p) for i, p in enumerate(ts_guesses)]

        if save_dir:
            self.save_guesses(save_dir, used_rp_combos, ts_mol.ToRWMol())

        return ts_mol


class DEGSMGuesser(TSInitialGuesser):
    """
    The class for generating TS guesses using the DE-GSM method.

    Args:
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """
    _avail = True

    def __init__(self,
                 method: str = "GFN2-xTB",
                 nprocs: int = 1,
                 memory: int = 1,
                 gsm_args: Optional[str] = "",
                 track_stats: Optional[bool] = False):
        """
        Initialize the DE-GSM TS initial guesser.

        Args:
            track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
        """
        super(DEGSMGuesser, self).__init__(track_stats)
        self.gsm_args = gsm_args
        self.method = method
        self.nprocs = nprocs
        self.memory = memory

        try:
            self.gsm_entry_point = os.environ["gsm"]
        except KeyError:
            raise RuntimeError('No GSM entry point is found in the PATH.')

    def generate_ts_guesses(self,
                            mols: list,
                            multiplicity: Optional[int] = None,
                            save_dir: Optional[str] = None):
        """
        Generate TS guesser.

        Args:
            mols (list): A list of reactant and product pairs.
            multiplicity (int, optional): The spin multiplicity of the reaction. Defaults to ``None``.
            save_dir (Optional[str], optional): The path to save the results. Defaults to ``None``.

        Returns:
            RDKitMol: The TS molecule in RDKitMol with 3D conformer saved with the molecule.
        """
        # #TODO: May add a support for scratch directory
        # currently use the save directory as the working directory
        # This may not be ideal for some QM software, and whether to add a support
        # for scratch directory is left for future decision
        work_dir = os.path.abspath(save_dir) if save_dir else tempfile.mkdtemp()
        lot_inp_file = os.path.join(work_dir, "qstart.inp")
        lot_inp_str = write_gaussian_gsm(self.method, self.memory, self.nprocs)
        with open(lot_inp_file, "w") as f:
            f.writelines(lot_inp_str)

        ts_guesses, used_rp_combos = [], []
        for i, (r_mol, p_mol) in enumerate(mols):

            # TODO: Need to clean the logic here, `ts_conf_dir` is used no matter `save_dir` being true
            ts_conf_dir = os.path.join(work_dir, f"degsm_conf{i}")
            if not os.path.exists(ts_conf_dir):
                os.makedirs(ts_conf_dir)

            r_xyz = r_mol.ToXYZ()
            p_xyz = p_mol.ToXYZ()

            xyz_file = os.path.join(ts_conf_dir, f"degsm_conf{i}.xyz")
            with open(xyz_file, "w") as f:
                f.write(r_xyz)
                f.write(p_xyz)

            try:
                command = f"{self.gsm_entry_point} -xyzfile {xyz_file} -nproc {self.nprocs} -multiplicity {multiplicity} -mode DE_GSM -package Gaussian -lot_inp_file {lot_inp_file} {self.gsm_args}"
                with open(os.path.join(ts_conf_dir, "degsm.log"), "w") as f:
                    gsm_run = subprocess.run(
                        [command],
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        cwd=ts_conf_dir,
                        shell=True,
                    )
                used_rp_combos.append((r_mol, p_mol))
                tsnode_path = os.path.join(ts_conf_dir, 'TSnode_0.xyz')
                with open(tsnode_path) as f:
                    positions = f.read().splitlines()[2:]
                    positions = np.array([line.split()[1:] for line in positions], dtype=float)
                ts_guesses.append(positions)
            except FileNotFoundError:
                pass

        if len(ts_guesses) == 0:
            return None

        # copy data to mol
        ts_mol = mols[0][0].Copy(quickCopy=True)
        ts_mol.EmbedMultipleNullConfs(len(ts_guesses))
        [ts_mol.GetConformer(i).SetPositions(p) for i, p in enumerate(ts_guesses)]

        if save_dir:
            self.save_guesses(save_dir, used_rp_combos, ts_mol.ToRWMol())

        return ts_mol
