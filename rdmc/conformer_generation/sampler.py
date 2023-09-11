#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modules for torsional sampling
"""

import os
import json
import pickle
import logging
import tempfile
from itertools import combinations, product
from typing import List, Tuple, Optional, Union

import numpy as np
from scipy import constants
from rdkit import Chem
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from rdmc.mol import RDKitMol
from rdmc.conformer_generation.utils import mol_to_dict
from rdmc.mathlib.greedymin import search_minimum
from rdmc.ts import get_formed_and_broken_bonds

try:
    from xtb.libxtb import VERBOSITY_FULL, VERBOSITY_MINIMAL, VERBOSITY_MUTED
    from xtb.utils import get_method, _methods
    from xtb.interface import Calculator
except ImportError:
    print("No xtb-python installation detected. Skipping import...")

try:
    import scine_sparrow
    import scine_utilities as su
except:
    print("No scine_sparrow installation detected. Skipping import...")


class TorsionalSampler:
    """
    A class to find possible conformers by sampling the PES for each torsional pair.
    You have to have the `Sparrow <https://github.com/qcscine/sparrow>`_ and `xtb-python <https://github.com/grimme-lab/xtb-python>`_ packages installed to run this workflow.

    Args:
        method (str, optional): The method to be used for automated conformer search. Only the methods available in Spharrow and xtb-python can be used.
                                Defaults to ``"GFN2-xTB"``.
        nprocs (int, optional): The number of processors to use. Defaults to ``1``.
        memory (int, optional): Memory in GB used by Gaussian. Defaults to ``1``.
        n_point_each_torsion (float, optional): Number of points to be sampled along each rotational mode. Defaults to ``45.``.
        n_dimension (int, optional): Number of dimensions. Defaults to ``2``. If ``-1`` is assigned, the number of dimension would be the number of rotatable bonds.
        optimizer (ConfGenOptimizer or TSOptimizer, optional): The optimizer used to optimize TS or stable specials geometries. Available options for
                                                               `TSOptimizer <rdmc.conformer_generation.ts_optimizers.TSOptimizer>`
                                                               are :obj:`SellaOptimizer <rdmc.conformer_generation.ts_optimizers.SellaOptimizer>`,
                                                               :obj:`OrcaOptimizer <rdmc.conformer_generation.ts_optimizers.OrcaOptimizer>`,
                                                               and :obj:`GaussianOptimizer <rdmc.conformer_generation.ts_optimizers.GaussianOptimizer>`.
        pruner (ConfGenPruner, optional): The pruner used to prune conformers based on geometric similarity after optimization. Available options are
                                          :obj:`CRESTPruner <rdmc.conformer_generation.pruners.CRESTPruner>` and
                                          :obj:`TorsionPruner <rdmc.conformer_generation.pruners.TorsionPruner>`.
        verifiers (TSVerifier, Verifier, list of TSVerifiers or list of Verifiers, optional): The verifier or a list of verifiers used to verify the obtained conformer. Available
                                                                                              options are
                                                                                              :obj:`GaussianIRCVerifier <rdmc.conformer_generation.ts_verifiers.GaussianIRCVerifier>`,
                                                                                              :obj:`OrcaIRCVerifier <rdmc.conformer_generation.ts_verifiers.OrcaIRCVerifier>`, and
                                                                                              :obj:`XTBFrequencyVerifier <rdmc.conformer_generation.ts_verifiers.XTBFrequencyVerifier>`.
    """

    def __init__(self,
                 method: str = "GFN2-xTB",
                 nprocs: int = 1,
                 memory: int = 1,
                 n_point_each_torsion: int = 45,
                 n_dimension: int = 2,
                 optimizer: Optional[Union["ConfGenOptimizer","TSOptimizer"]] = None,
                 pruner: Optional["ConfGenPruner"] = None,
                 verifiers: Optional[Union["TSVerifier",
                                           "Verifier",
                                           List["TSVerifier"],
                                           List["Verifier"]]] = None,
                 ):
        """
        Initiate the TorsionalSampler class object.

        Args:
            method (str, optional): The method to be used for automated conformer search. Only the methods available in Spharrow and xtb-python can be used.
                                    Defaults to ``"GFN2-xTB"``.
            nprocs (int, optional): The number of processors to use. Defaults to ``1``.
            memory (int, optional): Memory in GB used by Gaussian. Defaults to ``1``.
            n_point_each_torsion (float, optional): Number of points to be sampled along each rotational mode. Defaults to ``45.``.
            n_dimension (int, optional): Number of dimensions. Defaults to ``2``. If ``-1`` is assigned, the number of dimension would be the number of rotatable bonds.
            optimizer (ConfGenOptimizer or TSOptimizer, optional): The optimizer used to optimize TS or stable specials geometries. Available options for
                                                                `TSOptimizer <rdmc.conformer_generation.ts_optimizers.TSOptimizer>`
                                                                are :obj:`SellaOptimizer <rdmc.conformer_generation.ts_optimizers.SellaOptimizer>`,
                                                                :obj:`OrcaOptimizer <rdmc.conformer_generation.ts_optimizers.OrcaOptimizer>`,
                                                                and :obj:`GaussianOptimizer <rdmc.conformer_generation.ts_optimizers.GaussianOptimizer>`.
            pruner (ConfGenPruner, optional): The pruner used to prune conformers based on geometric similarity after optimization. Available options are
                                            :obj:`CRESTPruner <rdmc.conformer_generation.pruners.CRESTPruner>` and
                                            :obj:`TorsionPruner <rdmc.conformer_generation.pruners.TorsionPruner>`.
            verifiers (TSVerifier, Verifier, list of TSVerifiers or list of Verifiers, optional): The verifier or a list of verifiers used to verify the obtained conformer. Available
                                                                                                options are
                                                                                                :obj:`GaussianIRCVerifier <rdmc.conformer_generation.ts_verifiers.GaussianIRCVerifier>`,
                                                                                                :obj:`OrcaIRCVerifier <rdmc.conformer_generation.ts_verifiers.OrcaIRCVerifier>`, and
                                                                                                :obj:`XTBFrequencyVerifier <rdmc.conformer_generation.ts_verifiers.XTBFrequencyVerifier>`.
        """
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.method = method
        self.nprocs = nprocs
        self.memory = memory
        self.n_point_each_torsion = n_point_each_torsion
        self.n_dimension = n_dimension
        self.optimizer = optimizer
        self.pruner = pruner
        self.verifiers = [] if not verifiers else verifiers

    def get_conformers_by_change_torsions(self,
                                          mol: RDKitMol,
                                          id: int = 0,
                                          torsions: Optional[list] = None,
                                          exclude_methyl: bool = True,
                                          on_the_fly_check: bool = True,
                                          ) -> List[RDKitMol]:
        """
        Generate conformers by rotating the angles of the torsions. A on-the-fly check
        can be applied, which identifies the conformers with colliding atoms.

        Args:
            mol (RDKitMol): A RDKitMol molecule object.
            id (int): The ID of the conformer to be obtained. Defaults to ``0``.
            torsions (list): A list of four-atom-index lists indicating the torsional modes. Defaults to ``None``,
                             which means all the rotatable bonds will be used.
            exclude_methyl (bool): Whether exclude the torsions with methyl groups. Defaults to ``False``.
                                   Only valid if ``torsions`` is not provided.
            on_the_fly_filter (bool): Whether to check colliding atoms on the fly. Defaults to ``True``.

        Returns:
            lis: A list of RDKitMol of sampled 3D geometries for each torsional mode.
        """
        conf = mol.Copy().GetConformer(id=id)
        origin_coords = mol.GetPositions(id=id)
        if not torsions:
            torsions = mol.GetTorsionalModes(excludeMethyl=exclude_methyl)
        self.logger.info(f"Number of torsions: {len(torsions)}")
        conf.SetTorsionalModes(torsions)
        original_angles = conf.GetAllTorsionsDeg()

        # If `-1` is assigned for n_dimension, it would be the number of rotatable bonds.
        if self.n_dimension == -1:
            n_dimension = len(torsions)
            self.logger.info(f"Sampling {self.n_point_each_torsion} to the power of {n_dimension} conformers...")
        else:
            n_dimension = self.n_dimension

        conformers_by_change_torsions = []
        for torsion_pair in combinations(torsions, n_dimension):
            # Reset the geometry
            conf.SetPositions(origin_coords)

            # Get angles
            sampling = [
                self.n_point_each_torsion if tor in torsion_pair else 0
                for tor in torsions
            ]
            angles_list = get_separable_angle_list(sampling, original_angles)
            angle_mesh = product(*angles_list)

            # Generate conformers by rotating the angles of the torsions
            # The result will be saved into ``bookkeep``.
            bookkeep = {}
            all_torsions = conf.GetTorsionalModes()
            try:
                changing_torsions_index = []
                for tor in torsions:
                    changing_torsions_index.append(all_torsions.index(tor))
            except ValueError:
                raise ValueError(f"The torsion of {tor} is not in all_torsions.")

            original_angles = conf.GetAllTorsionsDeg()

            for ind, angles in enumerate(angle_mesh):
                for i, angle, tor in zip(range(len(angles)), angles, torsions):
                    conf.SetTorsionDeg(tor, angle)
                    original_angles[changing_torsions_index[i]] = angle
                bookkeep[ind] = {
                    "angles": original_angles.copy(),
                    "coords": conf.GetPositions(),
                }
                bookkeep[ind]["colliding_atoms"] = (
                    conf.HasCollidingAtoms() if on_the_fly_check else None
                )

            # Save all the sampled 3D geometries in a RDKitMol
            mols = mol.Copy()
            mols.SetProp("torsion_pair", str(torsion_pair))
            mols.EmbedMultipleNullConfs(len(bookkeep))
            for i in range(len(bookkeep)):
                mols.GetConformer(i).SetPositions(bookkeep[i]["coords"])
                mols.GetConformer(i).SetProp("angles", str(bookkeep[i]["angles"]))
                mols.GetConformer(i).SetProp("colliding_atoms", str(bookkeep[i]["colliding_atoms"]))
            conformers_by_change_torsions.append(mols)

        return conformers_by_change_torsions

    def __call__(self,
                 mol: RDKitMol,
                 id: int,
                 rxn_smiles: Optional[str] = None,
                 torsions: Optional[List] = None,
                 no_sample_dangling_bonds: bool = True,
                 no_greedy: bool = False,
                 save_dir: Optional[str] = None,
                 save_plot: bool = True,
                 ):
        """
        Run the workflow of conformer generation.

        Args:
            mol (RDKitMol): An RDKitMol object.
            id (int): The ID of the conformer to be obtained.
            rxn_smiles (str, optional): The SMILES of the reaction. The SMILES should be formatted similar to
                                        `"reactant1.reactant2>>product1.product2."`. Defaults to ``None``, which means
                                        ``torsions`` will be provided and used to generate conformers.
            torsions (list, optional): A list of four-atom-index lists indicating the torsional modes.
            no_sample_dangling_bonds (bool): Whether to sample dangling bonds. Defaults to ``False``.
            no_greedy (bool): Whether to use greedy algorithm to find local minima. If ``True``, all the sampled conformers
                              would be passed to the optimization and verification steps. Defaults to ``False``.
            save_dir (str or Pathlike object, optional): The path to save the outputs generated during the generation.
            save_plot (bool): Whether to save the heat plot for the PES of each torsional mode. Defaults to ``True``.
        """
        # Get bonds which will not be rotated during conformer searching
        sampler_mol = mol.Copy()
        if rxn_smiles:
            r_smi, p_smi = rxn_smiles.split(">>")
            r_mol = RDKitMol.FromSmiles(r_smi)
            p_mol = RDKitMol.FromSmiles(p_smi)
            formed_bonds, broken_bonds = get_formed_and_broken_bonds(r_mol, p_mol)
            bonds = formed_bonds + broken_bonds
        else:
            bonds = []

        # Use double bond to avoid to be counted as a torsional mode
        # If you want to include it, please use BondType.SINGLE
        rw_mol = sampler_mol.ToRWMol()
        sampler_mol.UpdatePropertyCache()

        if no_sample_dangling_bonds:
            set_BondType = Chem.BondType.DOUBLE
        else:
            set_BondType = Chem.BondType.SINGLE

        for bond_inds in bonds:
            bond = rw_mol.GetBondBetweenAtoms(bond_inds[0], bond_inds[1])
            if bond:
                bond.SetBondType(set_BondType)
            else:
                rw_mol.AddBond(*bond_inds, set_BondType)

        # Get all the sampled conformers for each torsinal pair
        sampler_mol = sampler_mol.FromMol(rw_mol)
        conformers_by_change_torsions = self.get_conformers_by_change_torsions(
            sampler_mol, id, torsions=torsions, on_the_fly_check=True
        )

        if conformers_by_change_torsions == []:
            self.logger.info("Doesn't find any torsional pairs! Using original result...")
            return mol

        if save_dir:
            conf_dir = os.path.join(save_dir, f"torsion_sampling_{id}")
            os.makedirs(conf_dir, exist_ok=True)

        minimum_mols = mol.Copy(quickCopy=True)
        if no_greedy:
            for confs in conformers_by_change_torsions:
                num = confs.GetNumConformers()
                for i in range(num):
                    colliding_atoms = json.loads(
                        confs.GetConformer(i).GetProp("colliding_atoms").lower()
                    )
                    if not colliding_atoms:
                        [minimum_mols.AddConformer(confs.GetConformer(i).ToConformer(), assignId=True)]

            if self.n_dimension == -1:
                n_conformers = minimum_mols.GetNumConformers()
                self.logger.info(f"After on the fly check of potentially colliding atoms, {n_conformers} conformers will be passed to the following optimization and verification steps.")
        else:
            # Setting the environmental parameters before running energy calculations
            try:
                original_OMP_NUM_THREADS = os.environ["OMP_NUM_THREADS"]
                original_OMP_STACKSIZE = os.environ["OMP_STACKSIZE"]
            except KeyError:
                original_OMP_NUM_THREADS = None
                original_OMP_STACKSIZE = None

            os.environ["OMP_NUM_THREADS"] = str(self.nprocs)
            os.environ["OMP_STACKSIZE"] = f"{self.memory}G"

            # Search the minimum points on all the scanned potential energy surfaces
            for confs in conformers_by_change_torsions:
                # Calculate energy for each conformer
                energies = []
                num = confs.GetNumConformers()
                for i in range(num):
                    colliding_atoms = json.loads(
                        confs.GetConformer(i).GetProp("colliding_atoms").lower()
                    )
                    if colliding_atoms:
                        energy = np.nan
                    else:
                        energy = get_energy(confs, confId=i, method=self.method)
                    confs.GetConformer(i).SetProp("Energy", str(energy))
                    energies.append(energy)

                # Reshape the energies from a 1-D list to corresponding np.ndarray
                energies = np.array(energies)
                if self.n_dimension == 1:
                    energies = energies.reshape(-1)
                else:
                    num = confs.GetNumConformers()
                    nsteps = int(round(len(energies) ** (1. / self.n_dimension)))
                    energies = energies.reshape((nsteps,) * self.n_dimension)

                # Find local minima on the scanned potential energy surface by greedy algorithm
                rescaled_energies, mask = preprocess_energies(energies)
                minimum_points = search_minimum(rescaled_energies, fsize=2)

                # Save the conformers located in local minima on PES to minimum_mols
                ids = []
                for minimum_point in minimum_points:
                    if len(minimum_point) == 1:
                        ids.append(minimum_point[0])
                    else:
                        ind = 0
                        for dimension, value in enumerate(minimum_point[::-1]):
                            ind += nsteps**dimension * value
                        ids.append(ind)

                [minimum_mols.AddConformer(confs.GetConformer(i).ToConformer(), assignId=True) for i in ids]

                if save_dir and save_plot and len(rescaled_energies.shape) in [1, 2]:
                    torsion_pair = confs.GetProp("torsion_pair")
                    title = f"torsion_pair: {torsion_pair}"
                    plot_save_path = os.path.join(conf_dir, f"{torsion_pair}.png")
                    plot_heat_map(
                        rescaled_energies,
                        minimum_points,
                        plot_save_path,
                        mask=mask,
                        detailed_view=False,
                        title=title,
                    )
            self.logger.info(f"{minimum_mols.GetNumConformers()} local minima on PES were found...")

            # Recovering the environmental parameters
            if original_OMP_NUM_THREADS and original_OMP_STACKSIZE:
                os.environ["OMP_NUM_THREADS"] = original_OMP_NUM_THREADS
                os.environ["OMP_STACKSIZE"] = original_OMP_STACKSIZE
            else:
                del os.environ["OMP_NUM_THREADS"]
                del os.environ["OMP_STACKSIZE"]

        # Run opt and verify guesses
        multiplicity = minimum_mols.GetSpinMultiplicity()
        self.logger.info("Optimizing guesses...")
        minimum_mols.KeepIDs = {i: True for i in range(minimum_mols.GetNumConformers())}  # map ids of generated guesses thru workflow

        try:
            mols = minimum_mols.ToRWMol()
            path = os.path.join(conf_dir, "sampling_confs.sdf")
            writer = Chem.rdmolfiles.SDWriter(path)
            for i in range(mols.GetNumConformers()):
                if rxn_smiles:
                    mols.SetProp("rxn_smiles", rxn_smiles)
                writer.write(mols, confId=i)
        except Exception:
            raise
        finally:
            writer.close()

        if self.optimizer:
            opt_minimum_mols = self.optimizer(
                minimum_mols,
                multiplicity=multiplicity,
                save_dir=conf_dir,
            )
        else:
            return mol

        if self.pruner:
            self.logger.info("Pruning species guesses...")
            _, unique_ids = self.pruner(
                mol_to_dict(opt_minimum_mols, conf_copy_attrs=["KeepIDs", "energy"]),
                sort_by_energy=False,
                return_ids=True,
            )
            self.logger.info(f"Pruned {self.pruner.n_pruned_confs} conformers")
            opt_minimum_mols.KeepIDs = {k: k in unique_ids and v for k, v in opt_minimum_mols.KeepIDs.items()}
            with open(os.path.join(conf_dir, "prune_check_ids.pkl"), "wb") as f:
                pickle.dump(opt_minimum_mols.KeepIDs, f)

        # Verify from lowest energy conformer to highest energy conformer
        # Stopped whenever one conformer pass all the verifiers
        self.logger.info("Verifying guesses...")
        energy_dict = opt_minimum_mols.energy
        sorted_index = [k for k, v in sorted(energy_dict.items(), key=lambda item: item[1]) if opt_minimum_mols.KeepIDs[k]]  # Order by energy
        for idx in sorted_index:
            energy = opt_minimum_mols.energy[idx]
            if energy >= mol.energy[id]:
                self.logger.info("Sampler doesn't find conformer with lower energy!! Using original result...")
                return mol

            opt_minimum_mols.KeepIDs = {i: False for i in range(opt_minimum_mols.GetNumConformers())}  # map ids of generated guesses thru workflow
            opt_minimum_mols.KeepIDs[idx] = True
            for verifier in self.verifiers:
                verifier(
                    opt_minimum_mols,
                    multiplicity=multiplicity,
                    save_dir=conf_dir,
                    rxn_smiles=rxn_smiles,
                )

            if opt_minimum_mols.KeepIDs[idx]:
                self.logger.info(f"Sampler finds conformer with lower energy. The energy decreases {mol.energy[id] - energy} kcal/mol.")
                mol.GetConformer(id).SetPositions(opt_minimum_mols.GetConformer(idx).GetPositions())
                mol.energy[id] = energy
                mol.frequency[id] = opt_minimum_mols.frequency[idx]
                return mol

        self.logger.info("Sampler doesn't find conformer with lower energy!! Using original result...")
        return mol


def get_separable_angle_list(samplings: Union[List, Tuple],
                             from_angles: Optional[Union[List, Tuple]] = None
                             ) -> List[List]:
    """
    Get a angle list for each input dimension. For each dimension
    The input can be a ``int`` indicating the angles will be evenly sampled;
    or a ``list`` indicating the angles to be sampled;

    Args:
        samplings (Union[List, Tuple]): An array of sampling information.
                                        For each element, it can be either list or int.
        from_angles (Union[List, Tuple]): An array of initial angles.
                                          If not set, all angles will begin at zeros.

    Returns:
        list: A list of sampled angles sets.

    Examples:

        .. code-block:: python

            get_separable_angle_list([[120, 240,], 4, 0])
            >>> [[120, 240], [0, 90, 180, 270], [0]]

    """
    from_angles = from_angles or len(samplings) * [0.0]
    angle_list = []
    for ind, angles in enumerate(samplings):
        # Only provide a number
        # This is the step number of the angles
        if isinstance(angles, (int, float)):
            try:
                step = 360 // angles
            except ZeroDivisionError:
                # Does not change
                angles = from_angles[ind] + np.array([0])
            else:
                angles = from_angles[ind] + np.array([step * i for i in range(angles)])
        elif isinstance(angles, list):
            angles = from_angles[ind] + np.array(angles)

        # Set to 0 - 360 range
        for i in range(angles.shape[0]):
            while angles[i] < 0.0:
                angles[i] += 360
            while angles[i] > 360.0:
                angles[i] -= 360

        angle_list.append(angles.tolist())
    return angle_list


def get_energy(mol: RDKitMol,
               confId: int = 0,
               method: str = "GFN2-xTB",
               ) -> float:
    """
    Calculate the energy of the ``RDKitMol`` with given ``confId``. The unit is in kcal/mol.
    Only support methods already supported either in sparrow or xtb-python.

    Args:
        mol (RDKitMol): A RDKitMol molecule object.
        confId (int): The ID of the conformer for calculating energy. Defaults to ``0``.
        method (str): Which semi-empirical method to be used in running energy calculation. Defaults to ``"GFN2-xTB"``.

    Returns:
        The energy of the conformer.
    """
    if method.lower() in _methods.keys():
        ANGSTROM_PER_BOHR = constants.physical_constants["Bohr radius"][0] * 1.0e10
        charge = mol.GetFormalCharge()
        uhf = mol.GetSpinMultiplicity() - 1
        numbers = np.array(mol.GetAtomicNumbers())
        positions = mol.GetPositions(confId) / ANGSTROM_PER_BOHR
        calc = Calculator(get_method(method), numbers, positions, charge, uhf)
        calc.set_verbosity(VERBOSITY_MUTED)
        res = calc.singlepoint()
        energy = res.get_energy()
    elif method.lower() in ["mndo", "am1", "pm3", "pm6"]:
        # Load xyz into calculator
        manager = su.core.ModuleManager()
        calculator = manager.get("calculator", method)
        calculator.settings["molecular_charge"] = mol.GetFormalCharge()
        calculator.settings["spin_multiplicity"] = mol.GetSpinMultiplicity()
        log = su.core.Log()
        log.output.remove("cout")
        calculator.log = log
        temp_dir = tempfile.mkdtemp()
        xyz_path = os.path.join(temp_dir, "mol.xyz")
        xyz_string = mol.ToXYZ(confId)
        with open(xyz_path, "w") as f:
            f.write(xyz_string)
        calculator.structure = su.io.read(xyz_path)[0]

        # Configure Calculator
        calculator.set_required_properties([su.Property.Energy])

        # Calculate
        results = calculator.calculate()
        energy = results.energy
    else:
        raise NotImplementedError(f"The {method} method is not supported.")

    return energy


def preprocess_energies(energies: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rescale the energy based on the lowest energy.

    Args:
        energies (np.ndarray): A np.ndarray containing the energies for each sampled point.

    Returns:
        tuple: The rescaled energies and the mask pointing out positions having values
    """
    max_energy = np.nanmax(energies)
    min_energy = np.nanmin(energies)

    mask = np.isnan(energies)
    replaced_energy = 0.99 * max_energy if max_energy < 0 else 1.01 * max_energy
    padded_energies = np.nan_to_num(energies, nan=replaced_energy)

    # Rescale the energy based on the lowest energy
    # This will not change the result of search but make detailed view more clear
    rescaled_energies = padded_energies - min_energy
    return rescaled_energies, mask


def plot_heat_map(energies: np.ndarray,
                  minimum_points: List[Tuple],
                  save_path: str,
                  mask: Optional[np.ndarray] = None,
                  detailed_view: bool = False,
                  title: Optional[str] = None,
                  ):
    """
    Plot and save the heat map of a given PES.

    Args:
        energies (np.ndarray): A ``np.ndarray`` containing the energies for each sampled point.
        minimum_points (List[Tuple]): A list of tuples containing the indices of the minimum points.
        save_path (str): The path to save the plot.
        mask (np.ndarray, optional): A ``np.ndarray`` containing the mask for the energies.
        detailed_view (bool): Whether to plot the detailed view of the PES. Defaults to ``False``.
        title (str, optional): The title of the plot.
    """
    import seaborn as sns

    if detailed_view:
        fig_size = (28, 20)
        annot = True  # detailed view
    else:
        fig_size = (5, 4)
        annot = False  # overlook view

    if mask is None:
        mask = np.isnan(energies)

    f, ax = plt.subplots(figsize=fig_size)

    # Plot as an heatmap by Seaborn
    if len(energies.shape) == 1:
        energies = energies.reshape(-1, 1)
        mask = mask.reshape(-1, 1)

    ax = sns.heatmap(
        energies,
        vmin=np.nanmax(energies),
        vmax=np.nanmax(energies),
        cmap="YlGnBu",
        annot=annot,
        annot_kws={"fontsize": 8},
        mask=mask,
        square=True,
    )

    # Identified the minimum by red rectangle patches
    for point in minimum_points:
        # In the heatmap, the first index is for the y-axis
        # while in the pyplot the first index is for the x-axis
        # therefore, for displaying, we need to invert the axis
        if len(point) == 1 and energies[point[0], 0] < 0.5:
            ax.add_patch(
                Rectangle((0, point[::-1][0]), 1, 1, fill=False, edgecolor="red", lw=2)
            )
        elif energies[point[0], point[1]] < 0.5:
            ax.add_patch(
                Rectangle(point[::-1], 1, 1, fill=False, edgecolor="red", lw=2)
            )

    if title:
        plt.title(title)

    plt.savefig(save_path, dpi=500)
    plt.close()
