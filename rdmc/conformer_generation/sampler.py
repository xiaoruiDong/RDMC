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
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from rdmc.mol import RDKitMol
from rdmc.ts import get_formed_and_broken_bonds

from xtb.libxtb import VERBOSITY_FULL, VERBOSITY_MINIMAL, VERBOSITY_MUTED
from xtb.utils import get_method, _methods
from xtb.interface import Calculator

try:
    import scine_sparrow
    import scine_utilities as su
except:
    print("No scine_sparrow installation deteced. Skipping import...")


class TorisonalSampler:
    """
    A class to find possible conformers by sampling the PES for each torsional pair.
    You have to have the Spharrow and xtb-python packages installed to run this workflow.
    """

    def __init__(
        self,
        method: str = "GFN2-xTB",
        nprocs: int = 1,
        memory: int = 1,
        n_point_each_torsion: int = 45,
        n_dimension: int = 2,
        optimizer: Optional["TSOptimizer"] = None,
        pruner: Optional["ConfGenPruner"] = None,
        verifiers: Optional[Union["TSVerifier", List["TSVerifier"]]] = None,
    ):
        """
        Initiate the TorisonalSampler class object.
        Args:
            method (str, optional): The method to be used for automated conformer search. Only the methods available in Spharrow and xtb-python can be used.
                                    Defaults to GFN2-xTB.
            nprocs (int, optional): The number of processors to use. Defaults to 1.
            memory (int, optional): Memory in GB used by Gaussian. Defaults to 1.
            n_point_each_torsion (int): Number of points to be sampled along each rotational mode. Defaults to 45.
            n_dimension (int): Number of dimensions. Defaults to 2.
            optimizer (TSOptimizer, optional): The optimizer used to optimize TS geometries. Available options are `SellaOptimizer`, `OrcaOptimizer`, and
                                               `GaussianOptimizer`.
            pruner (ConfGenPruner, optional): The pruner used to prune conformers based on geometric similarity after optimization. Available options are
                                              `CRESTPruner` and `TorsionPruner`.
            verifiers (TSVerifier or list of TSVerifiers, optional): The verifier or a list of verifiers used to verify the obtained TS conformer. Available
                                                                     options are `GaussianIRCVerifier`, `OrcaIRCVerifier`, and `XTBFrequencyVerifier`.
        """
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.method = method
        self.nprocs = nprocs
        self.memory = memory
        self.n_point_each_torsion = n_point_each_torsion
        self.n_dimension = n_dimension
        if self.n_dimension not in [1, 2]:
            raise NotImplementedError(f"The torsional sampling method doesn't supported sampling for {n_dimension} dimensions.")
        self.optimizer = optimizer
        self.pruner = pruner
        self.verifiers = [] if not verifiers else verifiers

    def get_conformers_by_change_torsions(
        self,
        mol: RDKitMol,
        id: int = 0,
        torsions: List = None,
        exclude_methyl: bool = True,
        on_the_fly_check: bool = True,
    ) -> List[RDKitMol]:
        """
        Generate conformers by rotating the angles of the torsions. A on-the-fly check
        can be applied, which identifies the conformers with colliding atoms.

        Args:
            mol (RDKitMol): A RDKitMol molecule object.
            id (int): The ID of the conformer to be obtained. Defaults to 0.
            torsions (list): A list of four-atom-index lists indicating the torsional modes.
            exclude_methyl (bool): Whether exclude the torsions with methyl groups. Defaults to False.
            on_the_fly_filter (bool): Whether to check colliding atoms on the fly. Defaults to True.

        Returns:
            A list of RDKitMol of sampled 3D geometries for each torsional mode.
        """
        conf = mol.Copy().GetConformer(id=id)
        origin_coords = mol.GetPositions(id=id)
        if not torsions:
            torsions = mol.GetTorsionalModes(excludeMethyl=exclude_methyl)
        self.logger.info(f"Number of torsions: {len(torsions)}")
        conf.SetTorsionalModes(torsions)
        original_angles = conf.GetAllTorsionsDeg()

        conformers_by_change_torsions = []
        for torsion_pair in combinations(torsions, self.n_dimension):
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
                changing_torsions_index = [all_torsions.index(tor) for tor in torsions]
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

    def __call__(
        self,
        mol: RDKitMol,
        id: int,
        rxn_smiles: Optional[str] = None,
        save_dir: Optional[str] = None,
        save_plot: bool = True,
    ):
        """
        Run the workflow of TS conformer generation.

        Args:
            mol (RDKitMol): An RDKitMol object.
            id (int): The ID of the conformer to be obtained.
            rxn_smiles (str, optional): The SMILES of the reaction. The SMILES should be formatted similar to `"reactant1.reactant2>>product1.product2."`.
            save_dir (str or Pathlike object, optional): The path to save the outputs generated during the generation.
            save_plot (bool): Whether to save the heat plot for the PES of each torsinal mode. Defaults to True.
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
        for bond in bonds:
            bond = rw_mol.GetBondBetweenAtoms(bond[0], bond[1])
            if bond:
                bond.SetBondType(Chem.BondType.DOUBLE)
            elif:
                rw_mol.AddBond(*bond, Chem.BondType.DOUBLE)

        # Get all the sampled conformers for each torsinal pair
        sampler_mol = sampler_mol.FromMol(rw_mol)
        conformers_by_change_torsions = self.get_conformers_by_change_torsions(
            sampler_mol, id, on_the_fly_check=True
        )

        if save_dir:
            ts_conf_dir = os.path.join(save_dir, f"torsion_sampling_{id}")
            os.makedirs(ts_conf_dir, exist_ok=True)

        # Setting the environmental parameters before running energy calculations
        original_OMP_NUM_THREADS = os.environ["OMP_NUM_THREADS"]
        original_OMP_STACKSIZE = os.environ["OMP_STACKSIZE"]
        os.environ["OMP_NUM_THREADS"] = str(self.nprocs)
        os.environ["OMP_STACKSIZE"] = f"{self.memory}G"

        # Search the minimum points on all the scanned potential energy surfaces
        minimum_mols = mol.Copy(quickCopy=True)
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
            elif self.n_dimension == 2:
                num = confs.GetNumConformers()
                nsteps = int(num**0.5)
                energies = energies.reshape(nsteps, nsteps)

            # Find local minima on the scanned potential energy surface by greedy algorithm
            rescaled_energies, mask = preprocess_energies(energies)
            minimum_points = search_minimum(rescaled_energies, fsize=2)

            # Save the conformers located in local minima on PES to minimum_mols
            ids = []
            for minimum_point in minimum_points:
                if len(minimum_point) == 1:
                    ids.append(minimum_point[0])
                else:
                    row, column = minimum_point
                    ids.append(nsteps * row + column)

            [minimum_mols.AddConformer(confs.GetConformer(i).ToConformer(), assignId=True) for i in ids]

            if save_dir and save_plot:
                torsion_pair = confs.GetProp("torsion_pair")
                title = f"torsion_pair: {torsion_pair}"
                plot_save_path = os.path.join(ts_conf_dir, f"{torsion_pair}.png")
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
        os.environ["OMP_NUM_THREADS"] = original_OMP_NUM_THREADS
        os.environ["OMP_STACKSIZE"] = original_OMP_STACKSIZE

        # Run opt and verify guesses
        multiplicity = minimum_mols.GetSpinMultiplicity()
        self.logger.info("Optimizing guesses...")
        minimum_mols.KeepIDs = {i: True for i in range(minimum_mols.GetNumConformers())}  # map ids of generated guesses thru workflow
        minimum_mols.FiltIDs = {i: True for i in range(minimum_mols.GetNumConformers())}  # map ids of generated guesses thru workflow

        if rxn_smiles:
            opt_minimum_mols = self.optimizer(
                minimum_mols,
                multiplicity=multiplicity,
                save_dir=ts_conf_dir,
                rxn_smiles=rxn_smiles,
            )
        else:
            mols_data = mol_to_dict(opt_minimum_mols)
            opt_minimum_mols_data = self.optimizer(mols_data)
            opt_minimum_mols = dict_to_mol(opt_minimum_mols_data)

        if pruner:
            self.logger.info("Pruning TS guesses...")
            _, unique_ids = self.pruner(
                mol_to_dict(opt_minimum_mols, conf_copy_attrs=["KeepIDs", "energy"]),
                sort_by_energy=False,
                return_ids=True,
            )
            self.logger.info(f"Pruned {pruner.n_pruned_confs} TS conformers")
            opt_minimum_mols.KeepIDs = {k: k in unique_ids and v for k, v in opt_minimum_mols.KeepIDs.items()}
            with open(os.path.join(ts_conf_dir, "prune_check_ids.pkl"), "wb") as f:
                pickle.dump(opt_minimum_mols.KeepIDs, f)

        # Verify from lowest energy conformer to highest energy conformer
        # Stopped whenever one conformer pass all the verifiers
        self.logger.info("Verifying guesses...")
        energy_dict = opt_minimum_mols.energy
        sorted_index = [k for k, v in sorted(energy_dict.items(), key=lambda item: item[1])]  # Order by energy
        for idx in sorted_index:
            energy = opt_minimum_mols.energy[idx]
            if energy >= mol.energy[id]:
                self.logger.info("Sampler doesn't find conformer with lower energy!! Using original result...")
                return mol

            opt_minimum_mols.KeepIDs = {i: False for i in range(opt_minimum_mols.GetNumConformers())}  # map ids of generated guesses thru workflow
            opt_minimum_mols.KeepIDs[idx] = True
            for verifier in self.verifiers:
                if rxn_smiles:
                    verifier(
                        opt_minimum_mols,
                        multiplicity=multiplicity,
                        save_dir=ts_conf_dir,
                        rxn_smiles=rxn_smiles,
                    )
                else:
                    raise NotImplementedError(f'{verifier} not supported for non-TS conformers.')

            if opt_minimum_mols.KeepIDs[idx]:
                self.logger.info(f"Sampler finds conformer with lower energy. The energy decreases {mol.energy[id] - energy} kcal/mol.")
                mol.GetConformer(id).SetPositions(opt_minimum_mols.GetConformer(idx).GetPositions())
                mol.energy[id] = energy
                mol.frequency[id] = opt_minimum_mols.frequency[idx]
                return mol

        self.logger.info("Sampler doesn't find conformer with lower energy!! Using original result...")


def get_separable_angle_list(
    samplings: Union[List, Tuple], from_angles: Optional[Union[List, Tuple]] = None
) -> List[List]:
    """
    Get a angle list for each input dimension. For each dimension
    The input can be a int, indicating the angles will be evenly sampled;
    Or a list, indicate the angles to be sampled;
    Examples:
    [[120, 240,], 4, 0] => [[120, 240],
                            [0, 90, 180, 270],
                            [0]]
    List of lists are returned for the sake of further calculation

    Args:
        samplings (Union[List, Tuple]): An array of sampling information.
                  For each element, it can be either list or int.
        from_angles (Union[List, Tuple]): An array of initial angles.
                    If not set, angles will begin at zeros.

    Returns:
        list: A list of sampled angles sets.
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


def get_energy(mol: RDKitMol, confId: int = 0, method: str = "GFN2-xTB") -> float:
    """
    Calculate the energy of the `RDKitMol` with given confId. The unit is in kcal/mol.
    Only support methods already suported either in Spharrow or xtb-python.

    Args:
        mol (RDKitMol): A RDKitMol molecule object.
        confId (int): The ID of the conformer for calculating energy. Defaults to 0.
        method (str): Which semiempirical method to be used in running energy calcualtion. Defaults to "GFN2-xTB".

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

def preprocess_energies(energies: np.ndarray):
    """
    Rescale the energy based on the lowest energy.

    Args:
        energies (np.ndarray): A np.ndarray containing the energies for each sampled point.

    Returns:
        The rescaled energies and the mask pointing out positions having values
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

def get_step_to_adjacent_points(
    fsize: int, dim: int = 2, cutoff: float = np.inf
) -> "generator":
    """Get a generator containig the adjacent points."""
    one_d_points = list(range(-fsize, fsize + 1))
    var_combinations = product(*[one_d_points] * dim)
    for points in var_combinations:
        dist = np.linalg.norm(np.array(points))
        if dist <= cutoff:
            yield points


def get_adjacent_energy(coord: List[Tuple], energies: np.ndarray) -> float:
    """Get the energies of adjacent points."""
    try:
        return energies[coord]
    except IndexError:
        new_coord = tuple(
            x if x < energies.shape[i] else x - energies.shape[i]
            for i, x in enumerate(coord)
        )
        return energies[new_coord]


def compare_to_adjacent_point(
    coord: List[Tuple],
    energies: np.ndarray,
    unchecked_points: List[Tuple],
    filters: List[Tuple],
) -> Tuple:
    """Compare the energy of current point and those of other points."""
    # each element is a coordinate
    new_coords = [tuple(x + var_x for x, var_x in zip(coord, var)) for var in filters]

    # Get the energies of adjacent points
    energies = [get_adjacent_energy(new_coord, energies) for new_coord in new_coords]

    # Sorted
    energies, new_coords = zip(*sorted(zip(energies, new_coords)))

    # Find the current point index and points that has higher energy than this point
    # Will be removed from unchecked points list
    cur_point_ind = new_coords.index(coord)
    for new_coord in new_coords[cur_point_ind:]:
        try:
            unchecked_points.remove(new_coord)
        except ValueError:
            # ValueError if coord_min is not in unchecked_points
            pass
    return new_coords[0]


def search_for_a_minimum(
    coord: Tuple,
    energies: np.ndarray,
    unchecked_points: List[Tuple],
    filters: List[Tuple],
) -> Tuple:
    """Search a minimum on a given PES."""
    while True:
        next_point = compare_to_adjacent_point(
            coord, energies, unchecked_points, filters
        )
        next_point = tuple(
            x if x >= 0 else energies.shape[i] + x for i, x in enumerate(next_point)
        )
        if next_point == coord:
            return coord
        elif next_point not in unchecked_points:
            return
        else:
            coord = next_point


def search_minimum(
    energies: np.ndarray, fsize: int, cutoff: float = np.inf
) -> List[Tuple]:
    """Search all the minimums on a given PES."""
    minimum = []

    dim = len(energies.shape)
    filters = list(get_step_to_adjacent_points(fsize, dim, cutoff))

    oned_points = [list(range(energies.shape[i])) for i in range(dim)]
    unchecked_points = list(product(*oned_points))

    while True:
        if not unchecked_points:
            break
        coord = unchecked_points[np.random.randint(len(unchecked_points))]
        new_min = search_for_a_minimum(coord, energies, unchecked_points, filters)
        if new_min:
            minimum.append(new_min)
    return minimum


def plot_heat_map(
    energies: np.ndarray,
    minimum_points: List[Tuple],
    save_path: str,
    mask: np.ndarray = None,
    detailed_view: bool = False,
    title: str = None,
):
    """Plot and save the heat map of a given PES."""
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
