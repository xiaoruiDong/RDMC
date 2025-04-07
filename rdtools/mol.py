import copy
from collections import Counter
from functools import lru_cache
from itertools import product as cartesian_product
from typing import List, Optional, Union, Sequence

import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Geometry.rdGeometry import Point3D

from rdtools.atommap import has_atom_map_numbers
from rdtools.atom import get_atom_mass, increment_radical
from rdtools.conf import (
    add_null_conformer,
    embed_multiple_null_confs,
    reflect as _reflect,
    set_conformer_coordinates,
)
from rdtools.conversion.xyz import xyz_to_coords


def get_spin_multiplicity(mol: Chem.Mol) -> int:
    """
    Get spin multiplicity of a molecule. The spin multiplicity is calculated
    using Hund's rule of maximum multiplicity defined as 2S + 1.

    Args:
        mol (Chem.Mol): The molecule to get spin multiplicity.

    Returns:
        int : Spin multiplicity.
    """
    return 1 + Descriptors.NumRadicalElectrons(mol)


def get_formal_charge(mol: Chem.Mol) -> int:
    """
    Get formal charge of a molecule.

    Args:
        mol (Chem.Mol): The molecule to get formal charge.

    Returns:
        int : Formal charge.
    """
    return Chem.GetFormalCharge(mol)


def get_mol_weight(
    mol: Chem.Mol,
    exact: bool = False,
    heavy_atoms: bool = False,
) -> float:
    """
    Get the molecule weight.

    Args:
        mol (Chem.Mol): The molecule to get the weight.
        exact (bool, optional): If ``True``, the exact weight is returned.
            Otherwise, the average weight is returned. Defaults to ``False``.
        heavy_atoms (bool, optional): If ``True``, the weight is calculated using only heavy atoms.
            Otherwise, the weight is calculated using all atoms. Defaults to ``False``.

    Returns:
        float: The weight of the molecule.
    """
    if heavy_atoms:
        return Descriptors.HeavyAtomMolWt(mol)
    if exact:
        return Descriptors.ExactMolWt(mol)
    return Descriptors.MolWt(mol)


def get_heavy_atoms(mol: Chem.Mol) -> list:
    """
    Get heavy atoms of a molecule.

    Args:
        mol (Chem.Mol): The molecule to get heavy atoms.

    Returns:
        list: the list of heavy atoms.
    """
    return [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1]


def get_element_symbols(mol: Chem.Mol) -> List[str]:
    """
    Get element symbols of a molecule.

    Args:
        mol (Chem.Mol): The molecule to get element symbols.

    Returns:
        List[str] : List of element symbols (e.g. ``["H", "C", "O",]`` etc.)
    """
    return [atom.GetSymbol() for atom in mol.GetAtoms()]


def get_atomic_nums(mol: Chem.Mol) -> List[int]:
    """
    Get atomic numbers of a molecule.

    Args:
        mol (Chem.Mol): The molecule to get atomic numbers.

    Returns:
        List[int] : List of atomic numbers (e.g. ``[1, 6, 8, ]`` etc.)
    """
    return [atom.GetAtomicNum() for atom in mol.GetAtoms()]


def get_atom_masses(mol: Chem.Mol) -> List[float]:
    """
    Get atomic masses of a molecule.

    Args:
        mol (Chem.Mol): The molecule to get atomic masses.

    Returns:
        List[float] : List of atomic masses (e.g. ``[1.008, 12.01, 15.999, ]`` etc.)
    """
    return [get_atom_mass(atom) for atom in mol.GetAtoms()]


def get_element_counts(mol: Chem.Mol) -> dict:
    """
    Get element counts of a molecule.

    Args:
        mol (Chem.Mol): The molecule to get element counts.

    Returns:
        dict: {"element_symbol": count}
    """
    return dict(Counter(get_element_symbols(mol)))


def combine_mols(
    mol1: Chem.Mol,
    mol2: Chem.Mol,
    offset: Optional[np.ndarray] = None,
    c_product: bool = False,
):
    """
    Combine two molecules (``mol1`` and ``mol2``).
    A new object instance will be created and changes are not made to the current molecule.

    Args:
        mol1 (Chem.Mol): The current molecule.
        mol2 (Chem.Mol): The molecule to be combined.
        offset (np.ndarray, optional): The offset to be added to the coordinates of ``mol2``. It should be a length-3 array.
                                       This is not used when any of the molecules has 0 conformer. Defaults to ``None``.
        c_product (bool, optional): If ``True``, generate conformers for every possible combination
                                    between the current molecule and the ``molFrag``. E.g.,
                                    (1,1), (1,2), ... (1,n), (2,1), ...(m,1), ... (m,n). :math:`N(conformer) = m \\times n.`

                                    Defaults to ``False``, meaning only generate conformers according to
                                    (1,1), (2,2), ... When ``c_product`` is set to ``False``, if the current
                                    molecule has 0 conformer, conformers will be embedded to the current molecule first.
                                    The number of conformers of the combined molecule will be equal to the number of conformers
                                    of ``molFrag``. Otherwise, the number of conformers of the combined molecule will be equal
                                    to the number of conformers of the current molecule. Some coordinates may be filled by 0s,
                                    if the current molecule and ``molFrag`` have different numbers of conformers.

    Returns:
        Chem.Mol: The combined molecule.
    """
    vector = Point3D()
    if not c_product and offset is not None:
        for i, coord in enumerate("xyz"):
            vector.__setattr__(coord, float(offset[i]))

    combined_mol = Chem.CombineMols(mol1, mol2, vector)
    if c_product:
        if offset is None:
            offset = np.zeros(3)
        c1s, c2s = mol1.GetConformers(), mol2.GetConformers()
        pos_list = [
            [c1.GetPositions(), c2.GetPositions() + offset]
            for c1, c2 in cartesian_product(c1s, c2s)
        ]
        if len(pos_list) > 0:
            embed_multiple_null_confs(combined_mol, len(pos_list), random=False)
            for i, pos in enumerate(pos_list):
                conf = combined_mol.GetConformer(i)
                set_conformer_coordinates(conf, np.concatenate(pos))

    return combined_mol


def force_no_implicit(mol: Chem.Mol):
    """
    Set no implicit hydrogen for atoms without implicit/explicit hydrogens. When
    manipulating molecules by changing number of radical electrons / charges and then updating the cached properties,
    additional hydrogens may be added to the molecule. This function helps avoid this problem.
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1 and not atom.GetTotalNumHs():
            atom.SetNoImplicit(True)


def reflect(mol: Chem.Mol, conf_id: int = 0):
    """
    Reflect the coordinates of the conformer of the molecule.

    Args:
        mol (Chem.Mol): The molecule to reflect.
        conf_id (int, optional): The conformer ID to reflect.
    """
    conf = mol.GetConformer(conf_id)
    _reflect(conf)


def fast_sanitize(mol: Chem.RWMol):
    """
    Only update the molecule's property and ring perception.
    """
    Chem.SanitizeMol(
        mol,
        sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES
        | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
    )


def get_writable_copy(mol: Chem.Mol):
    """
    Get an writable (editable) copy of the current molecule object.
    If it is an RWMol, then return a copy with the same type. Otherwise,
    return a copy in RWMol type.

    Args:
        mol(Mol): molecule to copy from.

    Returns:
        RWMol: The writable copy of the molecule
    """
    if isinstance(mol, Chem.RWMol):
        return copy.copy(mol)  # Write in this way to support RDKitMol defined in rdmc
    else:
        return Chem.RWMol(mol)


def get_closed_shell_mol(
    mol: Chem.RWMol,
    sanitize: bool = True,
    explicit: bool = True,
    max_num_hs: int = 12,
) -> "Chem.RWMol":
    """
    Get a closed shell molecule by removing all radical electrons and adding
    H atoms to these radical sites. This method currently only work for radicals
    and will not work properly for singlet radicals (number of radical electrons = 0).

    Args:
        mol (Chem.RWMol): The radical molecule
        sanitize (bool): Whether to sanitize the molecule. Defaults to ``True``.
        explicit (bool): Whether to use add H atoms explicitly.
                Defaults to ``True``, which is best used with radical molecules with
                hydrogen atoms explicitly defined. Setting it to ``False``, if the
                hydrogen atoms are implicitly defined.
        max_num_hs (int, optional): The max number of Hs to add on a single atom. This option allows partial
            saturation for a bi- or tri-radical site. E.g., [CH] => [CH3].
            Defaults to ``12``, which is equivalent to add as many Hs as possible to the radical site.

    Returns:
        RDKitMol: A closed shell molecule.
    """
    mol = get_writable_copy(mol)

    if explicit:
        return get_closed_shell_explicit(mol, sanitize, max_num_hs=max_num_hs)
    else:
        return get_closed_shell_implicit(mol, sanitize, max_num_hs=max_num_hs)


def get_closed_shell_explicit(
    mol: Chem.RWMol,
    sanitize: bool = True,
    max_num_hs: int = 12,
) -> Chem.RWMol:
    """
    Get the closed shell molecule of a radical molecule by explicitly adding
    hydrogen atoms to the molecule.

    Args:
        mol (Chem.RWMol): The radical molecule.
        sanitize (bool, optional): Whether to sanitize the molecule. Defaults to ``True``.
        max_num_hs (int, optional): The max number of Hs to add on a single atom. This option allows partial
            saturation for a bi- or tri-radical site. E.g., [CH] => [CH3].
            Defaults to ``12``, which is equivalent to add as many Hs as possible to the radical site.

    Returns:
        Chem.RWMol: The closed shell molecule.
    """

    h_atom_idx, num_orig_atoms = mol.GetNumAtoms(), mol.GetNumAtoms()

    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        num_rad_elecs = atom.GetNumRadicalElectrons()
        if num_rad_elecs:
            for _ in range(min(num_rad_elecs, max_num_hs)):
                mol.AddAtom(Chem.Atom(1))
                mol.AddBond(h_atom_idx, atom_idx, Chem.BondType.SINGLE)
                h_atom_idx += 1
                num_rad_elecs -= 1
            atom.SetNumRadicalElectrons(num_rad_elecs)

    if has_atom_map_numbers(mol):
        for atom_idx in range(num_orig_atoms, h_atom_idx):
            mol.GetAtomWithIdx(atom_idx).SetAtomMapNum(atom_idx + 1)

    if mol.GetNumConformers():
        # Set coordinates to the added H atoms
        for atom_idx in range(num_orig_atoms, h_atom_idx):
            Chem.rdmolops.SetTerminalAtomCoords(
                mol, atom_idx, mol.GetAtomWithIdx(atom_idx).GetNeighbors()[0].GetIdx()
            )

    if sanitize:
        fast_sanitize(mol)
    return mol


def get_closed_shell_implicit(
    mol: Chem.Mol,
    sanitize: bool = True,
    max_num_hs: int = 12,
) -> Chem.Mol:
    """
    Get the closed shell molecule of a radical molecule. This only adds Hs implicitly
    and no new atom is actually added to the molecule. This is suggested for the molecule
    objects with (most) H atoms are not explicitly defined.

    Args:
        mol (Chem.Mol): The radical molecule.
        sanitize (bool, optional): Whether to sanitize the molecule. Defaults to ``True``.
        max_num_hs (int, optional): The max number of Hs to add on a single atom. This option allows partial
            saturation for a bi- or tri-radical site. E.g., [CH] => [CH3].
            Defaults to ``12``, which is equivalent to add as many Hs as possible to the radical site.

    Returns:
        Chem.Mol: The closed shell molecule.
    """

    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons():
            atom.SetNumRadicalElectrons(
                max(0, atom.GetNumRadicalElectrons() - max_num_hs)
            )
            atom.SetNoImplicit(False)

    if sanitize:
        fast_sanitize(mol)
    return mol


@lru_cache(maxsize=1)
def get_heavy_hydrogen_query():
    return Chem.MolFromSmarts("[*]-[H]")


def get_dehydrogenated_mol(
    mol,
    kind: str = "radical",
    once_per_heavy: bool = True,
    only_on_atoms: Optional[list] = None,
) -> list:
    """
    Generate the molecules that have one less hydrogen atom compared to the reference molecule.
    This function only supports molecules that have H atoms explicitly defined. Note, this function
    doesn't filter out equivalent structures.

    Args:
        mol (Chem.Mol): The reference molecule
        kind (str, optional): The kind of generated molecules. The available options are
            "radical", "cation", and "anion".
        once_per_heavy (bool, optional): There can be multiple Hs on a single heavy atom.
            By setting this argument to ``True``, the function will only remove H atom
            once per heavy atoms. Otherwise, the function will comprehensively generate
            dehydrogenated molecule by remove every single H atoms. Defaults to ``True``.
        only_on_atoms (list, optional): This argument allows only operating on specific atoms.
            Defaults to None, operating on all atoms.

    Returns:
        list: a list of dehydrogenated molecules
    """
    tpl = get_heavy_hydrogen_query()
    hvy_h_pairs = mol.GetSubstructMatches(tpl, maxMatches=mol.GetNumAtoms())

    new_mols = []
    explored_hvy_atoms = set()
    for hvy_idx, h_idx in hvy_h_pairs:
        if only_on_atoms and hvy_idx not in only_on_atoms:
            continue
        elif once_per_heavy and hvy_idx in explored_hvy_atoms:
            continue
        elif once_per_heavy:
            explored_hvy_atoms.add(hvy_idx)

        new_mol = get_writable_copy(mol)
        hvy_atom = new_mol.GetAtomWithIdx(hvy_idx)

        if kind == "radical":
            increment_radical(hvy_atom)
        elif kind == "cation":
            hvy_atom.SetFormalCharge(hvy_atom.GetFormalCharge() + 1)
        elif kind == "anion":
            hvy_atom.SetFormalCharge(hvy_atom.GetFormalCharge() - 1)
        new_mol.RemoveAtom(h_idx)

        fast_sanitize(new_mol)

        new_mols.append(new_mol)

    return new_mols


def set_mol_positions(
    mol: Chem.Mol,
    coords: Union[Sequence, str],
    conf_id: int = 0,
    header: bool = False,
):
    """
    Set the positions of atoms to one of the conformer.

    Args:
        mol (Mol): The molecule object to change positions.
        coords (Union[sequence, str]): A tuple/list/ndarray containing atom positions;
                                       or a string with the typical XYZ formating.
        confId (int, optional): Conformer ID to assign the Positions to. Defaults to ``0``.
        header (bool): Whether the XYZ string has an header, if feeding in XYZ. Defaults to ``False``.
    """
    if isinstance(coords, str):
        coords = xyz_to_coords(coords, header=header)
    try:
        conf = mol.GetConformer(conf_id)
    except ValueError:
        if conf_id == 0:
            add_null_conformer(mol, conf_id=0, random=False)
            set_conformer_coordinates(mol.GetConformer(0), coords)
        else:
            raise ValueError(f"Conformer {conf_id} does not exist")
    else:
        set_conformer_coordinates(conf, coords)


def get_mol_weight(mol, heavy: bool = False, exact: bool = False) -> float:
    """
    Get molecular weight of the molecule.

    Args:
        heavy (bool, optional): Whether to ignore weight of the hydrogens. Defaults to ``False``.
        exact (bool, optional): Whether to use exact molecular weight (distinguishing isotopes). Defaults to ``False``.

    Returns:
        float : Molecular weight.
    """
    if not heavy and not exact:
        return Chem.Descriptors.MolWt(mol)
    elif not heavy and exact:
        return Chem.Descriptors.ExactMolWt(mol)
    elif exact:
        raise NotImplementedError
    else:
        return Chem.Descriptors.HeavyAtomMolWt(mol)
