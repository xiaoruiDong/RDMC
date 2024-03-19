from typing import Dict, List

from rdkit import Chem

from rdmc.rdtools.atommap import get_atom_map_numbers
from rdmc.rdtools.dist import get_adjacency_matrix, get_distance_matrix
from rdmc.rdtools.featurizer import get_fingerprint
from rdmc.rdtools.mol import (
    get_atomic_nums,
    get_atom_masses,
    get_element_counts,
    get_element_symbols,
    get_heavy_atoms,
    get_spin_multiplicity,
)
from rdmc.rdtools.bond import get_bonds_as_tuples
from rdmc.rdtools.torsion import get_torsional_modes


class MolAttrMixin:

    """
    A mixin used to get extra molecule attributes that are not directly available in Mol's attributes.
    """

    def GetAdjacencyMatrix(self) -> 'np.ndarray':
        """
        Get the adjacency matrix of the molecule.

        Returns:
            numpy.ndarray: A square adjacency matrix of the molecule, where a `1` indicates that atoms are bonded
                           and a `0` indicates that atoms aren't bonded.
        """
        return get_adjacency_matrix(self)

    def GetAtoms(self) -> List['Atom']:
        """
        This is a rewrite of GetAtoms(), based on the findings of `RDKit issue <https://github.com/rdkit/rdkit/issues/6208>`_.
        Although RDKit fixed this issue in version 2023.09, we still keep this function for backward compatibility.

        Returns:
            list: a list of Atoms.
        """
        return [self.GetAtomWithIdx(idx) for idx in range(self.GetNumAtoms())]

    def GetAtomMapNumbers(self):
        """
        Get the atom mapping.

        Returns:
            tuple: atom mapping numbers in the sequence of atom index.
        """
        return tuple(get_atom_map_numbers(self))

    def GetHeavyAtoms(self) -> List['Atom']:
        """
        Get heavy atoms of the molecule with the order consistent with the atom indexes.

        Returns:
            list: A list of heavy atoms.
        """
        return get_heavy_atoms(self)

    def GetAtomicNumbers(self) -> List[int]:
        """
        Get the Atomic numbers of the molecules. The atomic numbers are sorted by the atom indexes.

        Returns:
            list: A list of atomic numbers.
        """
        return get_atomic_nums(self)

    def GetBondsAsTuples(self) -> List[tuple]:
        """
        Generate a list of length-2 sets indicating the bonding atoms in the molecule.

        Returns:
            list: A list of length-2 sets indicating the bonding atoms.
        """
        return get_bonds_as_tuples(self)

    def GetElementSymbols(self) -> List[str]:
        """
        Get the element symbols of the molecules. The element symbols are sorted by the atom indexes.

        Returns:
            list: A list of element symbols.
        """
        return get_element_symbols(self)

    def GetElementCounts(self) -> Dict[str, int]:
        """
        Get the element counts of the molecules.

        Returns:
            dict: A dictionary of element counts.
        """
        return get_element_counts(self)

    def GetAtomMasses(self) -> List[float]:
        """
        Get the mass of each atom. The order is consistent with the atom indexes.

        Returns:
            list: A list of atom masses.
        """
        return get_atom_masses(self.GetAtomicNumbers())

    def GetDistanceMatrix(self, confId: int = 0) -> "np.ndarray":
        """
        Get the distance matrix of the molecule.

        Args:
            confId (int, optional): The conformer ID to extract distance matrix from.
                                    Defaults to ``0``.

        Returns:
            np.ndarray: A square distance matrix of the molecule.
        """
        return get_distance_matrix(self, confId)

    def GetFingerprint(
        self,
        fpType: str = "morgan",
        numBits: int = 2048,
        count: bool = False,
        dtype: str = "int32",
        **kwargs,
    ) -> "np.ndarray":
        """
        Get the fingerprint of the molecule.

        Args:
            fpType (str, optional): The type of the fingerprint. Defaults to ``'morgan'``.
            numBits (int, optional): The number of bits of the fingerprint. Defaults to ``2048``.
            count (bool, optional): Whether to count the number of occurrences of each bit. Defaults to ``False``.

        Returns:
            np.ndarray: A fingerprint of the molecule.
        """
        return get_fingerprint(
            self, fp_type=fpType, num_bits=numBits, count=count, dtype=dtype, **kwargs
        )

    def GetPositions(self, confId: int = 0) -> "np.ndarray":
        """
        Get the positions of the atoms in the molecule.

        Args:
            confId (int, optional): The conformer ID to extract positions from.
                                    Defaults to ``0``.

        Returns:
            np.ndarray: A matrix of shape (N, 3) where N is the number of atoms.
        """
        conf = self.GetConformer(confId)
        return conf.GetPositions()

    def GetSymmSSSR(self) -> tuple:
        """
        Get a symmetrized SSSR.

        Returns:
            tuple: A sequence of sequences containing the rings found as atom IDs.
        """
        return Chem.GetSymmSSSR(self)

    def GetTorsionalModes(
        self,
        excludeMethyl: bool = False,
        includeRings: bool = False,
    ) -> List[tuple]:
        """
        Get all of the torsional modes (rotors) from the molecule.

        Args:
            excludeMethyl (bool): Whether exclude the torsions with methyl groups. Defaults to ``False``.
            includeRings (bool): Whether or not to include ring torsions. Defaults to ``False``.

        Returns:
            list: A list of four-atom-indice to indicating the torsional modes.
        """
        return get_torsional_modes(self, excludeMethyl, includeRings)

    def GetFormalCharge(self) -> int:
        """
        Get formal charge of the molecule.

        Returns:
            int : Formal charge.
        """
        return Chem.GetFormalCharge(self)

    def GetSpinMultiplicity(self) -> int:
        """
        Get spin multiplicity of a molecule. The spin multiplicity is calculated
        using Hund's rule of maximum multiplicity defined as 2S + 1.

        Returns:
            int : Spin multiplicity.
        """
        return get_spin_multiplicity(self)
