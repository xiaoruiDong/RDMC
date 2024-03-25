import copy
from typing import Optional, Sequence, List, Tuple, Union

from rdkit import Chem

from rdtools.torsion import get_torsion_tops
from rdtools.atommap import renumber_atoms
from rdtools.bond import add_bonds
from rdtools.mol import get_closed_shell_mol, combine_mols


class MolOpsMixin:
    """This mixin includes methods that manipulate a molecule and returns a new molecule object"""

    def GetMolFrags(
        self,
        asMols: bool = False,
        sanitize: bool = True,
        frags: Optional[list] = None,
        fragsMolAtomMapping: Optional[list] = None,
    ) -> Tuple["Mol"]:
        """
        Finds the disconnected fragments from a molecule. For example, for the molecule "CC(=O)[O-].[NH3+]C",
        this function will split the molecules into a list of "CC(=O)[O-]" and "[NH3+]C". By defaults,
        this function will return a list of atom mapping, but options are available for getting mols.

        Args:
            asMols (bool, optional): Whether the fragments will be returned as molecules instead of atom IDs.
                                     Defaults to ``True``.
            sanitize (bool, optional): Whether the fragments molecules will be sanitized before returning them.
                                       Defaults to ``True``.
            frags (list, optional): If this is provided as an empty list, the result will be ``mol.GetNumAtoms()``
                                    long on return and will contain the fragment assignment for each Atom.
            fragsMolAtomMapping (list, optional): If this is provided as an empty list (``[]``), the result will be a
                                                  numFrags long list on return, and each entry will contain the
                                                  indices of the Atoms in that fragment: [(0, 1, 2, 3), (4, 5)].

        Returns:
            tuple: a tuple of atom mapping or a tuple of split molecules (RDKitMol).
        """
        mol_frags = Chem.GetMolFrags(
            self,
            asMols=asMols,
            sanitizeFrags=sanitize,
            frags=frags,
            fragsMolAtomMapping=fragsMolAtomMapping,
        )
        if asMols and isinstance(self, Chem.RWMol):
            return tuple(self.__class__(mol) for mol in mol_frags)
        return mol_frags

    def GetTorsionTops(
        self,
        torsion: Sequence,
        allowNonBondPivots: bool = False,
    ) -> tuple:
        """
        Generate tops for the given torsion. Top atoms are defined as atoms on one side of the torsion.
        The mol should be in one-piece when using this function, otherwise, the results will be
        misleading.

        Args:
            torsion (Sequence): The atom indices of the pivot atoms (length of 2) or
                a length-4 atom index sequence with the 2nd and 3rd are the pivot of the torsion.
            allowNonBondPivots (bool, optional): Allow pivots that are not directly bonded.
                Defaults to ``False``. There are cases like CC#CC or X...H...Y, where a user may want
                to define a torsion with a nonbonding pivots.

        Returns:
            tuple: Two frags, one of the top of the torsion, and the other top of the torsion.
        """
        return get_torsion_tops(self, torsion, allowNonBondPivots)

    def AddBonds(
        self,
        bonds: List[Tuple[int, int]],
        bondTypes: Optional[List[Chem.BondType]] = None,
        UpdateProperties: bool = True,
        inplace: bool = True,
    ) -> "Mol":
        """
        Add bonds to a molecule.

        Args:
            mol (Chem.RWMol): The molecule to be added.
            bond (tuple): The atom index of the bond to be added.
            bond_type (Chem.BondType, optional): The bond type to be added. Defaults to ``Chem.BondType.SINGLE``.
            update_properties (bool, optional): Whether to update the properties of the molecule. Defaults to ``True``.

        Returns:
            Mol: The molecule with bonds added.
        """
        return add_bonds(self, bonds, bondTypes, UpdateProperties, inplace)  # use copy.copy inside

    def GetClosedShellMol(
        self,
        cheap: bool = False,
        sanitize: bool = True,
    ) -> "Mol":
        """
        Get a closed shell molecule by removing all radical electrons and adding
        H atoms to these radical sites. This method currently only work for radicals
        and will not work properly for singlet radicals.

        Args:
            cheap (bool): Whether to use a cheap method where H atoms are only implicitly added.
                          Defaults to ``False``. Setting it to ``False`` only when the molecule
                          is immediately used for generating SMILES/InChI and other representations,
                          and no further manipulation is needed. Otherwise, it may be confusing as
                          the hydrogen atoms will not appear in the list of atoms, not display in the
                          2D graph, etc.
            sanitize (bool): Whether to sanitize the molecule. Defaults to ``True``.

        Returns:
            Mol: A closed shell molecule.
        """
        return get_closed_shell_mol(self, sanitize, cheap)  # use copy.copy inside

    def CombineMol(
        self,
        molFrag: "Mol",
        offset: Optional["np.ndarray"] = None,
        c_product: bool = False,
    ) -> "Mol":
        """
        Combine the current molecule with the given ``molFrag`` (another molecule
        or fragment). A new object instance will be created and changes are not made to the current molecule.

        Args:
            molFrag (RDKitMol or Mol): The molecule or fragment to be combined into the current one.
            offset (np.ndarray, optional): A 3-element vector used to define the offset. Defaults to None, for (0, 0, 0)
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
            RDKitMol: The combined molecule.
        """
        combined_mol = combine_mols(self, molFrag, offset, c_product)
        if not isinstance(self, Chem.RWMol):
            return combined_mol
        return self.__class__(combined_mol)

    def Copy(
        self,
        quickCopy: bool = False,
        confId: int = -1,
        copy_attrs: Optional[list] = None,
        deep: bool = False,
    ):
        """
        Copy the current molecule.

        Args:
            quickCopy (bool, optional): Whether to do a quick copy. Defaults to ``False``. A quick copy will neglect
                                        conformation, substance group, and bookmarks.
            confId (int): The conformer ID to be copied over. You may copy one of the conformers of the current molecule
                          by setting ``confId`` to the conformer ID; or you can copy all conformers by setting ``confId``
                          to ``-1``. Defaults to ``-1``. This argument is only valid when ``quickCopy`` is ``False``.
            copy_attrs (list, optional): A list of attributes to be copied. Defaults to ``None``.
            deep (bool, optional): Whether to do a deep copy. Defaults to ``False``.

        Returns:
            Mol: The copied molecule.
        """
        if not quickCopy and confId == -1:
            return copy.deepcopy(self) if deep else copy.copy(self)

        mol_copy = self.__class__(self, quickCopy, confId)
        if copy_attrs is not None:
            for attr in copy_attrs:
                if deep:
                    mol_copy.__setattr__(attr, copy.deepcopy(self.__getattribute__(attr)))
                else:
                    mol_copy.__setattr__(attr, copy.copy(self.__getattribute__(attr)))
        return mol_copy

    def RenumberAtoms(
        self,
        newOrder: Optional[Union[dict, list]] = None,
        updateAtomMap: bool = True,
    ) -> "Mol":
        """
        Return a new copy of Mol that has atom (index) reordered.

        Args:
            newOrder (list or dict, optional): The new ordering the atoms (should be numAtoms long).
                                               - If provided as a list, it should a list of atom indexes.
                                               E.g., if newOrder is ``[3,2,0,1]``, then atom ``3``
                                               in the original molecule will be atom ``0`` in the new one.
                                               - If provided as a dict, it should be a mapping between atoms. E.g.,
                                               if newOrder is ``{0: 3, 1: 2, 2: 0, 3: 1}``, then atom ``0`` in the
                                               original molecule will be atom ``3`` in the new one. Unlike the list case,
                                               the newOrder can be a partial mapping, but one should make sure all the pairs
                                               are included. E.g., ``{0: 3, 3: 0}``.
                                               - If no value provided (default), then the molecule
                                               will be renumbered based on the current atom map numbers. The latter is helpful
                                               when the sequence of atom map numbers and atom indexes are inconsistent.
            updateAtomMap (bool): Whether to update the atom map number based on the
                                  new order. Defaults to ``True``.

        Returns:
            Mol: Molecule with reordered atoms.
        """
        return renumber_atoms(self, newOrder, updateAtomMap)
