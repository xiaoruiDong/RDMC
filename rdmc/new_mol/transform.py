from copy import copy
from pathlib import Path
from typing import Optional, Union

from rdkit import Chem

from rdmc.rdtools.conversion.smiles import mol_from_smiles, mol_to_smiles
from rdmc.rdtools.conversion.rmg import mol_from_rmg_mol
from rdmc.rdtools.conversion.xyz import mol_from_xyz, mol_to_xyz
from rdmc.rdtools.obabel import (
    openbabel_mol_to_rdkit_mol as mol_from_openbabel_mol,
    rdkit_mol_to_openbabel_mol as mol_to_openbabel_mol,
)
from rdmc.rdtools.utils import get_fake_module

try:
    from ase import Atoms
except ImportError:
    Atoms = get_fake_module("Atom", "ase")

try:
    import networkx as nx
except ImportError:
    nx = get_fake_module("networkx", "networkx")


class MolFromMixin:

    @classmethod
    def FromSmarts(
        cls,
        smarts: str,
        **kwargs,
    ):
        """
        Convert a SMARTS to a new molecule object.

        Args:
            smarts (str): A SMARTS string of the molecule

        Returns:
            Mol: A molecule object corresponding to the SMARTS.
        """
        mol = Chem.MolFromSmarts(smarts, **kwargs)
        return cls(mol)

    @classmethod
    def FromInchi(
        cls,
        inchi: str,
        removeHs: bool = False,
        addHs: bool = True,
        sanitize: bool = True,
    ):
        """
        Convert a InChI string to a new molecule object.

        Args:
            inchi (str): A InChI string. https://en.wikipedia.org/wiki/International_Chemical_Identifier
            removeHs (bool, optional): Whether to remove hydrogen atoms from the molecule, Due to RDKit implementation,
                                       only effective when sanitize is ``True`` as well. ``True`` to remove.
            addHs (bool, optional): Whether to add explicit hydrogen atoms to the molecule. ``True`` to add.
                                    Only functioning when ``removeHs`` is ``False``.
            sanitize (bool, optional): Whether to sanitize the RDKit molecule, ``True`` to sanitize.

        Returns:
            Mol: A molecule object corresponding to the InChI.
        """
        mol = Chem.inchi.MolFromInchi(inchi, sanitize=sanitize, removeHs=removeHs)
        if not removeHs and addHs:
            mol = Chem.AddHs(mol)
        return cls(mol)

    @classmethod
    def FromRMGMol(
        cls,
        rmgMol: "rmgpy.molecule.Molecule",
        removeHs: bool = False,
        sanitize: bool = True,
    ):
        """
        Convert an RMG ``Molecule`` to a new molecule object.

        Args:
            rmgMol ('rmg.molecule.Molecule'): An RMG ``Molecule`` instance.
            removeHs (bool, optional): Whether to remove hydrogen atoms from the molecule, ``True`` to remove.
            sanitize (bool, optional): Whether to sanitize the RDKit molecule, ``True`` to sanitize.

        Returns:
            Mol: A molecule object corresponding to the RMG Molecule.
        """
        return cls(mol_from_rmg_mol(rmgMol, removeHs, sanitize))

    @classmethod
    def FromXYZ(
        cls,
        xyz: str,
        backend: str = "openbabel",
        header: bool = True,
        sanitize: bool = True,
        embed_chiral: bool = False,
        **kwargs,
    ):
        """
        Convert a xyz string to a new molecule object.

        Args:
            xyz (str): A XYZ String.
            backend (str): The backend used to perceive molecule. Defaults to ``'openbabel'``.
                            Currently, we only support ``'openbabel'`` and ``'xyz2mol'``.
            header (bool, optional): If lines of the number of atoms and title are included.
                                    Defaults to ``True.``
            sanitize (bool): Sanitize the RDKit molecule during conversion. Helpful to set it to ``False``
                            when reading in TSs. Defaults to ``True``.
            embed_chiral: ``True`` to embed chiral information. Defaults to ``True``.
            supported kwargs:
                xyz2mol:
                    - charge: The charge of the species. Defaults to ``0``.
                    - allow_charged_fragments: ``True`` for charged fragment, ``False`` for radical. Defaults to ``False``.
                    - use_graph: ``True`` to use networkx module for accelerate. Defaults to ``True``.
                    - use_huckel: ``True`` to use extended Huckel bond orders to locate bonds. Defaults to ``False``.
                    - forced_rdmc: Defaults to ``False``. In rare case, we may hope to use a tailored
                                    version of the Jensen XYZ parser, other than the one available in RDKit.
                                    Set this argument to ``True`` to force use RDMC's implementation,
                                    which user's may have some flexibility to modify.

            Returns:
                Mol: A molecule object corresponding to the xyz.
        """
        return cls(mol_from_xyz(xyz, backend, header, sanitize, embed_chiral, **kwargs))

    @classmethod
    def FromMolBlock(
        cls,
        molBlock: str,
        removeHs: bool = False,
        sanitize: bool = True,
    ):
        """
        Convert a string containing the Mol block to a molecule object.

        Args:
            MolBlock (str): string containing the Mol block.
            removeHs (bool): Whether or not to remove hydrogens from the input. Defaults to ``False``.
            sanitize (bool): Whether or not to use RDKit's sanitization algorithm to clean input; helpful to set this
                             to ``False`` when reading TS files. Defaults to ``True``.

        Returns:
            Mol: A molecule object corresponding to the Mol block string.
        """
        return cls(Chem.MolFromMolBlock(molBlock, removeHs=removeHs, sanitize=sanitize))

    @classmethod
    def FromOBMol(
        cls,
        obMol: "openbabel.OBMol",
        removeHs: bool = False,
        sanitize: bool = True,
        embed: bool = True,
    ):
        """
        Convert a OpenBabel Mol to a molecule object.

        Args:
            obMol (Molecule): An OpenBabel Molecule object for the conversion.
            removeHs (bool, optional): Whether to remove hydrogen atoms from the molecule, Defaults to ``False``.
            sanitize (bool, optional): Whether to sanitize the RDKit molecule. Defaults to ``True``.
            embed (bool, optional): Whether to embed 3D conformer from OBMol. Defaults to ``True``.

        Returns:
            Mol: A molecule object corresponding to the input OpenBabel Molecule object.
        """
        return cls(mol_from_openbabel_mol(obMol, removeHs, sanitize, embed))

    @classmethod
    def FromSmiles(
        cls,
        smiles: str,
        remove_hs: bool = False,
        add_hs: bool = True,
        sanitize: bool = True,
        allow_cxsmiles: bool = True,
        keep_atom_map: bool = True,
        assign_atom_map: bool = True,
    ):
        """
        Convert a SMILES string to a molecule object.

        Args:
            smiles (str): A SMILES representation of the molecule.
            remove_hs (bool, optional): Whether to remove hydrogen atoms from the molecule, ``True`` to remove.
            add_hs (bool, optional): Whether to add explicit hydrogen atoms to the molecule. ``True`` to add.
                                    Only functioning when removeHs is False.
            sanitize (bool, optional): Whether to sanitize the RDKit molecule, ``True`` to sanitize.
            allow_cxsmiles (bool, optional): Whether to recognize and parse CXSMILES. Defaults to ``True``.
            keep_atom_map (bool, optional): Whether to keep the atom mapping contained in the SMILES. Defaults
                                            Defaults to ``True``.
            assign_atom_map (bool, optional): Whether to assign the atom mapping according to the atom index
                                              if no atom mapping available in the SMILES. Defaults to ``True``.

        Returns:
            Mol: A molecule object corresponding to the SMILES.
        """
        return cls(mol_from_smiles(smiles, remove_hs, add_hs, sanitize, allow_cxsmiles, keep_atom_map, assign_atom_map))

    @classmethod
    def FromXYZFile(
        cls,
        path: str,
        backend: str = "openbabel",
        header: bool = True,
        sanitize: bool = True,
        embed_chiral: bool = False,
        **kwargs,
    ):
        """
        Convert a xyz file to a new molecule object.

        Args:
            path (str): The path to the XYZ file.
            backend (str): The backend used to perceive molecule. Defaults to ``'openbabel'``.
                            Currently, we only support ``'openbabel'`` and ``'xyz2mol'``.
            header (bool, optional): If lines of the number of atoms and title are included.
                                    Defaults to ``True.``
            sanitize (bool): Sanitize the RDKit molecule during conversion. Helpful to set it to ``False``
                            when reading in TSs. Defaults to ``True``.
            embed_chiral: ``True`` to embed chiral information. Defaults to ``True``.
            supported kwargs:
                xyz2mol:
                    - charge: The charge of the species. Defaults to ``0``.
                    - allow_charged_fragments: ``True`` for charged fragment, ``False`` for radical. Defaults to ``False``.
                    - use_graph: ``True`` to use networkx module for accelerate. Defaults to ``True``.
                    - use_huckel: ``True`` to use extended Huckel bond orders to locate bonds. Defaults to ``False``.
                    - forced_rdmc: Defaults to ``False``. In rare case, we may hope to use a tailored
                                    version of the Jensen XYZ parser, other than the one available in RDKit.
                                    Set this argument to ``True`` to force use RDMC's implementation,
                                    which user's may have some flexibility to modify.

            Returns:
                Mol: A molecule object corresponding to the xyz.
        """
        with open(path, "r") as f:
            xyz = f.read()
        return cls(mol_from_xyz(xyz, backend, header, sanitize, embed_chiral, **kwargs))

    @classmethod
    def MolFromSDFFile(
        cls,
        path: str,
        removeHs: bool = False,
        sanitize: bool = True,
        sameMol: bool = False,
    ):
        """
        Convert a SDF file to a new molecule object.

        Args:
            path (str): The path to the SDF file.
            removeHs (bool, optional): Whether to remove hydrogen atoms from the molecule, ``True`` to remove.
            sanitize (bool, optional): Whether to sanitize the RDKit molecule, ``True`` to sanitize.
            sameMol (bool, optional):  Whether or not all the conformers in the sdf file are for the same mol, in which case
                                       Only one molecule object is returned, otherwise, a list of molecule objects are returned.
                                       Defaults to ``False``.

        Returns:
            Mol: A molecule object corresponding to the SDF file if ``sameMol`` is True, otherwise a list of molecule objects.
        """
        suppl = Chem.SDMolSupplier(path, removeHs=removeHs, sanitize=sanitize)

        if not sameMol:
            return [cls(m) for m in suppl]


        new_mol = copy(suppl[0])
        for m in suppl:
            new_mol.AddConformer(m.GetConformer(), assignId=True)
        return new_mol

    @classmethod
    def FromFile(
        cls,
        path: str,
        **kwargs,
    ):
        """
        Convert a file to a new molecule object. The file type is determined by its suffix.
        For the detailed guidance of available keyword arguments, check the keyword arguments
        for the specific From{}File API. Currently it supports "xyz" and "sdf" file.

        Args:
            path (str): The path to the file.

        Returns:
            Mol: A molecule object corresponding to the file.
        """
        extension = Path(path).suffix.lower()

        if path.endswith(".xyz"):
            return cls.FromXYZFile(path, **kwargs)
        elif path.endswith(".sdf"):
            return cls.MolFromSDFFile(path, **kwargs)
        else:
            raise NotImplementedError(f"File type {extension} is not supported.")


class MolToMixin:

    """This class is a mixin intended to work with Mol object and its child classes"""

    def ToOBMol(self) -> "openbabel.OBMol":
        """
        Convert a molecule object to a ``OBMol``.

        Returns:
            OBMol: The corresponding openbabel ``OBMol``.
        """
        return mol_to_openbabel_mol(self)

    def ToRWMol(self) -> Chem.RWMol:
        """
        Convert a molecule object to a ``RWMol``.

        Returns:
            RWMol: The corresponding rdkit ``RWMol``.
        """
        return Chem.RWMol(self)

    def ToMol(self) -> Chem.Mol:
        """
        Convert a molecule object to a ``Mol``.

        Returns:
            Mol: The corresponding rdkit ``Mol``.
        """
        try:
            self.GetMol()
        except AttributeError:
            return Chem.Mol(self)

    def ToMolBlock(
        self,
        confId: int = -1,
        includeStereo: bool = True,
        kekulize: bool = True,
    ) -> str:
        """
        Convert a molecule object to a Mol block string. The defaults are consistent with Chem.MolToMolblock.

        Args:
            confId (int, optional): The conformer ID to be converted. Defaults to ``-1``.
            includeStereo (bool, optional): Whether to include stereo information in the Mol block. Defaults to ``True``.
            kekulize (bool, optional): Whether to kekulize the molecule. Defaults to ``True``.

        Returns:
            str: The corresponding Mol block string.
        """
        return Chem.MolToMolBlock(self, confId=confId, includeStereo=includeStereo, kekulize=kekulize)

    def ToInchi(
        self,
        options: str = "",
    ) -> str:
        """
        Convert a molecule object to an InChI string using RDKit builtin converter.

        Args:
            options (str, optional): The InChI generation options. Options should be
                                     prefixed with either a - or a / Available options are explained in the
                                     InChI technical FAQ: https://www.inchi-trust.org/technical-faq/#15.14 and
                                     https://www.inchi-trust.org/?s=user+guide. Defaults to "".

        Returns:
            str: The corresponding InChI string.
        """
        return Chem.rdinchi.MolToInchi(self, options=options)[0]

    def ToXYZ(
        self,
        confId: int = -1,
        header: bool = True,
        comment: str = "",
    ) -> str:
        """
        Convert a molecule object to a XYZ string.

        Args:
            conf_id (int, optional): The index of the conformer to be converted. Defaults to ``-1``, exporting the XYZ of the first conformer.
            header (bool, optional): If lines of the number of atoms and title are included. Defaults to ``True``.
            comment (str, optional): The comment to be added. Defaults to ``''``.

        Returns:
            str: The corresponding XYZ string.
        """
        return mol_to_xyz(self, confId, header, comment)

    def ToSmarts(
        self,
        isomericSmiles: bool = True,
        rootedAtAtom: int = -1,
    ) -> str:
        """
        Convert a molecule object to a SMARTS string.

        Args:
            isomericSmiles (bool, optional): Whether to generate isomeric SMILES and include information about stereochemistry.
                                             Defaults to ``True``.
            rootedAtAtom (int, optional): The atom index to be used as the root of the SMARTS. Defaults to ``-1``.

        Returns:
            str: The corresponding SMARTS string.
        """
        return Chem.MolToSmarts(self, isomericSmiles, rootedAtAtom)

    def ToSDFFile(
        self,
        path: str,
        confId: Optional[Union[int, list]] = None,
    ):
        """
        Convert a molecule object to a SDF file.

        Args:
            path (str): The path to the SDF file.
            confId (int or list, optional): The conformer ID to be converted. Defaults to ``None``, to write all conformers to the sdf file.
                                             If ``confId`` is a list or a int, it will writes the specified conformers
        """
        if confId is None:
            confId = range(self.GetNumConformers())
        elif isinstance(confId, int):
            confId = [confId]

        try:
            writer = Chem.rdmolfiles.SDWriter(str(path))
            for i in confId:
                writer.write(self, confId=i)
        except:
            pass
        finally:
            writer.close()

    def ToSmiles(
        self,
        stereo: bool = True,
        kekule: bool = False,
        canonical: bool = True,
        removeAtomMap: bool = True,
        removeHs: bool = True,
    ) -> str:
        """
        Convert a molecule object to a SMILES string.

        Args:
            stereo (bool, optional): Whether to include stereochemistry information in the SMILES. Defaults to ``True``.
            kekule (bool, optional): Whether to use Kekule encoding. Defaults to ``False``.
            canonical (bool, optional): Whether to use canonical SMILES. Defaults to ``True``.
            remove_atom_map (bool, optional): Whether to keep the Atom mapping contained in the SMILES. Defaults
                Defaults to ``True``.
            remove_hs (bool, optional): Whether to remove hydrogen atoms from the molecule. Defaults to ``True``.

        Returns:
            str: A SMILES string corresponding to the molecule.
        """
        return mol_to_smiles(self, stereo, kekule, canonical, removeAtomMap, removeHs)

    def ToAtoms(
        self,
        confId: int = 0,
    ) -> "Atoms":
        """
        Convert a molecule object to the ``ase.Atoms`` object.

        Args:
            confId (int): The conformer ID to be exported. Defaults to ``0``.

        Returns:
            Atoms: The corresponding ``ase.Atoms`` object.
        """
        atoms = Atoms(
            positions=self.GetPositions(id=confId),
            numbers=self.GetAtomicNumbers()
        )
        atoms.set_initial_magnetic_moments(
            [atom.GetNumRadicalElectrons() + 1 for atom in self.GetAtoms()]
        )
        atoms.set_initial_charges(
            [atom.GetFormalCharge() for atom in self.GetAtoms()]
        )
        return atoms

    def ToGraph(
        self,
        keep_bond_order: bool = False,
    ) -> "nx.Graph":
        """
        Convert a molecule object to a networkx graph.

        Args:
            keep_bond_order (bool): Whether to keep bond order information. Defaults to ``False``,
                                    meaning treat all bonds as single bonds.

        Returns:
            nx.Graph: A networkx graph representing the molecule.
        """
        nx_graph = nx.Graph()
        for atom in self.GetAtoms():
            nx_graph.add_node(
                atom.GetIdx(), symbol=atom.GetSymbol(), atomic_num=atom.GetAtomicNum()
            )

        for bond in self.GetBonds():
            bond_type = 1 if not keep_bond_order else bond.GetBondTypeAsDouble()
            nx_graph.add_edge(
                bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond_type
            )

        return nx_graph
