import math
from typing import Optional, Union, Sequence, List


from rdkit import Chem

from rdmc.conf import RDKitConf
from rdmc.rdtools.conf import (
    add_null_conformer,
    embed_multiple_null_confs,
    embed_conformer,
    embed_multiple_confs,
)
from rdmc.rdtools.dist import has_colliding_atoms
from rdmc.rdtools.compare import is_same_connectivity_conf
from rdmc.rdtools.mol import set_mol_positions


class MolConfMixin:

    def AddNullConformer(
        self,
        confId: Optional[int] = None,
        random: bool = True,
    ) -> None:
        """
        Embed a conformer with atoms' coordinates of random numbers or with all atoms
        located at the origin to the current `RDKitMol`.

        Args:
            confId (int, optional): Which ID to set for the conformer (will be added as the last conformer by default).
            random (bool, optional): Whether set coordinates to random numbers. Otherwise, set to all-zero
                                     coordinates. Defaults to ``True``.
        """
        add_null_conformer(self, confId, random)

    def AlignMol(
        self,
        prbMol: Optional[Chem.Mol] = None,
        refMol: Optional[Chem.Mol] = None,
        prbCid: int = -1,
        refCid: int = -1,
        reflect: bool = False,
        atomMaps: Optional[list] = None,
        maxIters: int = 1000,
        weights: list = [],
        *kwargs,
    ) -> float:
        """
        Align molecules based on a reference molecule. This function will also return the RMSD value for the best alignment.
        When leaving both ``prbMol`` and ``refMol`` blank, the function will align current molecule's conformers, and
        ``PrbCid`` or ``refCid`` must be provided.

        Args:
            refMol (Mol): RDKit molecule as a reference. Should not be provided with ``prbMol``.
            prbMol (Mol): RDKit molecules to align to the current molecule. Should not be provided with ``refMol``.
            prbCid (int, optional): The conformer id to be aligned. Defaults to ``-1``, the first conformation.
            refCid (int, optional): The id of reference conformer. Defaults to ``-1``, the first conformation.
            reflect (bool, optional): Whether to reflect the conformation of the probe molecule.
                                      Defaults to ``False``.
            atomMap (list, optional): A vector of pairs of atom IDs ``(prb AtomId, ref AtomId)``
                                      used to compute the alignments. If this mapping is not
                                      specified, an attempt is made to generate on by substructure matching.
            maxIters (int, optional): Maximum number of iterations used in minimizing the RMSD. Defaults to ``1000``.

        Returns:
            float: Minimized RMSD value.
            list: best atom map.
        """
        if prbMol is not None and refMol is not None:
            raise ValueError(
                "`refMol` and `prbMol` should not be provided simultaneously."
            )
        elif prbMol is None and refMol is None and prbCid == refCid:
            raise ValueError(
                "Cannot match the same conformer for the given molecule. `prbCid` and `refCid` needs"
                "to be different if either `prbMol` or `refMol` is not provided."
            )

        refMol = refMol or self
        prbMol = prbMol or self

        if atomMaps is None:
            atomMaps = [list(enumerate(range(self.GetNumAtoms())))]

        try:
            # Only added in version 2022
            rmsd, _, atom_map = Chem.rdMolAlign.GetBestAlignmentTransform(
                prbMol, refMol, prbCid, refCid, atomMaps, maxIters,
                weights=weights, reflect=reflect, maxIters=maxIters, *kwargs
            )
        except AttributeError:
            rmsd = math.inf
            for atom_map in atomMaps:
                cur_rmsd = Chem.rdMolAlign.AlignMol(
                    prbMol, refMol, prbCid, refCid, atom_map,
                    weights=weights, reflect=reflect, maxIters=maxIters
                )
                if cur_rmsd < rmsd:
                    rmsd = cur_rmsd
                    atom_map = atom_map

        return rmsd, atom_map

    def EmbedMultipleNullConfs(self, n: int = 10, random: bool = True):
        """
        Embed conformers with null or random atom coordinates. This helps the cases where a conformer
        can not be successfully embedded. You can choose to generate all zero coordinates or random coordinates.
        You can set to all-zero coordinates, if you will set coordinates later; You should set to random
        coordinates, if you want to optimize this molecule by force fields (RDKit force field cannot optimize
        all-zero coordinates).

        Args:
            n (int): The number of conformers to be embedded. Defaults to ``10``.
            random (bool, optional): Whether set coordinates to random numbers. Otherwise, set to all-zero
                                     coordinates. Defaults to ``True``.
        """
        embed_multiple_null_confs(self, n, random)

    def EmbedNullConformer(self, random: bool = True):
        """
        Embed a conformer with null or random atom coordinates. This helps the cases where a conformer
        can not be successfully embedded. You can choose to generate all zero coordinates or random coordinates.
        You can set to all-zero coordinates, if you will set coordinates later; You should set to random
        coordinates, if you want to optimize this molecule by force fields (RDKit force field cannot optimize
        all-zero coordinates).

        Args:
            random (bool, optional): Whether set coordinates to random numbers. Otherwise, set to all-zero
                                     coordinates. Defaults to ``True``.
        """
        self.EmbedMultipleNullConfs(n=1, random=random)

    def EmbedConformer(self, all_null: bool = True, **kwargs):
        """
        Embed a conformer to the ``RDKitMol``. This will overwrite current conformers. By default, it
        will first try embedding a 3D conformer; if fails, it then try to compute 2D coordinates
        and use that for the conformer structure; if both approaches fail, and embedding a null
        conformer is allowed, a conformer with all random coordinates will be embedded. The last one is
        helpful for the case where you can use `SetPositions` to set their positions afterward, or if you want to
        optimize the geometry using force fields.

        Args:
            allow_null (bool): If embedding 3D and 2D coordinates fails, whether to embed a conformer
                               with random coordinates, for each atom. Defaults to ``True``.
        """
        embed_conformer(self, all_null, **kwargs)

    def EmbedMultipleConfs(self, n: int = 10, allow_null: bool = True, **kwargs):
        """
        Embed multiple conformers to the ``RDKitMol``. This will overwrite current conformers. By default, it
        will first try embedding a 3D conformer; if fails, it then try to compute 2D coordinates
        and use that for the conformer structure; if both approaches fail, and embedding a null
        conformer is allowed, a conformer with all random coordinates will be embedded. The last one is
        helpful for the case where you can use `SetPositions` to set their positions afterward, or if you want to
        optimize the geometry using force fields.

        Args:
            n (int): The number of conformers to be embedded. Defaults to ``10``.
        """
        embed_multiple_confs(self, n, allow_null=allow_null, **kwargs)

    def GetBestAlign(
        self,
        refMol,
        prbCid: int = 0,
        refCid: int = 0,
        atomMaps: Optional[list] = None,
        maxIters: int = 1000,
    ):
        """
        This is a wrapper function for calling ``AlignMol`` twice, with ``reflect`` to ``True``
        and ``False``, respectively.

        Args:
            refMol (Mol): RDKit molecule as a reference.
            prbCid (int, optional): The conformer id to be aligned. Defaults to ``0``.
            refCid (int, optional): The id of reference conformer. Defaults to ``0``.
            reflect (bool, optional): Whether to reflect the conformation of the probe molecule.
                                      Defaults to ``False``.
            atomMap (list, optional): a vector of pairs of atom IDs ``(probe AtomId, ref AtomId)``
                                      used to compute the alignments. If this mapping is not
                                      specified an attempt is made to generate on by substructure matching.
            maxIters (int, optional): maximum number of iterations used in minimizing the RMSD. Defaults to ``1000``.

        Returns:
            float: RMSD value.
            list: best atom map.
            bool: if reflected conformer gives a better result.
        """
        results = []
        positions = []
        for reflect in [False, True]:
            results.append(
                self.AlignMol(
                refMol=refMol,
                prbCid=prbCid,
                refCid=refCid,
                atomMaps=atomMaps,
                maxIters=maxIters,
                reflect=reflect,
                )
            )
            positions.append(self.GetPositions(confId=prbCid))

        if results[0][0] < results[1][0]:
            rmsd, atom_map, reflect = results[0][0], results[0][1], True
            self.SetPositions(positions[0])
        else:
            rmsd, atom_map, reflect = results[1][0], results[1][1], False

        return rmsd, atom_map, reflect

    def SetPositions(
        self,
        coords: Union[Sequence, str],
        confId: int = 0,
        header: bool = False
    ):
        """
        Set the atom positions to one of the conformer.

        Args:
            coords (sequence): A tuple/list/ndarray containing atom positions;
                               or a string with the typical XYZ formating.
            id (int, optional): Conformer ID to assign the Positions to. Defaults to ``0``.
            header (bool): When the XYZ string has an header. Defaults to ``False``.
        """
        set_mol_positions(self, coords, confId, header)

    def GetEditableConformer(
        self,
        id: int = 0,
    ) -> "RDKitConf":
        """
        Get the embedded conformer according to ID.

        Args:
            id (int): The ID of the conformer to be obtained. The default is ``0``.

        Raises:
            ValueError: Bad id assigned.

        Returns:
            RDKitConf: A conformer corresponding to the ID.
        """
        conformer = self.GetConformer(id)
        rdkitconf = RDKitConf(conformer)
        rdkitconf.SetOwningMol(self)
        return rdkitconf

    def GetConformers(
        self,
        ids: Union[list, tuple] = [0],
        editable: bool = False,
    ) -> List["RDKitConf"]:
        """
        Get the embedded conformers according to IDs.

        Args:
            ids (Union[list, tuple]): The ids of the conformer to be obtained.
                                      The default is ``[0]``.
            editable (bool, optional): Whether to return the customized editable conformer
                                       Defaults to ``False``.

        Raises:
            ValueError: Bad id assigned.

        Returns:
            List[RDKitConf]: A list of conformers corresponding to the IDs.
        """
        if editable:
            return [self.GetEditableConformer(id) for id in ids]
        return [self.GetConformer(id) for id in ids]

    def GetAllConformers(self, editable: bool = False) -> List["Conformer"]:
        """
        Get all of the embedded conformers.

        Args:
            editable (bool): If return the editable conformer.

        Returns:
            List['RDKitConf']: A list all of conformers.
        """
        return self.GetConformers(
            list(range(self.GetNumConformers())),
            editable=editable,
        )

    def HasCollidingAtoms(
        self,
        confId: int = -1,
        threshold: float = 0.4,
        reference: str = "vdw",
    ) -> bool:
        """
        Check whether the molecule has colliding atoms. If the distance between two atoms <= threshold * reference distance,
        the atoms are considered to be colliding.

        Args:
            conf_id (int, optional): The conformer ID of the molecule to get the distance matrix of. Defaults to ``-1``.
            threshold (float): A multiplier applied on the reference matrix . Defaults to ``0.4``.
            reference (str): The reference matrix to use. Defaults to ``"vdw"``.
                            Options:
                            - ``"vdw"`` for reference distance based on van de Waals radius
                            - ``"covalent"`` for reference distance based on covalent radius
                            - ``"bond"`` for reference distance based on bond radius

        Returns:
            bool: Whether the molecule has colliding atoms.
        """
        return has_colliding_atoms(self, confId, threshold, reference)

    def IsSameConnectivityConformer(
        self,
        confId: int = 0,
        backend: str = "openbabel",
        **kwargs,
    ) -> bool:
        """
        Check whether the conformer of the molecule (defined by its spacial coordinates)
        as the same connectivity as the molecule.

        Args:
            confId (int, optional): The conformer ID. Defaults to ``0``.
            backend (str, optional): The backend to use for the comparison. Defaults to ``'openbabel'``.
            **kwargs: The keyword arguments to pass to the backend.

        Returns:
            bool: Whether the conformer has the same connectivity as the molecule.
        """
        return is_same_connectivity_conf(self, confId, backend, **kwargs)
