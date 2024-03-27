import os
import subprocess

from rdmc import RDKitMol

from rdmc.conformer_generation.ts_verifiers.base import IRCVerifier
from rdmc.external.inpwriter import write_orca_irc
from rdmc.conformer_generation.task.orca import OrcaTask


class OrcaIRCVerifier(IRCVerifier, OrcaTask):
    """
    The class for verifying the TS by calculating and checking its IRC analysis using Orca.

    Args:
        method (str, optional): The method to be used for TS optimization. you can use the level of theory available in Orca.
                                If you want to use XTB methods, you need to put the xtb binary into the Orca directory.
                                Defaults to ``"XTB2"``.
        nprocs (int, optional): The number of processors to use. Defaults to ``1``.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    path_prefix = "orca_irc"

    def __init__(self, **kwargs):
        super(IRCVerifier).__init__(**kwargs)

    def run_irc(
        self,
        ts_mol: "RDKitMol",
        conf_id: int,
        multiplicity: int = 1,
        **kwargs,
    ) -> list:
        """
        Verifying a single TS guess or optimized TS geometry.

        Args:
            ts_mol ('RDKitMol'): The TS in RDKitMol object with 3D geometries embedded.
            conf_id (int): The conformer ID.
            multiplicity (int, optional): The spin multiplicity of the TS. Defaults to ``1``.

        Returns:
            list: the adjacency matrix of the forward and reverse end.
        """

        # Create folder to save IRC input and output files
        work_dir = self.work_dir / f"{self.path_prefix}{conf_id}"
        work_dir.mkdir(parents=True, exist_ok=True)

        input_file = work_dir / f"{self.path_prefix}.inp"
        output_file = work_dir / f"{self.path_prefix}.log"

        # Create and save the Orca input file
        input_content = write_orca_irc(
            ts_mol,
            conf_id=conf_id,
            method=self.method,
            mult=multiplicity,
            nprocs=self.nprocs,
        )
        with open(input_file, "w") as f:
            f.writelines(input_content)

        # Run the Orca IRC using subprocess
        with open(output_file, "w") as f:
            subprocess_run = subprocess.run(
                [self.binary_path, input_file],
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
            )
        if subprocess_run.returncode != 0:
            print(f"Run into error when running Orca IRC.")
            return []

        adj_mats = []
        for direction in ["F", "B"]:
            xyz_file = work_dir / f"orca_irc_IRC_{direction}.xyz"
            try:
                irc_mol = RDKitMol.FromFile(xyz_file, sanitize=False)
                adj_mats.append(irc_mol.GetAdjacencyMatrix())
            except BaseException as e:
                raise RuntimeError(
                    f"Run into error when obtaining adjacency matrix from IRC output file ({xyz_file}). Got: {e}"
                )

        return adj_mats
