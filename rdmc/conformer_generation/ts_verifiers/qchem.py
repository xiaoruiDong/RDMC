from rdmc import RDKitMol
from rdmc.conformer_generation.task.qchem import QChemTask
from rdmc.conformer_generation.ts_verifiers.base import IRCVerifier
from rdmc.external.inpwriter import write_qchem_irc


class QChemIRCVerifier(QChemTask, IRCVerifier):
    """
    The class for verifying the TS by calculating and checking its IRC analysis using QChem.

    Args:
        method (str, optional): The method to be used for TS optimization. you can use the method available in QChem.
                                Defaults to ``"wB97x-d3"``.
        basis (str, optional): The method to be used for TS optimization. you can use the basis available in QChem.
                                Defaults to ``"def2-tzvp"``.
        nprocs (int, optional): The number of processors to use. Defaults to ``1``.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    path_prefix = "qchem_irc"

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

        input_file = work_dir / f"{self.path_prefix}.qcin"
        output_file = work_dir / f"{self.path_prefix}.log"

        # Generate and save input file
        input_content = write_qchem_irc(
            ts_mol,
            conf_id=conf_id,
            method=self.method,
            basis=self.basis,
            mult=multiplicity,
        )

        with open(input_file, "w") as f:
            f.writelines(input_content)

        # Run the IRC using subprocess
        subprocess_run = self.subprocess_runner(
            input_file,
            output_file,
            work_dir,
        )

        adj_mats = []
        try:
            log = self.logparser(output_file)
            for cid in [log.get_irc_midpoint() - 1, -2]:
                adj_mats.append(
                    log.get_mol(
                        refid=cid,
                        sanitize=False,
                        backend="openbabel",
                    ).GetAdjacencyMatrix()
                )
        except BaseException as e:
            raise RuntimeError(
                f"Run into error when obtaining adjacency matrix from IRC output file ({output_file}). Got: {e}"
            )

        return adj_mats
