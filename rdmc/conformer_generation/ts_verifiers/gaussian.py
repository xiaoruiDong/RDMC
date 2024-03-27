from rdmc import RDKitMol
from rdmc.conformer_generation.ts_verifiers.base import IRCVerifier
from rdmc.conformer_generation.task.gaussian import GaussianTask
from rdmc.external.inpwriter import write_gaussian_irc


class GaussianIRCVerifier(IRCVerifier, GaussianTask):
    """
    The class for verifying the TS by calculating and checking its IRC analysis using Gaussian.

    Args:
        method (str, optional): The method to be used for TS optimization. you can use the level of theory available in Gaussian.
                                We provided a script to run XTB using Gaussian, but there are some extra steps to do.
                                Defaults to ``"GFN2-xTB"``.
        nprocs (int, optional): The number of processors to use. Defaults to ``1``.
        memory (int, optional): Memory in GB used by Gaussian. Defaults to ``1``.
        fc_kw (str, optional): Keyword specifying how often to compute force constants Defaults to ``"calcall"``.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    path_prefix = "gaussian_irc"

    def __init__(
        self,
        method: str = "GFN2-xTB",
        nprocs: int = 1,
        memory: int = 1,
        fc_kw: str = "calcall",
        track_stats: bool = False,
    ):
        super(IRCVerifier).__init__(
            method=method, nprocs=nprocs, memory=memory, track_stats=track_stats
        )
        self.fc_kw = fc_kw

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

        adj_mats = []
        # Conduct forward and reverse IRCs
        for direction in ["forward", "reverse"]:

            input_file = work_dir / f"{self.path_prefix}{direction}.gjf"
            output_file = work_dir / f"{self.path_prefix}{direction}.log"

            # Generate and save input file
            input_content = write_gaussian_irc(
                ts_mol,
                conf_id=conf_id,
                method=self.method,
                direction=direction,
                charge=0,  # temporarily hardcoded
                mult=multiplicity,
                nprocs=self.nprocs,
                memory=self.memory,
                hess=self.fc_kw,
            )
            with open(input_file, "w") as f:
                f.writelines(input_content)

            # Run the IRC using subprocess
            subprocess_run = self.subprocess_runner(
                input_file,
                output_file,
                work_dir,
            )

            # Extract molecule adjacency matrix from IRC results
            # TBD: We can stop running IRC if one side of IRC fails
            # I personally think it is worth to continue to run the other IRC just to provide more sights
            try:
                glog = self.logparser(output_file)
                adj_mats.append(
                    glog.get_mol(
                        refid=glog.num_all_geoms - 1,  # The last geometry in the job
                        converged=False,
                        sanitize=False,
                        backend="openbabel",
                    ).GetAdjacencyMatrix()
                )
            except BaseException as e:
                raise RuntimeError(
                    f"Run into error when obtaining adjacency matrix from IRC output file ({output_file}). Got: {e}"
                )

        return adj_mats
