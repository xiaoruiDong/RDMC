import os
from typing import Optional
import subprocess
import pickle

from rdmc import RDKitMol

from rdmc.conformer_generation.ts_verifiers.base import TSVerifier
from rdmc.external.inpwriter import write_gaussian_irc
from rdmc.external.logparser import GaussianLog


class GaussianIRCVerifier(TSVerifier):
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

    def __init__(
        self,
        method: str = "GFN2-xTB",
        nprocs: int = 1,
        memory: int = 1,
        fc_kw: str = "calcall",
        track_stats: bool = False,
    ):
        """
        Initiate the Gaussian IRC verifier.

        Args:
            method (str, optional): The method to be used for TS optimization. you can use the level of theory available in Gaussian.
                                    We provided a script to run XTB using Gaussian, but there are some extra steps to do. Defaults to GFN2-xTB.
            nprocs (int, optional): The number of processors to use. Defaults to 1.
            memory (int, optional): Memory in GB used by Gaussian. Defaults to 1.
            fc_kw (str, optional): Keyword specifying how often to compute force constants Defaults to "calcall".
            track_stats (bool, optional): Whether to track the status. Defaults to False.
        """
        super(GaussianIRCVerifier, self).__init__(track_stats)

        self.method = method
        self.nprocs = nprocs
        self.memory = memory
        self.fc_kw = fc_kw

        for version in ["g16", "g09", "g03"]:
            GAUSSIAN_ROOT = os.environ.get(f"{version}root")
            if GAUSSIAN_ROOT:
                break
        else:
            raise RuntimeError("No Gaussian installation found.")

        self.gaussian_binary = os.path.join(GAUSSIAN_ROOT, version, version)

    def verify_ts_guesses(
        self,
        ts_mol: "RDKitMol",
        multiplicity: int = 1,
        save_dir: Optional[str] = None,
        **kwargs,
    ) -> RDKitMol:
        """
        Verifying TS guesses (or optimized TS geometries).

        Args:
            ts_mol ('RDKitMol'): The TS in RDKitMol object with 3D geometries embedded.
            multiplicity (int, optional): The spin multiplicity of the TS. Defaults to ``1``.
            save_dir (str, optional): The directory path to save the results. Defaults to ``None``.

        Returns:
            RDKitMol: The molecule in RDKitMol object with verification results stored in ``KeepIDs``.
        """
        for i in range(ts_mol.GetNumConformers()):
            if ts_mol.KeepIDs[i]:

                # Create folder to save Gaussian IRC input and output files
                gaussian_dir = os.path.join(save_dir, f"gaussian_irc{i}")
                os.makedirs(gaussian_dir, exist_ok=True)

                irc_check = True
                adj_mat = []
                # Conduct forward and reverse IRCs
                for direction in ["forward", "reverse"]:

                    gaussian_input_file = os.path.join(
                        gaussian_dir, f"gaussian_irc_{direction}.gjf"
                    )
                    gaussian_output_file = os.path.join(
                        gaussian_dir, f"gaussian_irc_{direction}.log"
                    )

                    # Generate and save input file
                    gaussian_str = write_gaussian_irc(
                        ts_mol,
                        conf_id=i,
                        method=self.method,
                        direction=direction,
                        mult=multiplicity,
                        nprocs=self.nprocs,
                        memory=self.memory,
                        hess=self.fc_kw,
                    )
                    with open(gaussian_input_file, "w") as f:
                        f.writelines(gaussian_str)

                    # Run IRC using subprocess
                    with open(gaussian_output_file, "w") as f:
                        gaussian_run = subprocess.run(
                            [self.gaussian_binary, gaussian_input_file],
                            stdout=f,
                            stderr=subprocess.STDOUT,
                            cwd=os.getcwd(),
                        )

                    # Extract molecule adjacency matrix from IRC results
                    # TBD: We can stop running IRC if one side of IRC fails
                    # I personally think it is worth to continue to run the other IRC just to provide more sights
                    try:
                        glog = GaussianLog(gaussian_output_file)
                        adj_mat.append(
                            glog.get_mol(
                                refid=glog.num_all_geoms
                                - 1,  # The last geometry in the job
                                converged=False,
                                sanitize=False,
                                backend="openbabel",
                            ).GetAdjacencyMatrix()
                        )
                    except Exception as e:
                        print(
                            f"Run into error when obtaining adjacency matrix from IRC output file. Got: {e}"
                        )
                        ts_mol.KeepIDs[i] = False
                        irc_check = False

                # Bypass the further steps if IRC job fails
                if not irc_check and len(adj_mat) != 2:
                    ts_mol.KeepIDs[i] = False
                    continue

                # Generate the adjacency matrix from the SMILES
                r_smi, p_smi = kwargs["rxn_smiles"].split(">>")
                r_adj = RDKitMol.FromSmiles(r_smi).GetAdjacencyMatrix()
                p_adj = RDKitMol.FromSmiles(p_smi).GetAdjacencyMatrix()
                f_adj, b_adj = adj_mat

                rf_pb_check, rb_pf_check = False, False
                try:
                    rf_pb_check = (r_adj == f_adj).all() and (p_adj == b_adj).all()
                    rb_pf_check = (r_adj == b_adj).all() and (p_adj == f_adj).all()
                except AttributeError:
                    print(
                        "Error! Likely that the reaction smiles doesn't correspond to this reaction."
                    )

                check = rf_pb_check or rb_pf_check
                ts_mol.KeepIDs[i] = check

            else:
                ts_mol.KeepIDs[i] = False

        if save_dir:
            with open(os.path.join(save_dir, "irc_check_ids.pkl"), "wb") as f:
                pickle.dump(ts_mol.KeepIDs, f)

        return ts_mol
