import os
from typing import Optional
import subprocess
import pickle

from rdmc import RDKitMol

from rdmc.conformer_generation.ts_verifiers.base import TSVerifier
from rdmc.external.inpwriter import write_orca_irc


class OrcaIRCVerifier(TSVerifier):
    """
    The class for verifying the TS by calculating and checking its IRC analysis using Orca.

    Args:
        method (str, optional): The method to be used for TS optimization. you can use the level of theory available in Orca.
                                If you want to use XTB methods, you need to put the xtb binary into the Orca directory.
                                Defaults to ``"XTB2"``.
        nprocs (int, optional): The number of processors to use. Defaults to ``1``.
        track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
    """

    def __init__(
        self, method: str = "XTB2", nprocs: int = 1, track_stats: bool = False
    ):
        """
        Initiate the Orca IRC verifier.

        Args:
            method (str, optional): The method to be used for TS optimization. you can use the level of theory available in Orca.
                                    If you want to use XTB methods, you need to put the xtb binary into the Orca directory.
                                    Defaults to ``"XTB2"``.
            nprocs (int, optional): The number of processors to use. Defaults to ``1``.
            track_stats (bool, optional): Whether to track the status. Defaults to ``False``.
        """
        super(OrcaIRCVerifier, self).__init__(track_stats)

        self.method = method
        self.nprocs = nprocs

        ORCA_BINARY = os.environ.get("ORCA")
        if not ORCA_BINARY:
            raise RuntimeError("No Orca binary is found in the PATH.")
        else:
            self.orca_binary = ORCA_BINARY

    def verify_ts_guesses(
        self,
        ts_mol: "RDKitMol",
        multiplicity: int = 1,
        save_dir: Optional[str] = None,
        **kwargs,
    ) -> "RDKitMol":
        """
        Verifying TS guesses (or optimized TS geometries).

        Args:
            ts_mol (RDKitMol): The TS in RDKitMol object with 3D geometries embedded.
            multiplicity (int, optional): The spin multiplicity of the TS. Defaults to ``1``.
            save_dir (str, optional): The directory path to save the results. Defaults to ``None``.

        Returns:
            RDKitMol: The molecule in RDKitMol object with verification results stored in ``KeepIDs``.
        """
        for i in range(ts_mol.GetNumConformers()):
            if ts_mol.KeepIDs[i]:

                # Create and save the Orca input file
                orca_str = write_orca_irc(
                    ts_mol,
                    conf_id=i,
                    method=self.method,
                    mult=multiplicity,
                    nprocs=self.nprocs,
                )
                orca_dir = os.path.join(save_dir, f"orca_irc{i}")
                os.makedirs(orca_dir)

                orca_input_file = os.path.join(orca_dir, "orca_irc.inp")
                with open(orca_input_file, "w") as f:
                    f.writelines(orca_str)

                # Run the Orca IRC using subprocess
                with open(os.path.join(orca_dir, "orca_irc.log"), "w") as f:
                    orca_run = subprocess.run(
                        [self.orca_binary, orca_input_file],
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        cwd=os.getcwd(),
                    )
                if orca_run.returncode != 0:
                    ts_mol.KeepIDs[i] = False
                    continue

                # Generate the adjacency matrix from the SMILES
                r_smi, p_smi = kwargs["rxn_smiles"].split(">>")
                r_adj = RDKitMol.FromSmiles(r_smi).GetAdjacencyMatrix()
                p_adj = RDKitMol.FromSmiles(p_smi).GetAdjacencyMatrix()

                # Read the terminal geometries from the IRC analysis into RDKitMol
                try:
                    irc_f_mol = RDKitMol.FromFile(
                        os.path.join(orca_dir, "orca_irc_IRC_F.xyz"), sanitize=False
                    )
                    irc_b_mol = RDKitMol.FromFile(
                        os.path.join(orca_dir, "orca_irc_IRC_B.xyz"), sanitize=False
                    )
                except FileNotFoundError:
                    ts_mol.KeepIDs[i] = False
                    continue

                # Generate the adjacency matrix from the mols in the IRC analysis
                f_adj = irc_f_mol.GetAdjacencyMatrix()
                b_adj = irc_b_mol.GetAdjacencyMatrix()

                # Comparing the adjacency matrix
                rf_pb_check, rb_pf_check = False, False
                try:
                    rf_pb_check = (r_adj == f_adj).all() and (p_adj == b_adj).all()
                    rb_pf_check = (r_adj == b_adj).all() and (p_adj == f_adj).all()
                except AttributeError:
                    print(
                        "Error! Likely that the reaction smiles doesn't correspond to this reaction."
                    )

                irc_check = rf_pb_check or rb_pf_check
                ts_mol.KeepIDs[i] = irc_check

            else:
                ts_mol.KeepIDs[i] = False

        if save_dir:
            with open(os.path.join(save_dir, "irc_check_ids.pkl"), "wb") as f:
                pickle.dump(ts_mol.KeepIDs, f)

        return ts_mol
