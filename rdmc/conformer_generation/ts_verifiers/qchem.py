import os
from typing import Optional
import subprocess
import pickle

from rdmc import RDKitMol

from rdmc.conformer_generation.ts_verifiers.base import TSVerifier
from rdmc.conformer_generation.task.qchem import QChemTask
from rdmc.external.inpwriter import write_qchem_irc
from rdmc.external.logparser import QChemLog


class QChemIRCVerifier(QChemTask, TSVerifier):
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

    def run(
        self,
        ts_mol: "RDKitMol",
        multiplicity: int = 1,
        **kwargs,
    ):
        """
        Verifying TS guesses (or optimized TS geometries).

        Args:
            ts_mol ('RDKitMol'): The TS in RDKitMol object with 3D geometries embedded.
            multiplicity (int, optional): The spin multiplicity of the TS. Defaults to ``1``.
            save_dir (_type_, optional): The directory path to save the results. Defaults to ``None``.

        Returns:
            RDKitMol: The molecule in RDKitMol object with verification results stored in ``KeepIDs``.
        """
        for i in range(ts_mol.GetNumConformers()):
            if ts_mol.KeepIDs[i]:

                # Create folder to save QChem IRC input and output files
                qchem_dir = os.path.join(self.save_dir, f"qchem_irc{i}")
                os.makedirs(qchem_dir, exist_ok=True)

                irc_check = True
                adj_mat = []
                # Conduct IRC
                qchem_input_file = os.path.join(qchem_dir, f"qchem_irc.qcin")
                qchem_output_file = os.path.join(qchem_dir, f"qchem_irc.log")

                # Generate and save input file
                qchem_str = write_qchem_irc(
                    ts_mol,
                    conf_id=i,
                    method=self.method,
                    basis=self.basis,
                    mult=multiplicity,
                )
                with open(qchem_input_file, "w") as f:
                    f.writelines(qchem_str)

                # Run IRC using subprocess
                with open(qchem_output_file, "w") as f:
                    qchem_run = subprocess.run(
                        [self.binary_path, "-nt", str(self.nprocs), qchem_input_file],
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        cwd=os.getcwd(),
                    )

                # Extract molecule adjacency matrix from IRC results
                # TBD: We can stop running IRC if one side of IRC fails
                # I personally think it is worth to continue to run the other IRC just to provide more sights
                try:
                    log = QChemLog(qchem_output_file)
                    adj_mat.append(
                        log.get_mol(
                            refid=log.get_irc_midpoint() - 1,
                            sanitize=False,
                            backend="openbabel",
                        ).GetAdjacencyMatrix()
                    )
                    adj_mat.append(
                        log.get_mol(
                            refid=-2,  # The second to last geometry in the job
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

        if self.save_dir:
            with open(os.path.join(self.save_dir, "irc_check_ids.pkl"), "wb") as f:
                pickle.dump(ts_mol.KeepIDs, f)

        return ts_mol
