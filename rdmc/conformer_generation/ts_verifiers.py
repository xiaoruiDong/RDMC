#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Modules for verifying optimized ts
"""

# Import RDMC first to avoid unexpected errors
from rdmc import RDKitMol

import os
import pickle
from glob import glob
import subprocess
from time import time
from typing import Optional

from rdmc.external.xtb_tools.opt import run_xtb_calc
from rdmc.external.orca import write_orca_irc
from rdmc.external.gaussian import GaussianLog, write_gaussian_irc
from rdmc.conformer_generation.utils import convert_log_to_mol

# Check TS-Screener
try:
    from ts_ml.trainers.ts_screener_trainer import LitScreenerModule
    from ts_ml.dataloaders.ts_screener_loader import mol2data
    from torch_geometric.data import Batch

except ImportError:
    print("No TS-ML installation detected. Skipping import...")


class TSVerifier:
    """
    The abstract class for TS verifiers.
    """
    def __init__(self,
                 track_stats: bool = False):
        """
        Initialize the TS verifier.

        Args:
            track_stats (bool, optional): Whether to track status. Defaults to False.
        """
        self.track_stats = track_stats
        self.n_failures = None
        self.percent_failures = None
        self.n_opt_cycles = None
        self.stats = []

    def verify_ts_guesses(self,
                          ts_mol: 'RDKitMol',
                          multiplicity: int = 1,
                          save_dir: Optional[str] = None,
                          **kwargs):
        """
        The abstract method for verifying TS guesses (or optimized TS geometries). The method need to take
        `ts_mol` in RDKitMol, `keep_ids` in list, `multiplicity` in int, and `save_dir` in str, and returns
        a list indicating the ones passing the check.

        Args:
            ts_mol ('RDKitMol'): The TS in RDKitMol object with 3D geometries embedded.
            multiplicity (int, optional): The spin multiplicity of the TS. Defaults to 1.
            save_dir (_type_, optional): The directory path to save the results. Defaults to None.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def __call__(self,
                 ts_mol: 'RDKitMol',
                 multiplicity: int = 1,
                 save_dir: Optional[str] = None,
                 **kwargs):
        """
        Run the workflow for verifying the TS guessers (or optimized TS conformers).

        Args:
            ts_mol ('RDKitMol'): The TS in RDKitMol object with 3D geometries embedded.
            multiplicity (int, optional): The spin multiplicity of the TS. Defaults to 1.
            save_dir (_type_, optional): The directory path to save the results. Defaults to None.

        Returns:
            list: a list of true and false
        """
        time_start = time()
        self.verify_ts_guesses(
            ts_mol=ts_mol,
            multiplicity=multiplicity,
            save_dir=save_dir,
            **kwargs
        )

        if self.track_stats:
            time_end = time()
            stats = {"time": time_end - time_start}
            self.stats.append(stats)

        return


class XTBFrequencyVerifier(TSVerifier):
    """
    The class for verifying the TS by calculating and checking its frequencies using XTB.
    """
    def __init__(self,
                 cutoff_frequency: int = -100,
                 track_stats: bool = False):
        """
        Initiate the XTB frequency verifier.

        Args:
            cutoff_frequency (int, optional): Cutoff frequency above which a frequency does not correspond to a TS
                imaginary frequency to avoid small magnitude frequencies which correspond to internal bond rotations
                (defaults to -100 cm-1)
            track_stats (bool, optional): Whether to track stats. Defaults to False.
        """
        super(XTBFrequencyVerifier, self).__init__(track_stats)

        self.cutoff_frequency = cutoff_frequency

    def verify_ts_guesses(self,
                          ts_mol: 'RDKitMol',
                          multiplicity: int = 1,
                          save_dir: Optional[str] = None,
                          **kwargs):
        """
        Verifying TS guesses (or optimized TS geometries).

        Args:
            ts_mol ('RDKitMol'): The TS in RDKitMol object with 3D geometries embedded.
            multiplicity (int, optional): The spin multiplicity of the TS. Defaults to 1.
            save_dir (_type_, optional): The directory path to save the results. Defaults to None.

        Returns:
            list
        """
        for i in range(ts_mol.GetNumConformers()):
            if ts_mol.KeepIDs[i]:
                if ts_mol.frequency[i] is None:
                    props = run_xtb_calc(ts_mol, confId=i, job="--hess", uhf=multiplicity - 1)
                    frequencies = props["frequencies"]
                else:
                    frequencies = ts_mol.frequency[i]

                # Check if the number of large negative frequencies is equal to 1
                freq_check = sum(frequencies < self.cutoff_frequency) == 1
                ts_mol.KeepIDs[i] = freq_check

        if save_dir:
            with open(os.path.join(save_dir, "freq_check_ids.pkl"), "wb") as f:
                pickle.dump(ts_mol.KeepIDs, f)

        return


class OrcaIRCVerifier(TSVerifier):
    """
    The class for verifying the TS by calculating and checking its IRC analysis using Orca.
    """

    def __init__(self,
                 method: str = "XTB2",
                 nprocs: int = 1,
                 track_stats: bool = False):
        """
        Initiate the Orca IRC verifier.

        Args:
            method (str, optional): The method to be used for TS optimization. you can use the level of theory available in Orca.
                                    If you want to use XTB methods, you need to put the xtb binary into the Orca directory. Defaults to XTB2.
            nprocs (int, optional): The number of processors to use. Defaults to 1.
            track_stats (bool, optional): Whether to track the status. Defaults to False.
        """
        super(OrcaIRCVerifier, self).__init__(track_stats)

        self.method = method
        self.nprocs = nprocs

        ORCA_BINARY = os.environ.get("ORCA")
        if not ORCA_BINARY:
            raise RuntimeError('No Orca binary is found in the PATH.')
        else:
            self.orca_binary = ORCA_BINARY

    def verify_ts_guesses(self,
                          ts_mol: 'RDKitMol',
                          multiplicity: int = 1,
                          save_dir: Optional[str] = None,
                          **kwargs):
        """
        Verifying TS guesses (or optimized TS geometries).

        Args:
            ts_mol ('RDKitMol'): The TS in RDKitMol object with 3D geometries embedded.
            multiplicity (int, optional): The spin multiplicity of the TS. Defaults to 1.
            save_dir (_type_, optional): The directory path to save the results. Defaults to None.
        """
        for i in range(ts_mol.GetNumConformers()):
            if ts_mol.KeepIDs[i]:

                # Create and save the Orca input file
                orca_str = write_orca_irc(ts_mol,
                                          confId=i,
                                          method=self.method,
                                          mult=multiplicity,
                                          nprocs=self.nprocs)
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
                    irc_f_mol = RDKitMol.FromFile(os.path.join(orca_dir, "orca_irc_IRC_F.xyz"), sanitize=False)
                    irc_b_mol = RDKitMol.FromFile(os.path.join(orca_dir, "orca_irc_IRC_B.xyz"), sanitize=False)
                except FileNotFoundError:
                    ts_mol.KeepIDs[i] = False
                    continue

                # Generate the adjacency matrix from the mols in the IRC analysis
                f_adj = irc_f_mol.GetAdjacencyMatrix()
                b_adj = irc_b_mol.GetAdjacencyMatrix()

                # Comparing the adjacency matrix
                try:
                    rf_pb_check = ((r_adj == f_adj).all() and (p_adj == b_adj).all())
                    rb_pf_check = ((r_adj == b_adj).all() and (p_adj == f_adj).all())
                except AttributeError:
                    print("Error! Likely that the reaction smiles doesn't correspond to this reaction.")

                irc_check = rf_pb_check or rb_pf_check
                ts_mol.KeepIDs[i] = irc_check

            else:
                ts_mol.KeepIDs[i] = False

        if save_dir:
            with open(os.path.join(save_dir, "irc_check_ids.pkl"), "wb") as f:
                pickle.dump(ts_mol.KeepIDs, f)

        return


class GaussianIRCVerifier(TSVerifier):
    """
    The class for verifying the TS by calculating and checking its IRC analysis using Gaussian.
    """

    def __init__(self,
                 method: str = "GFN2-xTB",
                 nprocs: int = 1,
                 fc_kw: str = "calcall",
                 track_stats: bool = False):
        """
        Initiate the Gaussian IRC verifier.

        Args:
            method (str, optional): The method to be used for TS optimization. you can use the level of theory available in Gaussian.
                                    We provided a script to run XTB using Gaussian, but there are some extra steps to do. Defaults to GFN2-xTB.
            nprocs (int, optional): The number of processors to use. Defaults to 1.
            fc_kw (str, optional): Keyword specifying how often to compute force constants Defaults to "calcall".
            track_stats (bool, optional): Whether to track the status. Defaults to False.
        """
        super(GaussianIRCVerifier, self).__init__(track_stats)

        self.method = method
        self.nprocs = nprocs
        self.fc_kw = fc_kw

        for version in ['g16', 'g09', 'g03']:
            GAUSSIAN_ROOT = os.environ.get(f"{version}root")
            if GAUSSIAN_ROOT:
                break
        else:
            raise RuntimeError('No Gaussian installation found.')

        self.gaussian_binary = os.path.join(GAUSSIAN_ROOT, version, version)

    def verify_ts_guesses(self,
                          ts_mol: 'RDKitMol',
                          multiplicity: int = 1,
                          save_dir: Optional[str] = None,
                          **kwargs):
        """
        Verifying TS guesses (or optimized TS geometries).

        Args:
            ts_mol ('RDKitMol'): The TS in RDKitMol object with 3D geometries embedded.
            multiplicity (int, optional): The spin multiplicity of the TS. Defaults to 1.
            save_dir (_type_, optional): The directory path to save the results. Defaults to None.
        """
        for i in range(ts_mol.GetNumConformers()):
            if ts_mol.KeepIDs[i]:

                # Create folder to save Gaussian IRC input and output files
                gaussian_dir = os.path.join(save_dir, f"gaussian_irc{i}")
                os.makedirs(gaussian_dir, exist_ok=True)

                irc_check = True
                adj_mat = []
                # Conduct forward and reverse IRCs
                for direction in ['forward', 'reverse']:

                    gaussian_input_file = os.path.join(gaussian_dir, f"gaussian_irc_{direction}.gjf")
                    gaussian_output_file = os.path.join(gaussian_dir, f"gaussian_irc_{direction}.log")

                    # Generate and save input file
                    gaussian_str = write_gaussian_irc(
                        ts_mol,
                        confId=i,
                        method=self.method,
                        direction=direction,
                        mult=multiplicity,
                        nprocs=self.nprocs,
                        fc_kw=self.fc_kw,
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
                    if gaussian_run.returncode == 0:
                        try:
                            glog = GaussianLog(gaussian_output_file)
                            adj_mat.append(glog.get_mol(refid=glog.num_all_geoms-1,  # The last geometry in the job
                                                        converged=False,
                                                        sanitize=False,
                                                        backend='openbabel').GetAdjacencyMatrix())
                        except Exception as e:
                            print(f'Run into error when obtaining adjacency matrix from IRC output file. Got: {e}')
                            ts_mol.KeepIDs[i] = False
                            irc_check = False
                    else:
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
                try:
                    rf_pb_check = ((r_adj == f_adj).all() and (p_adj == b_adj).all())
                    rb_pf_check = ((r_adj == b_adj).all() and (p_adj == f_adj).all())
                except AttributeError:
                    print("Error! Likely that the reaction smiles doesn't correspond to this reaction.")

                check = rf_pb_check or rb_pf_check
                ts_mol.KeepIDs[i] = check

            else:
                ts_mol.KeepIDs[i] = False

        if save_dir:
            with open(os.path.join(save_dir, "irc_check_ids.pkl"), "wb") as f:
                pickle.dump(ts_mol.KeepIDs, f)

        return


class TSScreener(TSVerifier):
    """
    The class for screening TS guesses using graph neural networks.
    """

    def __init__(self,
                 trained_model_dir: str,
                 threshold: Optional[int],
                 track_stats: Optional[bool] = False):
        """
        Initialize the TS-Screener model.

        Args:
            trained_model_dir (str): The path to the directory storing the trained TS-Screener model.
            threshold (int): Threshold prediction at which we classify a failure/success.
            track_stats (bool, optional): Whether to track timing stats. Defaults to False.
        """
        super(TSScreener, self).__init__(track_stats)

        # Load the TS-Screener model
        self.module = LitScreenerModule.load_from_checkpoint(
            checkpoint_path=os.path.join(trained_model_dir, "best_model.ckpt")
        )

        # Setup configuration
        self.config = self.module.config
        self.module.model.eval()
        self.threshold = threshold

    def verify_ts_guesses(self,
                          ts_mol: 'RDKitMol',
                          multiplicity: int = 1,
                          save_dir: Optional[str] = None,
                          **kwargs):
        """
        Screen poor TS guesses by using reacting mode from frequency calculation.

        Args:
            ts_mol ('RDKitMol'): The TS in RDKitMol object with 3D geometries embedded.
            multiplicity (int, optional): The spin multiplicity of the TS. Defaults to 1.
            save_dir (_type_, optional): The directory path to save the results. Defaults to None.

        Returns:
            None
        """
        rxn_smiles = kwargs["rxn_smiles"]
        mol_data, ids = [], []

        # parse all optimization folders (which hold the frequency jobs)
        for log_dir in sorted([d for d in glob(os.path.join(save_dir, "*opt*")) if os.path.isdir(d)], \
                              key=lambda x: int(x.split("opt")[-1])):

            idx = int(log_dir.split("opt")[-1])
            if ts_mol.KeepIDs[idx]:
                freq_log_path = glob(os.path.join(log_dir, "*opt.log"))[0]
                ts_freq_mol = convert_log_to_mol(freq_log_path)

                if ts_freq_mol is None:
                    ts_mol.KeepIDs.update({idx: False})
                    continue

                ts_freq_mol.SetProp("Name", rxn_smiles)
                data = mol2data(ts_freq_mol, self.module.config, eval_mode=True)

                mol_data.append(data)
                ids.append(idx)

        # create data batch and run screener model
        batch_data = Batch.from_data_list(mol_data)
        preds = self.module.model(batch_data) > self.threshold

        # update which TSs to keep
        updated_keep_ids = {idx: pred.item() for idx, pred in zip(ids, preds)}
        ts_mol.KeepIDs.update(updated_keep_ids)

        # write ids to file
        with open(os.path.join(save_dir, "screener_check_ids.pkl"), "wb") as f:
            pickle.dump(ts_mol.KeepIDs, f)
