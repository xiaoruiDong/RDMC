from abc import abstractmethod
from typing import Optional
import pickle

from rdmc.conformer_generation.task.basetask import BaseTask


class FreqVerifier(BaseTask):
    """
    A verifier based on frequency jobs
    """

    allowed_number_negative_frequencies = 0
    default_cutoff_frequency = 0.0

    def __init__(
        self, cutoff_frequency: Optional[float] = None, track_stats: bool = False
    ):
        """
        Initiate the XTB frequency verifier.

        Args:
            cutoff_frequency (float, optional): Cutoff frequency above which a frequency does not correspond to a TS
                imaginary frequency to avoid small magnitude frequencies which correspond to internal bond rotations
                Defaults to ``-10.`` cm-1
            track_stats (bool, optional): Whether to track stats. Defaults to ``False``.
        """
        super().__init__(track_stats)

        self.cutoff_frequency = cutoff_frequency or self.default_cutoff_frequency

    def run(
        self,
        mol,
        **kwargs,
    ):
        """
        Verifying stable species or TS geometries with frequency calculations. It will first check if the frequencies are available
        from the ``mol`` passed in, and make use of them. Otherwise, it will launch new jobs.

        Args:
            mol ('RDKitMol'): The stable species in RDKitMol object with 3D geometries embedded.

        Returns:
            RDKitMol: The molecule in RDKitMol object with verification results stored in ``KeepIDs``.
        """
        if mol.GetNumAtoms() == 1:
            # There is no frequency mode for uni-atom molecules
            # todo: add a log message
            pass
        else:
            for i in range(mol.GetNumConformers()):
                if not mol.KeepIDs[i]:
                    # Check if the conformer is good till this step
                    continue

                if hasattr(mol, "frequency") and mol.frequency[i] is not None:
                    frequencies = mol.frequency[i]
                else:
                    frequencies = self.calc_freq(mol, conf_id=i, **kwargs)

                freq_check = (
                    sum(frequencies < self.cutoff_frequency)
                    == self.allowed_number_negative_frequencies
                )

                # update properties
                mol.frequency[i] = frequencies
                mol.KeepIDs[i] = freq_check

        if self.save_dir:
            with open(self.save_dir / "freq_check_ids.pkl", "wb") as f:
                pickle.dump(mol.KeepIDs, f)

        return mol

    @abstractmethod
    def calc_freq(self, mol, conf_id, **kwargs) -> "np.ndarray":
        """
        Calculate the frequencies for conf_id'th geometry of a molecule. The function is expected
        to return a np.ndarray.

        Args:
            mol ('RDKitMol'): The molecule in RDKitMol object with 3D geometries embedded.
            conf_id (int): The conformer id.

        Returns:
            np.ndarray: an array of frequencies.
        """
        raise NotImplementedError
