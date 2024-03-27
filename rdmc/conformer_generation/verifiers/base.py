from typing import Optional
from abc import abstractmethod


from rdmc.conformer_generation.task.basetask import BaseTask


class Verifier(BaseTask):

    @abstractmethod
    def run(
        self,
        mol: "RDKitMol",
        multiplicity: int = 1,
        **kwargs,
    ):
        """
        The abstract method for verifying guesses (or optimized stable species geometries). The method needs to at least
        take ``mol`` and returns a mol with KeepIDs set/updated.

        Args:
            mol ('RDKitMol'): The stable species in RDKitMol object with 3D geometries embedded.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError
