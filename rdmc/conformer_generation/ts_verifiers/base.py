from rdmc.conformer_generation.task.basetask import BaseTask
from rdmc.conformer_generation.verifiers.base import FreqVerifier


class TSFreqVerifier(FreqVerifier):

    allowed_number_negative_frequencies = 1
    default_cutoff_frequency = -10.0


class IRCVerifier(BaseTask):

    pass
