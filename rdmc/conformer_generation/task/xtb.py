from rdmc.conformer_generation.comp_env import xtb_available

from rdmc.conformer_generation.task.basetask import BaseTask


class XTBTask(BaseTask):

    def __init__(self, track_stats: bool = False):
        super().__init__(track_stats)

    def is_available(self):
        return xtb_available
