from rdmc.conformer_generation.comp_env import xtb_available

from rdmc.conformer_generation.task.basetask import BaseTask


class XTBTask(BaseTask):

    def is_available(self):
        return xtb_available
