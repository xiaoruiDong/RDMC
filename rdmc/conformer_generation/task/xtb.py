from rdmc.conformer_generation.comp_env import xtb_available


class XTBTask:

    def is_available(self):
        return xtb_available
