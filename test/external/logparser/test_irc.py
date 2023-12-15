from rdmc.external.logparser import GaussianLog
from rdmc.utils import repo_dir

data_path = repo_dir / 'test' / 'data'

def test_fail_irc_with_correction_steps():
    log_file = data_path / 'gaussian_irc_with_correction_failed.log'
    log = GaussianLog(log_file)
    assert not log.success
    # See if energy values can be correctedly loaded
    assert log.get_scf_energies(converged=True).shape[0] == 7
    assert log.get_scf_energies(converged=False).shape[0] == 35
