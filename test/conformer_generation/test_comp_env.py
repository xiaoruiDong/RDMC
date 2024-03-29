import pytest

from rdmc.conformer_generation.comp_env.software import try_import


@pytest.mark.parametrize(
    "full_module_name",
    [
        "rdkit",
        "rdkit.Chem",
    ],
)
def test_try_import(full_module_name):

    # Test rdkit since it should be available.

    try_import(full_module_name, namespace=globals())

    assert full_module_name.split(".")[-1] in globals()
    # If this is an actual module, no erorr will be raised
    globals().pop(full_module_name.split(".")[-1]).__name__


def test_try_import_fake():

    full_module_name = "fake_module.fake_submodule.fake_func"
    try_import(full_module_name, namespace=globals())

    assert full_module_name.split(".")[-1] in globals()
    with pytest.raises(RuntimeError):
        globals().pop(full_module_name.split(".")[-1]).__name__
