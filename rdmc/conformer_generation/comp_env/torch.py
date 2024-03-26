from rdmc.conformer_generation.comp_env.software import try_import

try_import(
    "torch",
    namespace=globals(),
    package_name="PyTorch",
)
