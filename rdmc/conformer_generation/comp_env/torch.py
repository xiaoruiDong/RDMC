from rdmc.conformer_generation.comp_env.software import try_import

try_import(
    "torch",
    namespace=globals(),
    package_name="PyTorch",
)

try_import(
    "torch.nn.functional",
    alias="F",
    namespace=globals(),
    package_name="PyTorch",
)
