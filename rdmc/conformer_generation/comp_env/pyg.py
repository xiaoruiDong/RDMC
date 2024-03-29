from rdmc.conformer_generation.comp_env.software import try_import

try_import(
    "torch_geometric.data.Batch",
    namespace=globals(),
    package_name="PyTorch Geometric",
)
