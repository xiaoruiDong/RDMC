from rdmc.conformer_generation.comp_env.software import try_import


package_name = "TS-ML"
namespace = globals()

modules = [
    "ts_ml.dataloaders.ts_egnn_loader.TSDataset",
    "ts_ml.dataloaders.ts_gcn_loader.TSGCNDataset",
    "ts_ml.dataloaders.ts_screener_loader.mol2data",
    "ts_ml.trainers.ts_egnn_trainer.LitTSModule",
    ("ts_ml.trainers.ts_gcn_trainer.LitTSModule", "LitTSGCNModule"),
    "ts_ml.trainers.ts_screener_trainer.LitScreenerModule",
]

for module in modules:
    if isinstance(module, tuple):
        try_import(
            module[0], alias=module[1], namespace=namespace, package_name=package_name
        )
    else:
        try_import(module, namespace=namespace, package_name=package_name)
