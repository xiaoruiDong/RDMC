from rdmc.conformer_generation.comp_env.software import try_import


package_name = "ase"
namespace = globals()
modules = [
    "ase.autoneb.AutoNEB",
    "ase.calculators.orca.ORCA",
    "ase.calculators.calculator.Calculator",
    "ase.calculators.calculator.CalculatioFailed",
    "ase.optimize.QuasiNewton",
]

for module in modules:
    try_import(
        module,
        namespace=namespace,
        package_name=package_name,
    )
