from rdmc.conformer_generation.comp_env.software import register_binary

for bin_name in ['g16', 'g09', 'g03']:
    register_binary(bin_name)
