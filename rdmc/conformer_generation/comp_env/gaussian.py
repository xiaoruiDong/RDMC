from rdmc.conformer_generation.comp_env.software import has_binary


def get_default_gaussian_binary():
    for bin_name in ["g16", "g09", "g03"]:
        if has_binary(bin_name):
            return bin_name
    return None


gaussian_available = get_default_gaussian_binary() is not None
