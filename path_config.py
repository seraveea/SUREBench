import os


def get_data_root(args=None):
    """Return dataset root directory from args, env, or default."""
    if args is not None:
        data_root = getattr(args, "data_root", "")
        if data_root:
            return data_root
    return os.environ.get("SUREBENCH_DATA_ROOT", "./data")


def data_path(*parts, args=None):
    """Build a path under dataset root."""
    return os.path.join(get_data_root(args), *parts)


def dataset_cache_dir(name, args=None):
    """Cache dir for Hugging Face datasets under dataset root."""
    return data_path(name, args=args)


# ---------------------------------------------------------------------------
# Model path helpers
# ---------------------------------------------------------------------------

def get_model_root(args=None):
    """Return model root directory from args, env, or default."""
    if args is not None:
        model_root = getattr(args, "model_root", "")
        if model_root:
            return model_root
    return os.environ.get("SUREBENCH_MODEL_ROOT", "./models")


def model_path(*parts, args=None):
    """Build a path under model root."""
    return os.path.join(get_model_root(args), *parts)
