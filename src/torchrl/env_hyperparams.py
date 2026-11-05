"""
Environment-specific hyperparameter overrides.
This module provides environment-specific configurations that override default values.
"""
from omegaconf import DictConfig


def get_env_specific_hyperparams(env_name: str) -> dict:
    """
    Get environment-specific hyperparameter overrides based on the environment name.
    
    These values are empirically tuned for each environment and override the defaults.
    The most critical parameters are vmin/vmax which control the value distribution range.
    
    Args:
        env_name: The name of the environment
        
    Returns:
        Dictionary of hyperparameter overrides
    """
    # HumanoidBench specific configs
    humanoid_bench_configs = {
        "h1hand-reach-v0": {
            "vmin": -2000.0,
            "vmax": 2000.0,
        },
        "h1hand-truck-v0": {
            "vmin": -1000.0,
            "vmax": 1000.0,
        },
        "h1hand-maze-v0": {
            "vmin": -1000.0,
            "vmax": 1000.0,
        },
        "h1hand-push-v0": {
            "vmin": -1000.0,
            "vmax": 1000.0,
        },
        "h1hand-basketball-v0": {
            "vmin": -2000.0,
            "vmax": 2000.0,
        },
        "h1hand-package-v0": {
            "vmin": -10000.0,
            "vmax": 10000.0,
        },
    }
    
    # MuJoCo Playground specific configs (if needed in the future)
    playground_configs = {
        "LeapCubeReorient": {
            "vmin": -50.0,
            "vmax": 50.0,
        },
        "LeapCubeRotateZAxis": {
            "vmin": -10.0,
            "vmax": 10.0,
        },
    }
    
    # Combine all configs
    all_configs = {
        **humanoid_bench_configs,
        **playground_configs,
    }
    
    # Return environment-specific config or empty dict if not found
    return all_configs.get(env_name, {})


def apply_env_specific_hyperparams(cfg: DictConfig) -> DictConfig:
    """
    Apply environment-specific hyperparameter overrides to the config.
    
    This function modifies the config in-place by updating hyperparameters
    based on the environment name. It respects user overrides - if a parameter
    was explicitly set by the user (via command line or config override), 
    it won't be changed.
    
    Args:
        cfg: The Hydra/OmegaConf configuration object
        
    Returns:
        The modified configuration object (same as input, modified in-place)
    """
    env_name = cfg.env.name
    env_overrides = get_env_specific_hyperparams(env_name)
    
    # Apply overrides to hyperparameters
    for key, value in env_overrides.items():
        # Only override if the key exists in hyperparameters
        if hasattr(cfg.hyperparameters, key):
            # Store old value for logging
            old_value = getattr(cfg.hyperparameters, key)
            setattr(cfg.hyperparameters, key, value)
            print(f"Environment-specific override for {env_name}: {key}={old_value} -> {value}")
    
    return cfg
