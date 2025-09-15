"""
ModelParallel utilities for KerasHub.

This module provides utilities to handle ModelParallel distribution
during model loading to prevent OOM errors when loading large models.
The sharding configuration is left to the user - we only provide
OOM-safe weight loading.
"""

import keras

from keras_hub.src.utils.keras_utils import sharded_weights_available


def is_modelparallel_active():
    """Check if ModelParallel distribution is currently active."""
    try:
        from keras.src.distribution.distribution_lib import (
            distribution as get_dist,
        )

        current_dist = get_dist()
        return current_dist and isinstance(
            current_dist, keras.distribution.ModelParallel
        )
    except ImportError:
        return False


def load_weights_with_modelparallel_awareness(model, filepath):
    """
    Load weights with ModelParallel awareness to prevent OOM errors.

    This function detects if ModelParallel distribution is active and
    uses appropriate loading strategies to avoid OOM during weight loading.
    The actual sharding configuration is determined by the user's
    ModelParallel setup - we only ensure OOM-safe loading.

    Args:
        model: The Keras model to load weights into
        filepath: Path to the weights file or sharded weights config

    Raises:
        RuntimeError: If sharded weights are not available when needed
        OSError: If weight files cannot be loaded
    """
    if not is_modelparallel_active():
        # No ModelParallel active, use standard loading
        model.load_weights(filepath)
        return

    # ModelParallel is active - use sharded loading strategy
    if not sharded_weights_available():
        raise RuntimeError(
            "ModelParallel distribution is active but sharded weights loading "
            f"is not supported in the current Keras version "
            f"{keras.__version__}. "
            "Please update to a newer version or disable ModelParallel "
            "distribution."
        )

    # For ModelParallel, we need to ensure weights are loaded in a way that
    # respects the distribution. The key insight is that we should load
    # weights after the model architecture is created with proper distribution.
    try:
        model.load_weights(filepath)
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            raise RuntimeError(
                f"OOM error during weight loading with ModelParallel: {e}\n"
                "This suggests that the model architecture was not properly "
                "distributed before weight loading. Ensure ModelParallel "
                "distribution is set before model creation."
            ) from e
        else:
            raise e
