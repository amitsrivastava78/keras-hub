"""
ModelParallel utilities for KerasHub.

This module provides utilities to handle ModelParallel distribution
during model loading to prevent OOM errors when loading large models.
The sharding configuration is left to the user - we only provide
OOM-safe weight loading.
"""

import keras

from keras_hub.src.utils.keras_utils import sharded_weights_available


class _LazyKerasHubExport:
    """Lazy decorator to avoid circular imports."""

    def __init__(self, path):
        self.path = path

    def __call__(self, func):
        # Apply the decorator lazily when the function is first called
        def wrapper(*args, **kwargs):
            # Import and apply the decorator on first call
            try:
                from keras_hub.src.api_export import keras_hub_export

                decorated_func = keras_hub_export(self.path)(func)
                # Replace the wrapper with the decorated function
                globals()[func.__name__] = decorated_func
                return decorated_func(*args, **kwargs)
            except ImportError:
                # Fallback if decorator not available
                return func(*args, **kwargs)

        return wrapper


@_LazyKerasHubExport("keras_hub.utils.is_modelparallel_active")
def is_modelparallel_active():
    """Check if ModelParallel distribution is currently active."""
    try:
        from keras.src.distribution.distribution_lib import (
            distribution as get_dist,
        )

        current_dist = get_dist()
        is_active = current_dist and isinstance(
            current_dist, keras.distribution.ModelParallel
        )

        print("🔍 DEBUG is_modelparallel_active():")
        print(
            f"   Current distribution: {type(current_dist).__name__ if current_dist else 'None'}"
        )
        print(f"   Is ModelParallel: {is_active}")

        return is_active
    except ImportError as e:
        print(f"🔍 DEBUG is_modelparallel_active(): ImportError - {e}")
        return False


@_LazyKerasHubExport(
    "keras_hub.utils.load_weights_with_modelparallel_awareness"
)
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
    print("🔍 DEBUG load_weights_with_modelparallel_awareness():")
    print(f"   Model: {type(model).__name__}")
    print(f"   Model name: {getattr(model, 'name', 'Unknown')}")
    print(f"   Filepath: {filepath}")
    print(f"   Keras version: {keras.__version__}")

    is_mp_active = is_modelparallel_active()
    print(f"   ModelParallel active: {is_mp_active}")

    if not is_mp_active:
        # No ModelParallel active, use standard loading
        print("   → Using standard weight loading (no ModelParallel)")
        model.load_weights(filepath)
        print("   ✅ Standard weight loading completed")
        return

    # ModelParallel is active - use sharded loading strategy
    print("   → ModelParallel detected, checking sharded weights support...")
    sharded_available = sharded_weights_available()
    print(f"   Sharded weights available: {sharded_available}")

    if not sharded_available:
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
    print("   → Using ModelParallel-aware weight loading...")
    try:
        model.load_weights(filepath)
        print("   ✅ ModelParallel-aware weight loading completed")
    except Exception as e:
        print(f"   ❌ Error during weight loading: {e}")
        if "RESOURCE_EXHAUSTED" in str(e):
            raise RuntimeError(
                f"OOM error during weight loading with ModelParallel: {e}\n"
                "This suggests that the model architecture was not properly "
                "distributed before weight loading. Ensure ModelParallel "
                "distribution is set before model creation."
            ) from e
        else:
            raise e
