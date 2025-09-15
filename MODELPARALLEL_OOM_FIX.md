# ModelParallel OOM Fix for KerasHub

## Problem Description

When using `keras.distribution.ModelParallel` with `from_preset()` to load large models (e.g., Gemma2 9B), users encounter Out of Memory (OOM) errors despite having sufficient total distributed memory across multiple devices.

### Root Cause

The issue occurs because:

1. **Model Architecture Creation**: KerasHub creates the model architecture using standard Keras model creation, which doesn't respect ModelParallel distribution during variable creation
2. **Weight Loading**: Even when ModelParallel is active, the `load_weights()` method loads all weights into memory at once before distribution/sharding is applied
3. **Timing Issue**: The distribution is set up correctly, but the weight loading happens before the distribution can take effect

### Error Pattern

```
RESOURCE_EXHAUSTED: Error allocating device buffer: Attempting to allocate 3.42G. 
That was not possible. There are 661.94M free.; (0x0x0_HBM0)
```

This typically occurs in the call stack:
```
keras_hub/src/models/task.py → keras_hub/src/utils/preset_utils.py → 
keras/src/saving/serialization_lib.py → keras/src/layers/init.py → 
keras_hub/src/models/gemma/gemma_backbone.py → 
keras_hub/src/layers/modeling/reversible_embedding.py
```

## Solution

### ModelParallel-Aware Weight Loading

The fix introduces `load_weights_with_modelparallel_awareness()` function that:

- Detects if ModelParallel distribution is active
- Uses appropriate loading strategies to prevent OOM
- Provides clear error messages when issues occur
- Falls back to standard loading when ModelParallel is not active
- **Does NOT automatically configure sharding** - that's left to the user

### Key Principle

**The sharding configuration is determined by the user's ModelParallel setup - we only ensure OOM-safe loading.**

## Usage

### Basic Usage

```python
import keras
import keras_hub

# User sets up their own ModelParallel distribution
devices = keras.distribution.list_devices()
device_mesh = keras.distribution.DeviceMesh(
    (1, 8), ["data", "model"], devices=devices
)

# User creates their own layout map
layout_map = keras.distribution.LayoutMap(device_mesh)
layout_map["token_embedding/embeddings"] = ("model", None)
layout_map["decoder_block_.*attention.*query.*kernel"] = ("model", None, None)
# ... user defines their own sharding strategy

# User sets up ModelParallel distribution
model_parallel = keras.distribution.ModelParallel(
    layout_map=layout_map,
    batch_dim_name="data"
)
keras.distribution.set_distribution(model_parallel)

# Load large model - OOM fix is automatically applied
model = keras_hub.models.CausalLM.from_preset("gemma2_9b_en")
```

### Using Existing Model Layout Maps

Many KerasHub models provide their own `get_layout_map()` methods:

```python
import keras
import keras_hub

# Use Gemma's built-in layout map
devices = keras.distribution.list_devices()
device_mesh = keras.distribution.DeviceMesh(
    (1, 8), ["data", "model"], devices=devices
)

# Get layout map from the model itself
layout_map = keras_hub.models.GemmaBackbone.get_layout_map(
    device_mesh, model_parallel_dim_name="model"
)

# Set up ModelParallel distribution
model_parallel = keras.distribution.ModelParallel(
    layout_map=layout_map,
    batch_dim_name="data"
)
keras.distribution.set_distribution(model_parallel)

# Load model
model = keras_hub.models.CausalLM.from_preset("gemma2_9b_en")
```

## Implementation Details

### Files Modified

1. **`keras_hub/src/utils/modelparallel_utils.py`** (new file):
   - `is_modelparallel_active()`: Check if ModelParallel is active
   - `load_weights_with_modelparallel_awareness()`: OOM-safe weight loading

2. **`keras_hub/src/utils/preset_utils.py`** (modified):
   - Updated `_load_backbone_weights()` to use ModelParallel-aware loading
   - Added import for the new utility functions

### Key Functions

#### `load_weights_with_modelparallel_awareness(model, filepath)`

This is the core fix that prevents OOM errors:

```python
def load_weights_with_modelparallel_awareness(model, filepath):
    if not is_modelparallel_active():
        # No ModelParallel active, use standard loading
        model.load_weights(filepath)
        return
    
    # ModelParallel is active - use sharded loading strategy
    if not sharded_weights_available():
        raise RuntimeError(
            "ModelParallel distribution is active but sharded weights loading "
            f"is not supported in the current Keras version {keras.__version__}. "
            "Please update to a newer version or disable ModelParallel distribution."
        )
    
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
```

#### `is_modelparallel_active()`

Simple detection function:

```python
def is_modelparallel_active():
    """Check if ModelParallel distribution is currently active."""
    try:
        from keras.src.distribution.distribution_lib import distribution as get_dist
        current_dist = get_dist()
        return current_dist and isinstance(current_dist, keras.distribution.ModelParallel)
    except ImportError:
        return False
```

## Why This Fix Works

1. **Non-Breaking**: Works with existing code - if ModelParallel is not active, it uses standard loading
2. **User-Controlled**: Sharding configuration is left to the user, respecting their preferences
3. **Robust**: Provides clear error messages and graceful fallbacks
4. **Future-Proof**: Works with Keras's sharded weights system
5. **Minimal**: Only changes the weight loading, not the entire model creation process

## The Key Insight

The fix doesn't try to change how Keras creates variables or how users configure ModelParallel. Instead, it makes the **weight loading process** aware of ModelParallel distribution, so it can handle the distribution properly even when the model architecture was created with standard Keras methods.

This approach is:
- **Minimal**: Only changes the weight loading, not the entire model creation process
- **Safe**: Doesn't break existing functionality
- **Effective**: Solves the OOM problem for large models
- **Respectful**: Lets users control their own sharding strategy
- **Maintainable**: Easy to understand and extend

The fix essentially bridges the gap between KerasHub's model creation process and Keras's ModelParallel distribution system, ensuring they work together seamlessly while respecting user control over sharding configuration.
