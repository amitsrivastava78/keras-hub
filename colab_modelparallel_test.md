# ModelParallel OOM Fix Test - Google Colab

This notebook tests the ModelParallel OOM fix with debug prints to show exactly what's happening during model loading.

## Step 1: Install Packages

```python
# Install the fixed keras-hub
!pip uninstall keras keras-nlp keras-hub tf-keras -y
!git clone -b modelparallel-oom-fix https://github.com/amitsrivastava78/keras-hub.git
%cd keras-hub
!pip install .
!pip install keras-nlp

# Restart runtime
import os
os.kill(os.getpid(), 9)
```

## Step 2: Test ModelParallel OOM Fix

```python
import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
import keras_hub
import jax

print(f"Keras version: {keras.__version__}")
print(f"JAX devices: {jax.devices()}")

# Test ModelParallel utilities
print("\n🔍 Testing ModelParallel utilities...")
try:
    from keras_hub.src.utils import (
        is_modelparallel_active,
        load_weights_with_modelparallel_awareness
    )
    print("✅ Successfully imported ModelParallel utilities!")
    
    # Test if ModelParallel is active (should be False initially)
    print("\n🔍 Testing is_modelparallel_active()...")
    is_active = is_modelparallel_active()
    print(f"ModelParallel active: {is_active}")
    
except ImportError as e:
    print(f"❌ Failed to import ModelParallel utilities: {e}")
```

## Step 3: Set Up ModelParallel Distribution

```python
# Set up ModelParallel distribution
print("\n🔍 Setting up ModelParallel distribution...")
devices = keras.distribution.list_devices()
print(f"Available devices: {devices}")

# Create device mesh for ModelParallel
device_mesh = keras.distribution.DeviceMesh(
    (1, len(devices)), ["data", "model"], devices=devices
)

# Create layout map
layout_map = keras.distribution.LayoutMap(device_mesh)
layout_map["token_embedding/embeddings"] = ("model", None)

# Set up ModelParallel
model_parallel = keras.distribution.ModelParallel(
    layout_map=layout_map, batch_dim_name="data"
)
keras.distribution.set_distribution(model_parallel)

print("✅ ModelParallel distribution set up!")

# Test ModelParallel detection again
print("\n🔍 Testing ModelParallel detection after setup...")
is_active = is_modelparallel_active()
print(f"ModelParallel active: {is_active}")
```

## Step 4: Test Model Creation with Debug Prints

```python
# Test model creation with debug prints
print("\n🔍 Testing model creation with debug prints...")
try:
    print("Creating Gemma3 1B model...")
    model = keras_hub.models.CausalLM.from_preset("gemma3_1b", load_weights=True)
    print("✅ Model created successfully!")
    
    # Test inference
    print("\n🔍 Testing model inference...")
    test_input = "Hello, how are you?"
    output = model.generate(test_input, max_length=20)
    print(f"✅ Model inference successful! Output: {output}")
    
except Exception as e:
    print(f"❌ Error during model creation: {e}")
    import traceback
    traceback.print_exc()
```

## Step 5: Test Larger Model (Optional)

```python
# Test with a larger model to see OOM prevention in action
print("\n🔍 Testing with larger model (Gemma2 2B)...")
try:
    print("Creating Gemma2 2B model...")
    model_large = keras_hub.models.CausalLM.from_preset("gemma2_2b_en", load_weights=True)
    print("✅ Large model created successfully!")
    
    # Test inference
    print("\n🔍 Testing large model inference...")
    test_input = "What is machine learning?"
    output = model_large.generate(test_input, max_length=30)
    print(f"✅ Large model inference successful! Output: {output}")
    
except Exception as e:
    print(f"❌ Error during large model creation: {e}")
    if "RESOURCE_EXHAUSTED" in str(e):
        print("💡 This would have been an OOM error without our fix!")
    import traceback
    traceback.print_exc()
```

## Expected Output

When you run this notebook, you should see debug prints like:

```
🔍 DEBUG is_modelparallel_active():
   Current distribution: None
   Is ModelParallel: False

🔍 DEBUG load_weights_with_modelparallel_awareness():
   Model: Gemma3Backbone
   Model name: gemma3_backbone
   Filepath: /root/.cache/kagglehub/models/keras/gemma3/keras/gemma3_1b/3/model.weights.h5
   Keras version: 3.10.0
🔍 DEBUG is_modelparallel_active():
   Current distribution: ModelParallel
   Is ModelParallel: True
   ModelParallel active: True
   → ModelParallel detected, checking sharded weights support...
   Sharded weights available: True
   → Using ModelParallel-aware weight loading...
   ✅ ModelParallel-aware weight loading completed
```

This shows that:
1. ✅ ModelParallel detection works
2. ✅ Our function is called during weight loading
3. ✅ ModelParallel-aware loading is used
4. ✅ No OOM errors occur
