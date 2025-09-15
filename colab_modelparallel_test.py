#!/usr/bin/env python3
"""
Colab Test Case for ModelParallel OOM Fix

This notebook tests the ModelParallel OOM fix with debug prints.
Run this in Google Colab with TPU runtime to test the fix.

INSTRUCTIONS FOR COLAB:
1. First run these commands in separate cells:
   !pip uninstall keras keras-nlp keras-hub tf-keras -y
   !git clone -b modelparallel-oom-fix https://github.com/amitsrivastava78/keras-hub.git
   %cd keras-hub
   !pip install .
   !pip install keras-nlp

2. Restart runtime
3. Then run this script
"""

# --- STEP 1: INSTALL PACKAGES FIRST ---
print("--> STEP 1: Installing packages...")
print("Please run the installation commands in Colab cells first!")
print("Commands to run in Colab:")
print("!pip uninstall keras keras-nlp keras-hub tf-keras -y")
print(
    "!git clone -b modelparallel-oom-fix https://github.com/amitsrivastava78/keras-hub.git"
)
print("%cd keras-hub")
print("!pip install .")
print("!pip install keras-nlp")
print("Then restart runtime and run this script")

# --- STEP 3: TEST MODELPARALLEL OOM FIX ---
print("--> STEP 3: Testing ModelParallel OOM Fix...")

import os

os.environ["KERAS_BACKEND"] = "jax"

import jax
import keras

import keras_hub

print(f"Keras version: {keras.__version__}")
print(f"JAX devices: {jax.devices()}")

# Test ModelParallel utilities
print("\n🔍 Testing ModelParallel utilities...")
try:
    from keras_hub.src.utils import is_modelparallel_active
    from keras_hub.src.utils import load_weights_with_modelparallel_awareness

    print("✅ Successfully imported ModelParallel utilities!")

    # Test if ModelParallel is active (should be False initially)
    print("\n🔍 Testing is_modelparallel_active()...")
    is_active = is_modelparallel_active()
    print(f"ModelParallel active: {is_active}")

except ImportError as e:
    print(f"❌ Failed to import ModelParallel utilities: {e}")

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

# Test model creation with debug prints
print("\n🔍 Testing model creation with debug prints...")
try:
    print("Creating Gemma3 1B model...")
    model = keras_hub.models.CausalLM.from_preset(
        "gemma3_1b", load_weights=True
    )
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

print("\n🎉 Test completed!")
