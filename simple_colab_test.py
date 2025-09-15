#!/usr/bin/env python3
"""
Simple Colab Test for ModelParallel OOM Fix - Using only from_preset()

This test only uses from_preset() to load models and see the debug prints.
No direct imports of ModelParallel utilities needed.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import jax
import keras

import keras_hub

print(f"Keras version: {keras.__version__}")
print(f"JAX devices: {jax.devices()}")

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

# Test model creation with debug prints
print("\n🔍 Testing model creation with debug prints...")
print("This will show debug output from our ModelParallel utilities!")
print("Watch for the 🔍 DEBUG messages below...\n")

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
print("If you saw 🔍 DEBUG messages above, our ModelParallel fix is working!")
