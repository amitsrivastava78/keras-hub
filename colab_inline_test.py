# Colab Inline Test - Copy and paste this into a Colab cell

import os

os.environ["KERAS_BACKEND"] = "jax"

import jax
import keras

import keras_hub

print(f"Keras version: {keras.__version__}")
print(f"JAX devices: {jax.devices()}")

# Test 1: Check if our functions are importable
print("\n🔍 Test 1: Checking if ModelParallel utilities are importable...")
try:
    from keras_hub.src.utils import is_modelparallel_active
    from keras_hub.src.utils import load_weights_with_modelparallel_awareness

    print("✅ ModelParallel utilities imported successfully!")

    # Test 2: Check if our debug prints work
    print("\n🔍 Test 2: Testing debug prints...")
    is_active = is_modelparallel_active()
    print(f"ModelParallel active: {is_active}")

except ImportError as e:
    print(f"❌ Failed to import ModelParallel utilities: {e}")

# Test 3: Check if our debug prints are in the source code
print("\n🔍 Test 3: Checking source code...")
try:
    import inspect

    source = inspect.getsource(load_weights_with_modelparallel_awareness)
    if "🔍 DEBUG load_weights_with_modelparallel_awareness" in source:
        print("✅ Debug prints found in source code!")
    else:
        print("❌ Debug prints NOT found in source code!")
        print("This means Colab is using an old version!")
except Exception as e:
    print(f"❌ Error checking source code: {e}")

# Test 4: Set up ModelParallel and test
print("\n🔍 Test 4: Setting up ModelParallel...")
devices = keras.distribution.list_devices()
print(f"Available devices: {devices}")

device_mesh = keras.distribution.DeviceMesh(
    (1, len(devices)), ["data", "model"], devices=devices
)

layout_map = keras.distribution.LayoutMap(device_mesh)
layout_map["token_embedding/embeddings"] = ("model", None)

model_parallel = keras.distribution.ModelParallel(
    layout_map=layout_map, batch_dim_name="data"
)
keras.distribution.set_distribution(model_parallel)

print("✅ ModelParallel distribution set up!")

# Test 5: Test model creation
print("\n🔍 Test 5: Testing model creation...")
try:
    print("Creating Gemma3 1B model...")
    model = keras_hub.models.CausalLM.from_preset(
        "gemma3_1b", load_weights=True
    )
    print("✅ Model created successfully!")

    # Test inference
    print("\n🔍 Test 6: Testing inference...")
    test_input = "Hello, how are you?"
    output = model.generate(test_input, max_length=20)
    print(f"✅ Inference successful! Output: {output}")

except Exception as e:
    print(f"❌ Error during model creation: {e}")
    import traceback

    traceback.print_exc()

print("\n🎉 Test completed!")
