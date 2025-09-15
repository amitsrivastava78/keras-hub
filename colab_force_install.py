# Colab Force Installation - Run this in Colab to fix the installation

# Step 1: Completely remove old versions
!pip uninstall keras keras-nlp keras-hub tf-keras -y
!pip cache purge

# Step 2: Remove any cached installations
!rm -rf /usr/local/lib/python3.12/dist-packages/keras_hub*
!rm -rf /root/.cache/pip

# Step 3: Clone and install our fixed version
!git clone -b modelparallel-oom-fix https://github.com/amitsrivastava78/keras-hub.git
%cd keras-hub

# Step 4: Force reinstall
!pip install --force-reinstall --no-cache-dir .
!pip install keras-nlp

# Step 5: Verify installation
print("🔍 Verifying installation...")
import sys
print(f"Python path: {sys.path}")

# Check if our version is installed
try:
    import keras_hub
    print(f"KerasHub location: {keras_hub.__file__}")
    
    # Check if our functions are available
    from keras_hub.src.utils import is_modelparallel_active, load_weights_with_modelparallel_awareness
    print("✅ ModelParallel utilities imported successfully!")
    
    # Check source code
    import inspect
    source = inspect.getsource(load_weights_with_modelparallel_awareness)
    if "🔍 DEBUG load_weights_with_modelparallel_awareness" in source:
        print("✅ Debug prints found in source code!")
        print("🎉 Installation successful!")
    else:
        print("❌ Debug prints NOT found - installation failed!")
        
except Exception as e:
    print(f"❌ Installation failed: {e}")
    print("Please restart runtime and try again!")
