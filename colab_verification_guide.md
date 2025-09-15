# Colab Installation Verification

Run this in Colab to verify our ModelParallel fix is properly installed:

## Step 1: Install the fixed version
```bash
!pip uninstall keras keras-nlp keras-hub tf-keras -y
!git clone -b modelparallel-oom-fix https://github.com/amitsrivastava78/keras-hub.git
%cd keras-hub
!pip install .
!pip install keras-nlp
```

## Step 2: Restart runtime

## Step 3: Run verification test
```python
exec(open('colab_installation_test.py').read())
```

## Expected Output

If the installation worked correctly, you should see:

```
🔍 Test 1: Checking if ModelParallel utilities are importable...
✅ ModelParallel utilities imported successfully!

🔍 Test 2: Testing debug prints...
============================================================
🔍 DEBUG is_modelparallel_active():
   Current distribution: ModelParallel
   Is ModelParallel: True
============================================================
ModelParallel active: True

🔍 Test 3: Checking source code...
✅ Debug prints found in source code!

🔍 Test 4: Setting up ModelParallel...
Available devices: ['tpu:0', 'tpu:1', 'tpu:2', 'tpu:3', 'tpu:4', 'tpu:5', 'tpu:6', 'tpu:7']
✅ ModelParallel distribution set up!

🔍 Test 5: Testing model creation...
Creating Gemma3 1B model...
🎯 KerasPresetLoader.load_task() called!
   Task class: Gemma3CausalLM
   load_weights: True
   load_task_weights: True
🎯 Task loading: load_weights=True, calling _load_backbone_weights
🎯 Task loading: No task weights, cleaning up backbone memory
🎯 Task loading: About to call _load_backbone_weights
🚀 _load_backbone_weights() called!
   Backbone: Gemma3Backbone
   Backbone name: gemma3_backbone
================================================================================
🔍 DEBUG load_weights_with_modelparallel_awareness():
   Model: Gemma3Backbone
   Model name: gemma3_backbone
   Filepath: /root/.cache/kagglehub/models/keras/gemma3/keras/gemma3_1b/3/model.weights.h5
   Keras version: 3.11.3
================================================================================
============================================================
🔍 DEBUG is_modelparallel_active():
   Current distribution: ModelParallel
   Is ModelParallel: True
============================================================
   ModelParallel active: True
   → ModelParallel detected, checking sharded weights support...
   Sharded weights available: True
   → Using ModelParallel-aware weight loading...
   ✅ ModelParallel-aware weight loading completed
🎯 Task loading: _load_backbone_weights completed
✅ Model created successfully!

🔍 Test 6: Testing inference...
✅ Inference successful! Output: Hello, how are you?

🎉 Test completed!
```

## Troubleshooting

If you don't see the debug prints:

1. **Check Test 3**: If it says "Debug prints NOT found in source code!", the installation didn't work
2. **Try reinstalling**: Make sure to restart runtime after installation
3. **Check branch**: Verify you're on the `modelparallel-oom-fix` branch
4. **Check imports**: If Test 1 fails, there's an import issue

The debug prints confirm that:
- ✅ Our ModelParallel utilities are working
- ✅ The loading path goes through our functions
- ✅ ModelParallel detection works correctly
- ✅ OOM-safe weight loading is used
