# Quick Colab Test - Run this single line in Colab

exec("""
import os
os.environ['KERAS_BACKEND'] = 'jax'
import keras, keras_hub, jax
print(f'Keras: {keras.__version__}, JAX devices: {len(jax.devices())}')
try:
    from keras_hub.src.utils import is_modelparallel_active, load_weights_with_modelparallel_awareness
    print('✅ ModelParallel utilities imported!')
    import inspect
    source = inspect.getsource(load_weights_with_modelparallel_awareness)
    if '🔍 DEBUG load_weights_with_modelparallel_awareness' in source:
        print('✅ Debug prints found in source!')
    else:
        print('❌ Debug prints NOT found - using old version!')
except Exception as e:
    print(f'❌ Error: {e}')
""")
