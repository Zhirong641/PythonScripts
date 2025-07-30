import tensorflow as tf

print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPUs detected: {gpus}")
else:
    print("🚫 No GPU detected")
