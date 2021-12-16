import tensorflow as tf
with tf.device(tf.DeviceSpec(device_type="GPU", device_index=1)):
    from tensorflow.python.client import device_lib 
    print(device_lib.list_local_devices())
    
    
X_test_text = np.array(["banyak gaya urus dulu kemiskinan pengangguran korupsi demokrasi"])
X_test_emoji = np.array(["🤣🤣"])
y_test_sarcasm = np.array([1])