
# find the device type TF is running on
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU not found')
print('Found GPU at: {}'.format(device_name))

# to obtain additional information about hardware
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

# for cpu and ram information
!cat /proc.cpuinfo
!cat /proc/meminfo


