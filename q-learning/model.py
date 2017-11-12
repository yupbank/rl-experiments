import tensorflow as tf

def Q(state, action_space):
    conv1 = tf.contrib.layers.conv2d(state, num_outputs=32, kernel_size=(8, 8), stride=(4, 4), scope='l1')
    conv2 = tf.contrib.layers.conv2d(conv1, num_outputs=64, kernel_size=(4, 4), stride=(2, 2), scope='l2')
    conv3 = tf.contrib.layers.conv2d(conv2, num_outputs=64, kernel_size=(3, 3), stride=(1, 1), scope='l3')
    fc = tf.contrib.layers.fully_connected(conv3, 512)
    prob = tf.contrib.layers.fully_connected(fc, action_space, activation_fn=None)
    return prob
