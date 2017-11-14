import tensorflow as tf

NUM_OF_ACTION = 4

def q_function(state, num_of_action=NUM_OF_ACTION, scope=''):
    with tf.variable_scope(scope):
        conv1 = tf.contrib.layers.conv2d(state, num_outputs=32, kernel_size=(8, 8), stride=(4, 4), scope='l1')
        conv2 = tf.contrib.layers.conv2d(conv1, num_outputs=64, kernel_size=(4, 4), stride=(2, 2), scope='l2')
        conv3 = tf.contrib.layers.conv2d(conv2, num_outputs=64, kernel_size=(3, 3), stride=(1, 1), scope='l3')
        flattened = tf.contrib.layers.flatten(conv3, scope='flattened')
        fc = tf.contrib.layers.fully_connected(flattened, 512, scope='fc')
        score = tf.contrib.layers.fully_connected(fc, num_of_action, activation_fn=None, scope='action_score')
        return score


def prepare_imgae(input_placeholder):
    gray_image = tf.image.rgb_to_grayscale(input_placeholder)
    return tf.image.resize_images(gray_image, [80, 80])


def prepare_action(action_holder, num_of_action=NUM_OF_ACTION):
    return tf.one_hot(action_holder, num_of_action, 1.0, 0.0, name='action_one_hot')

