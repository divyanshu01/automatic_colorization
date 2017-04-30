import tensorflow as tf

def conv2d(x, w, b):
	conv = tf.nn.conv2d(x,w, strides=[1, 1, 1, 1], padding='SAME')
	return tf.nn.bias_add(conv, b)

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.02)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape)
	return tf.Variable(initial)

def relu(conv):
	return tf.nn.relu(conv)

def avg_pool(x):
	return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

