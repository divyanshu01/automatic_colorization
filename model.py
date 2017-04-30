import tensorflow as tf
import functions as fun

def model(image):
	w1_1 = fun.weight_variable([3, 3, 1, 64])
	b1_1 = fun.bias_variable([64])
	conv1_1 = fun.conv2d(image, w1_1, b1_1)
	relu1_1 = fun.relu(conv1_1)
	pool1 = fun.avg_pool(relu1_1)

	
	w2_1 = fun.weight_variable([3, 3, 64, 128])
	b2_1 = fun.weight_variable([128])
	conv2_1 = fun.conv2d(pool1, w2_1, b2_1)
	relu2_1 = fun.relu(conv2_1)
	
	w2_2 = fun.weight_variable([3, 3, 128, 128])
	b2_2 = fun.weight_variable([128])
	conv2_2 = fun.conv2d(relu2_1, w2_2, b2_2)
	relu2_2 = fun.relu(conv2_2)
	pool2 = fun.avg_pool(relu2_2)


	w3_1 = fun.weight_variable([3, 3, 128, 64])
	b3_1 = fun.bias_variable([64])
	conv3_1 = fun.conv2d(pool2, w3_1, b3_1)
	relu3_1 = fun.relu(conv3_1)

	w3_2 = fun.weight_variable([3, 3, 64, 64])
	b3_2 = fun.bias_variable([64])
	conv3_2 = fun.conv2d(relu3_1, w3_2, b3_2)
	relu3_2 = fun.relu(conv3_2)

	w3_3 = fun.weight_variable([3, 3, 64, 64])
	b3_3 = fun.bias_variable([64])
	conv3_3 = fun.conv2d(relu3_2, w3_3, b3_3)
	relu3_3 = fun.relu(conv3_3)

	w3_4 = fun.weight_variable([3, 3, 64, 32])
	b3_4 = fun.bias_variable([32])
	conv3_4 = fun.conv2d(relu3_3, w3_4, b3_4)
	relu3_4 = fun.relu(conv3_4)
	pool3 = fun.avg_pool(relu3_4)


	w4_1 = fun.weight_variable([3, 3, 32, 32])
	b4_1 = fun.bias_variable([32])
	conv4_1 = fun.conv2d(pool3, w4_1, b4_1)
	relu4_1 = fun.relu(conv4_1)

	w4_2 = fun.weight_variable([3, 3, 32, 32])
	b4_2 = fun.bias_variable([32])
	conv4_2 = fun.conv2d(relu4_1, w4_2, b4_2)
	relu4_2 = fun.relu(conv4_2)

	w4_3 = fun.weight_variable([3, 3, 32, 32])
	b4_3 = fun.bias_variable([3, 3, 32, 32])
	conv4_3 = fun.conv2d(relu4_2, w4_3, b4_3)
	relu4_3 = fun.relu(conv4_3)

	w4_4 = fun.weight_variable([3, 3, 32, 32])
	b4_4 = fun.bias_variable([3, 3, 32, 32])
	conv4_4 = fun.conv2d(relu4_3, w4_4, b4_4)
	relu4_4 = fun.relu(conv4_4)
	pool4 = fun.avg_pool(relu4_4)


	w5_1 = fun.weight_variable([3, 3, 32, 64])
	b5_1 = fun.bias_variable([64])
	conv5_1 = fun.conv2d(pool4, w5_1, b5_1)
	relu5_1 = fun.relu(conv5_1)

	w5_2 = fun.weight_variable([3, 3, 64, 64])
