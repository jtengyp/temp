
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIRECTORY = '/home/jteng/jteng/data_program/data/mnist'
IMAGE_SIZE = 28
NUM_LABELS = 10
NUM_CHANNELS = 1
REGULARIZER_WEIGHT = 5e-4
BATCH_SIZE = 50
NUM_EPOCHS = 10000
LEARNING_RATE = 0.001
VALID_FREQUENCY = 100
SEED = 66478	

def main():
	mnist = input_data.read_data_sets(DATA_DIRECTORY, one_hot=True)

	x = tf.placeholder('float32', shape=[None, 784])
	y = tf.placeholder('float32', shape=[None, 10])

	x_image = tf.reshape(x, [-1, 28, 28, 1])

	W_conv1 = tf.Variable(
				tf.truncated_normal([5,5,NUM_CHANNELS, 32], 
								stddev = 0.1,
								seed = SEED,
								dtype = tf.float32))
	b_conv1 = tf.Variable(tf.zeros([32]), dtype = tf.float32)

	W_conv2 = tf.Variable(
				tf.truncated_normal([5,5,32,64],
									stddev = 0.1,
									seed = SEED,
									dtype = tf.float32))
	b_conv2 = tf.Variable(tf.constant(0.1, shape = [64]), dtype = tf.float32)

	W_fc1 = tf.Variable(
				tf.truncated_normal([IMAGE_SIZE//4 * IMAGE_SIZE//4 * 64, 1024],
									stddev = 0.1,
									seed = SEED,
									dtype = tf.float32))
	b_fc1 = tf.Variable(tf.constant(0.1, shape = [1024], dtype = tf.float32))

	W_fc2 = tf.Variable(
				tf.truncated_normal([1024, NUM_LABELS],
									stddev = 0.1,
									seed = SEED,
									dtype = tf.float32))
	b_fc2 = tf.Variable(tf.constant(0.1, shape = [NUM_LABELS], dtype = tf.float32))

	def model(data, keep_prob = 0.5, train = False):
		conv1 = tf.nn.conv2d(data, W_conv1, strides = [1,1,1,1], padding = 'SAME')
		h1 = tf.nn.relu(tf.nn.bias_add(conv1, b_conv1))
		pool1 = tf.nn.max_pool(h1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

		conv2 = tf.nn.conv2d(pool1, W_conv2, strides = [1,1,1,1], padding = 'SAME')
		h2 = tf.nn.relu(tf.nn.bias_add(conv2, b_conv2))
		pool2 = tf.nn.max_pool(h2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

		#pool2_shape = pool2.get_shape().as_list()
		#pool2_flat = tf.reshape(pool2, [pool2_shape[0], pool2_shape[1]*pool2_shape[2]*pool2_shape[3]])
		pool2_flat = tf.reshape(pool2, [-1, 7*7*64])

		fc1 = tf.nn.relu(tf.matmul(pool2_flat, W_fc1) + b_fc1)
		if train:
			fc1 = tf.nn.dropout(fc1, keep_prob, seed=SEED)

		y_pred = tf.matmul(fc1, W_fc2) + b_fc2
		return y_pred

	logits = model(x_image, keep_prob = 0.5, train = True)
	prediction = tf.nn.softmax(logits)

	correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	valid_logits = model(x_image)
	valid_prediction = tf.nn.softmax(valid_logits)
	valid_correct_pred = tf.equal(tf.argmax(valid_prediction,1), tf.argmax(y,1))
	valid_accuracy = tf.reduce_mean(tf.cast(valid_correct_pred, tf.float32))

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
						labels = y, logits = logits))
	regularizer = (tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
			  tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))

	loss += REGULARIZER_WEIGHT * regularizer

	batch = tf.Variable(0, dtype = tf.float32)
	train_size = mnist.train.labels.shape[0]
	'''
	learning_rate = tf.train.exponential_decay(
					LEARNING_RATE,		# base learning rate
					batch * BATCH_SIZE,  # current index into the dataset
					train_size,			# decay step
					0.95,				# decay rate
					staircase=True)
	optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step = batch)
	'''
	optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		print('Initialized!')
		for step in xrange(NUM_EPOCHS):
			batch_data = mnist.train.next_batch(BATCH_SIZE)
			feed_dict = {x: batch_data[0], y: batch_data[1]}
			sess.run(optimizer, feed_dict = feed_dict)

			if step % VALID_FREQUENCY == 0:
				l, predictions, train_accuracy = sess.run([loss, prediction, accuracy], feed_dict = feed_dict)
				valid_set_accuracy = sess.run(valid_accuracy, feed_dict = {x: mnist.validation.images[:], y: mnist.validation.labels[:]})
				print 'Step: %d, Loss: %.4f, Train accuracy: %.4f, Valid accuracy: %.4f' % (step, l, train_accuracy, valid_set_accuracy)

	
if __name__ == '__main__':
	main()
