'''
A Reccurent Neural Network (LSTM) implementation example using TensorFlow library.

Based on:
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Edited by Tan

'''

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np


'''
Input datas, generate binary numbers from 1-128 and
labels as one-hot-vectors
'''
def genData(numFeat):
    numData = np.power(2,numFeat) #128
    binary = lambda n: n>0 and [n&1]+binary(n>>1) or []
    binaryNumbers = np.zeros((numData, numFeat))
    for n in range(1, numData):
        zeros = np.zeros(numFeat)
        bin = binary(n)
        #print(n, bin)
        for i in range(len(bin)-1, -1, -1):
            zeros[i] = bin[i]
            binaryNumbers[n] = zeros[::-1]

    labels = np.zeros((numData, numData))
    for i in range(0, labels.shape[0]):
        labels[i, i]=1

    return binaryNumbers, labels



# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 7 # MNIST data input (img shape: 28*28)
n_steps = 1 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 128 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()


#Mix the input data
_x, _y = genData(7)
mixedArray = np.arange(128)  # number of datas
np.random.shuffle(mixedArray)
_x_mix = np.zeros([128, 7])
_y_mix = np.zeros([128, 128])
for i in range(128):
    index = mixedArray[i]
    _x_mix[i] = _x[index]
    _y_mix[i] = _y[index]


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = _x_mix, _y_mix #mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    #test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    #test_label = mnist.test.labels[:test_len]
    _x = _x.reshape((128, n_steps, n_input))
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: _x, y: _y}))