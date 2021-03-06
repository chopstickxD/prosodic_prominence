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
import sklearn.metrics as metrics
import random_search_params as hyperParams
from generatData import examples, labels, test_data, test_labels, useTest

# Parameters
useRandSearch = 0
if useRandSearch is 1:
    searchSpace_lstm = [0.01, 10, 0.1, 1000, 1, 1, 1000]
    numParams_lstm = 7
    parameters = hyperParams.random_search_params(numParams_lstm, searchSpace_lstm)
    learning_rate = parameters[0]  # 0.0050733 #0.001
    forget_bias = parameters[1]  # 5.91269 #3.0
    hyperParam = parameters[2]  # 0.08105 #0.0625 #best result so far with 0.0625 and 10
    batch_size = parameters[3]  # 737 #200
    input_keep_prob = parameters[4]
    output_keep_prob = parameters[5]
    n_hidden = int(parameters[6]) # hidden layer num of features
else:
    #searchSpace_lstm = [1, 1]
    #numParams_lstm = 2
    #parameters = hyperParams.random_search_params(numParams_lstm, searchSpace_lstm)
    learning_rate = 0.0050733  # 0.001
    forget_bias = 5.91269  # 3.0
    hyperParam = 0.06105  # 0.0625 #best result so far with 0.0625 and 10
    batch_size = 737 #811  # 737 #200
    input_keep_prob = 0.85174  # parameters[0]
    output_keep_prob = 0.75501  # parameters[1]
    n_hidden = 400 #661   # hidden layer num of features #100 worked well

numEx = examples.shape[0]
# epochs = int(numEx/batch_size)+1
display_step = 1
training_iter = 90
training_epoch = 20
numFrames = examples.shape[1]
#useTest = 1  # use data for testing 1 without speaker, 0 with speaker
print("Data ready, time to learn! Here are the parameters: \n")
print("useTest: ", useTest)
print("learning_rate: ", learning_rate)
print("forget_bias", forget_bias)
print("batch_size: ", batch_size)
print("hyperParam: ", hyperParam)
print("n_hidden: ", n_hidden)

print("input_keep_prob", input_keep_prob)
print("output_keep_prob", output_keep_prob)

# Network Parameters
n_input = 13  # MFCC frame Input shape numFrame*shape
n_steps = numFrames  # numFrames
n_classes = 2  # Prominence classes [1, 0] prominent or [0, 1] not prominent
drop = 1

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

input_keep_prob_tensor = tf.placeholder(tf.float32)
output_keep_prob_tensor = tf.placeholder(tf.float32)

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
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=forget_bias)

    dropout = rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=input_keep_prob_tensor,
                                      output_keep_prob=output_keep_prob_tensor)
    # Get lstm cell output
    outputs, states = rnn.rnn(dropout, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def l2regularization(weights, biases):
    l2reg = tf.nn.l2_loss(weights['out']) + tf.nn.l2_loss(biases['out'])
    return l2reg


'''
Mixing the Input
'''


def mixExamples(examples, labels, numFrames, numEx):
    mixedArray = np.arange(len(examples))  # number of datas
    np.random.shuffle(mixedArray)
    _x_mix = np.zeros([numEx, numFrames, 13])
    _y_mix = np.zeros([numEx, 2])
    for i in range(numEx):
        index = mixedArray[i]
        _x_mix[i] = examples[index]
        _y_mix[i] = labels[index]
    return _x_mix, _y_mix


pred = RNN(x, weights, biases)
pred1 = tf.argmax(pred, 1)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

if hyperParam is not None:
    cost += l2regularization(weights, biases) * hyperParam

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Mix the input data
_x, _y = mixExamples(examples, labels, numFrames, numEx)


numEx = len(_x)

print("_x before: ", _x.shape)
print("_y before: ", len(_y))

mini_bag = int(len(_x) / 6)




'''for using test/valid data in training accuracy'''
if useTest is 1:
    #valid_double = test_data

    numTest = len(test_data)
    valid_x = test_data.reshape((numTest, n_steps, n_input))
    valid_y = test_labels
    valid_x, valid_y = mixExamples(valid_x, valid_y, numFrames, numTest)
else:
    valid_data = _x[0:mini_bag]
    valid_labels = _y[0:mini_bag]
    #valid_double = valid_data

    indices = np.arange(mini_bag)
    _x = np.delete(_x, indices, 0)
    _y = np.delete(_y, indices, 0)

    numTest = len(valid_data)
    valid_x = valid_data.reshape((numTest, n_steps, n_input))
    valid_y = valid_labels
    valid_x, valid_y = mixExamples(valid_x, valid_y, numFrames, numTest)


#made sure that validation data is not part of training data
counter = 0
for data in valid_x:
    current = data
    for data2 in _x:
        if np.array_equal(current, data2):
            counter += 1

print("counter: ", counter)


print("_x after: ", _x.shape)
print("_y after: ", len(_y))
print("valid_data: ", valid_x.shape)
print("valid_labels: ", len(valid_y))


# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    iter = 1  # for the iteration
    step = 0  # for the data
    epoch = 0
    numEx = len(_x)
    # Keep training until reach max iterations
    while epoch < training_epoch:
        # index for the mini batch, reset if reach total batch size
        index = step * batch_size
        if index > numEx:
            step = 0
            index = 0

            _x, _y = mixExamples(_x, _y, numFrames, numEx)
            batch_x, batch_y = _x[index:(index + batch_size)], _y[index:(index + batch_size)]

            epoch += 1
            step += 1
            print("Epoch ", epoch, " done")
        else:
            step += 1
            # mini batch
            batch_x, batch_y = _x[index:(index + batch_size)], _y[index:(index + batch_size)]

        # Reshape data to get 29 seq of 13 elements
        batch_x = batch_x.reshape((batch_x.shape[0], n_steps, n_input))

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       input_keep_prob_tensor: input_keep_prob,
                                       output_keep_prob_tensor: output_keep_prob})

        if iter % display_step == 0 and epoch < training_epoch:
            # Calculate batch accuracy
            # print(valid_x.shape)
            acc, _pred_ = sess.run([accuracy, pred1], feed_dict={x: valid_x, y: valid_y,
                                                                 input_keep_prob_tensor: 1.0,
                                                                 output_keep_prob_tensor: 1.0})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y,
                                             input_keep_prob_tensor: input_keep_prob,
                                             output_keep_prob_tensor: output_keep_prob})

            print("Iteration " + str(iter) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        iter += 1
        if epoch >= training_epoch:
            valid_y_ = np.argmax(valid_y, 1)
            confusion_matrix = metrics.confusion_matrix(valid_y_, _pred_, [0, 1])
            print("Last Training: Confusion Matrix\n", confusion_matrix)
            print("Last Training: Accuracy Score\n", metrics.accuracy_score(valid_y_, _pred_))
            print("Last Training: F1 Score\n", metrics.f1_score(valid_y_, _pred_))
            print("Last Training: Recall Score\n", metrics.recall_score(valid_y_, _pred_))
            print("Last Training: Precision Score\n", metrics.precision_score(valid_y_, _pred_))

    print("Optimization Finished!\n\n")

    print("learning_rate: ", learning_rate, "forget_bias", forget_bias, "batch_size: ",
          batch_size, "\n hyperParam: ", hyperParam, "input_keep_prob: ", input_keep_prob,
          "output_keep_prob: ", output_keep_prob, "n_hidden: ", n_hidden, "useTest:", useTest, "\n")

    print("Performance Metrics:\n")
    # Calculate accuracy

    '''
    if useTest is 1:

        'here using a speaker which is not in training data to test'
        numTest = len(test_data)
        _x = test_data.reshape((numTest, n_steps, n_input))
        _y = test_labels
        _x, _y = mixExamples(_x, _y, numFrames, numTest)

        _y_ = np.argmax(_y, 1)
        print("Used Test\n")
    else:

        'here exclude a part of the data to test'
        numTest = len(valid_data)
        _x = valid_data.reshape((numTest, n_steps, n_input))
        _y = valid_labels
        _x, _y = mixExamples(_x, _y, numFrames, numTest)
        print("Used Valid\n")
    '''

    # made sure that validation data is not part of training data
    counter = 0
    for data in valid_x:
        current = data
        for data2 in _x:
            if np.array_equal(current, data2):
                counter += 1
    print("counter: ", counter)

    print("_x after: ", len(_x))
    print("_y after: ", len(_y))
    print("valid_data: ", len(valid_x))
    print("valid_labels: ", len(valid_y))


    _y_ = np.argmax(valid_y, 1)

    # print(_x.shape)

    acc1, _pred_ = sess.run([accuracy, pred1], feed_dict={x: valid_x, y: valid_y,
                                                          input_keep_prob_tensor: 1.0,
                                                          output_keep_prob_tensor: 1.0})
    print("Testing Accuracy:", acc1)

    confusion_matrix = metrics.confusion_matrix(_y_, _pred_, [0, 1])
    recall = metrics.recall_score(_y_, _pred_)
    specificity = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0])
    uwAccuracy = (recall + specificity) / 2

    print("Confusion Matrix\n", confusion_matrix)
    print("Accuracy Score\n", metrics.accuracy_score(_y_, _pred_))
    print("Unweighted Accuracy\n", uwAccuracy)
    print("F1 Score\n", metrics.f1_score(_y_, _pred_))
    print("Recall Score\n", recall)
    print("Precision Score\n", metrics.precision_score(_y_, _pred_))
