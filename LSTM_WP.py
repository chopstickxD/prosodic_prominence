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
import preprocessing as pre
import sklearn.metrics as metrics

#Some other parameters
pathData = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/CNE/*16k.wav'
pathData2 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/GEN/*16k.wav'
pathData3 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/HRO/*16k.wav'
pathData4 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/ICO/*16k.wav'
pathData5 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/KTA/*16k.wav'

pathLabel = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_CNE.csv'
pathLabel2 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_GEN.csv'
pathLabel3 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_HRO.csv'
pathLabel4 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_ICO.csv'
pathLabel5 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_KTA.csv'

beginningStep = [8000, 14400, 19200, 24000, 30400]
wordStep = 4800
maxStep = 64000
sampleRate = 16000

#reading the data and preprocessing
examples1, labels1 = pre.makeInputSeq(pathData, pathLabel, beginningStep, wordStep, maxStep, sampleRate)
examples2, labels2 = pre.makeInputSeq(pathData2, pathLabel2, beginningStep, wordStep, maxStep, sampleRate)
examples3, labels3 = pre.makeInputSeq(pathData3, pathLabel3, beginningStep, wordStep, maxStep, sampleRate)
examples4, labels4 = pre.makeInputSeq(pathData4, pathLabel4, beginningStep, wordStep, maxStep, sampleRate)
test_data, test_labels = pre.makeInputSeq(pathData5, pathLabel5, beginningStep, wordStep, maxStep, sampleRate)

examples = np.concatenate((examples1, examples2, examples3, examples4))
labels = np.concatenate((labels1, labels2, labels3, labels4))
#print(len(examples))

# Parameters
learning_rate = 0.001
hyperParam = 0.01
batch_size = 100
numEx = examples.shape[0]
epochs = int(numEx/batch_size)+1
display_step = 1
training_iter = 150

# Network Parameters
n_input = 13 # MFCC frame Input shape numFrame*shape
n_steps = 29 # numFrames
n_hidden = 128 # hidden layer num of features
n_classes = 2 # Prominence classes [1, 0] prominent or [0, 1] not prominent


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

def l2regularization(weights, biases):
    l2reg = tf.nn.l2_loss(weights['out']) + tf.nn.l2_loss(biases['out'])
    return l2reg

'''
Mixing the Input
'''
def mixExamples(examples, labels):
    mixedArray = np.arange(len(examples))  # number of datas
    np.random.shuffle(mixedArray)
    _x_mix = np.zeros([numEx, 29, 13])
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
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#Mix the input data
_x, _y = mixExamples(examples, labels)

# Initializing the variables
init = tf.initialize_all_variables()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    iter = 1 #for the iteration
    step = 0 #for the data
    # Keep training until reach max iterations
    while iter <= training_iter:
        # index for the mini batch, reset if reach total batch size
        index = step * batch_size
        if index > numEx:
            step = 0
            index = 0
            _x, _y = mixExamples(examples, labels)
        else:
            step += 1

        # mini batch
        batch_x, batch_y = _x[index:(index+batch_size)], _y[index:(index+batch_size)]
        # Reshape data to get 29 seq of 13 elements
        batch_x = batch_x.reshape((batch_x.shape[0], n_steps, n_input))

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if iter % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print ("Iteration " + str(iter) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        iter += 1
    print("Optimization Finished!")

    # Calculate accuracy
    _x = test_data.reshape((len(test_data), n_steps, n_input))
    _y = test_labels

    acc1 , _pred_ = sess.run([accuracy, pred1], feed_dict={x: _x, y: _y})
    print("Testing Accuracy:", acc1)

    _y_ = np.argmax(_y, 1)
    print("Confusion Matrix\n", metrics.confusion_matrix(_y_, _pred_, [0, 1]))
    print("Accuracy Score\n", metrics.accuracy_score(_y_, _pred_))
    print("F1 Score\n", metrics.f1_score(_y_, _pred_))
    print("Recall Score\n", metrics.recall_score(_y_, _pred_))
    print("Precision Score\n", metrics.precision_score(_y_, _pred_))
