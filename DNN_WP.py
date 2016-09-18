'''
A Multilayer Perceptron implementation example using TensorFlow library.
based on 'Author: Aymeric Damien Project: https://github.com/aymericdamien/TensorFlow-Examples/'

Now its a DNN with 6 hidden_layers
and edited by Tan
'''


import tensorflow as tf
import numpy as np
import sklearn.metrics as metrics
import random_search_params as hyperParams
from generatData import examples, labels, test_data, test_labels, useTest



# Parameters
searchSpace_lstm = [0.001, 0.1, 1000, 1]
numParams_dnn = 4

parameters = hyperParams.random_search_params(numParams_dnn, searchSpace_lstm)

learning_rate = 0.0007533#parameters[0]#0.000139#0.0005986
hyperParam = 0.0463#parameters[1]#0.0699
batch_size = 162#parameters[2] #162 #350
keep_prob =  0.2515#parameters[3]

numEx = len(examples)
display_step = 1
training_iter = 50
numFrames = examples.shape[1] #frames of the mfcc

print("learning_rate: ", learning_rate)
print("batch_size: ", batch_size)
print("hyperParam: ", hyperParam)
print("keep_prob", keep_prob)

# Network Parameters
n_hidden_1 = 840  # 1st layer number of features
n_hidden_2 = 810  # 2nd layer number of features
n_hidden_3 = 800  # 3rd layer number of features
n_hidden_4 = 795  # 4th layer number of features
n_hidden_5 = 785  # 5th layer number of features
n_hidden_6 = 703  # 6th layer number of features
n_hidden_7 = 694  # 7th layer number of features      694 o. 794
n_hidden_8 = 689  # 8th layer number of features 770
n_input = 13*numFrames      # data input (13mfcc*numframes)
n_classes = 2  # total classes [0 1] [1 0]



# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

keep_prob_tensor = tf.placeholder(tf.float32)


# Create model
def multilayer_perceptron(x, weights, biases):
    # Reshaping to (n_steps*batch_size, n_input)

    #x = tf.reshape(x, [-1, n_input])

    # Hidden layer with RELU activation
    layer_1 = computation_layer(x, weights['h1'], biases['b1'])
    layer_2 = computation_layer(layer_1, weights['h2'], biases['b2'])
    layer_3 = computation_layer(layer_2, weights['h3'], biases['b3'])
    layer_4 = computation_layer(layer_3, weights['h4'], biases['b4'])
    layer_5 = computation_layer(layer_4, weights['h5'], biases['b5'])
    layer_6 = computation_layer(layer_5, weights['h6'], biases['b6'])
    layer_7 = computation_layer(layer_6, weights['h7'], biases['b7'])
    layer_8 = computation_layer(layer_7, weights['h8'], biases['b8'])
    # Output layer with linear activation

    dropout = tf.nn.dropout(layer_8, keep_prob_tensor)
    out_layer = tf.matmul(dropout, weights['out']) + biases['out']

    #out_layer = tf.matmul(layer_3, weights['out']) + biases['out']

    return out_layer

#create a layer
def computation_layer(input, weights, biases):
    layer_i = tf.add(tf.matmul(input, weights), biases)
    layer_i = tf.nn.relu(layer_i)
    return layer_i

def l2regularization(weights, biases):

    regularizers = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(biases['b1']) + tf.nn.l2_loss(weights['h2']) + tf.nn.l2_loss(
        biases['b2']) + tf.nn.l2_loss(weights['h3']) + tf.nn.l2_loss(biases['b3']) + tf.nn.l2_loss(weights['h4']) + tf.nn.l2_loss(
        biases['b4']) + tf.nn.l2_loss(weights['h5']) + tf.nn.l2_loss(biases['b5']) + tf.nn.l2_loss(weights['h6']) + tf.nn.l2_loss(
        biases['b6']) + tf.nn.l2_loss(weights['h7']) + tf.nn.l2_loss(biases['b7'])

    return regularizers

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

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
    'h6': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_6])),
    'h7': tf.Variable(tf.random_normal([n_hidden_6, n_hidden_7])),
    'h8': tf.Variable(tf.random_normal([n_hidden_7, n_hidden_8])),
    'out': tf.Variable(tf.random_normal([n_hidden_8, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'b5': tf.Variable(tf.random_normal([n_hidden_5])),
    'b6': tf.Variable(tf.random_normal([n_hidden_6])),
    'b7': tf.Variable(tf.random_normal([n_hidden_7])),
    'b8': tf.Variable(tf.random_normal([n_hidden_8])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)
pred1 = tf.argmax(pred, 1)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
if hyperParam is not None:
    cost += l2regularization(weights, biases) * hyperParam

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

#Mix the input data
_x, _y = mixExamples(examples, labels, numFrames, numEx)

print("_x before: ", _x.shape)
print("_y before: ", _y.shape)

mini_bag = int(len(_x)*0.3)
full_bag = len(_x)
if useTest is 1:
    valid_x, valid_y = mixExamples(test_data, test_labels, numFrames, len(test_data))
    valid_x = test_data.reshape((test_data.shape[0], n_input))
    valid_y = test_labels

else:
    valid_data = _x[0:mini_bag]
    valid_labels = _y[0:mini_bag]

    indices = np.arange(int(mini_bag))
    _x = np.delete(_x, indices, 0)
    _y = np.delete(_y, indices, 0)

    valid_x, valid_y = mixExamples(valid_data, valid_labels, numFrames, len(valid_data))
    valid_x = valid_data.reshape((valid_data.shape[0], n_input))
    valid_y = valid_labels


numEx = len(_x)

print("_x after: ", _x.shape)
print("_y after: ", _y.shape)
print("valid_data: ", valid_x.shape)
print("valid_labels: ", valid_y.shape)


# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_iter):
        avg_cost = 0.
        total_batch = int(numEx/batch_size)#-3
        # Loop over all batches
        step=0
        for i in range(total_batch):
            index = i * batch_size

            batch_x, batch_y = _x[index:(index + batch_size)], _y[index:(index + batch_size)]
            # Run optimization op (backprop) and cost op (to get loss value)
            batch_x = batch_x.reshape((batch_x.shape[0], n_input))
            _, c, _pred_ = sess.run([optimizer, cost, pred1], feed_dict={x: batch_x, y: batch_y,
                                                                         keep_prob_tensor: keep_prob})
            acc = sess.run(accuracy, feed_dict={x: valid_x, y: valid_y, keep_prob_tensor: 1.0})
            '''
            correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            '''

            # Compute average loss
            avg_cost += c / total_batch
            #print("c ", c)
            #print("total_batch ", total_batch)
            #print("avg_cost ", avg_cost)

        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.5f}".format(avg_cost/1000000000), "*10⁹ accuracy=", "{:.5f}".format(acc))

        if epoch == training_iter:
            _y_ = np.argmax(batch_y, 1)

            confusion_matrix = metrics.confusion_matrix(_y_, _pred_, [0, 1])
            print("Last Training: Confusion Matrix\n", confusion_matrix)
            print("Last Training: Accuracy Score\n", metrics.accuracy_score(_y_, _pred_))
            print("Last Training: F1 Score\n", metrics.f1_score(_y_, _pred_))
            print("Last Training: Recall Score\n", metrics.recall_score(_y_, _pred_))
            print("Last Training: Precision Score\n", metrics.precision_score(_y_, _pred_))

    print ("Optimization Finished!\n\n")

    print("learning_rate: ", learning_rate, "batch_size: ",
          batch_size, "hyperParam: ", hyperParam, "keep_prob", keep_prob, "\n")

    print("Performance Metrics:\n")
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    '''
    _x, _y = mixExamples(test_data, test_labels, numFrames, len(test_data))
    _x = test_data.reshape((test_data.shape[0], n_input))
    _y = test_labels
    '''
    '''
    useTest = 0
    if useTest == 1:
        _x, _y = mixExamples(test_data, test_labels, numFrames, len(test_data))
        _x = test_data.reshape((test_data.shape[0], n_input))
        _y = test_labels
    else:
        _x, _y = mixExamples(valid_data, valid_labels, numFrames, len(valid_data))
        _x = valid_data.reshape((valid_data.shape[0], n_input))
        _y = valid_labels
    '''

    _y_ = np.argmax(valid_y, 1)

    acc1, _pred_ = sess.run([accuracy, pred1], feed_dict={x: valid_x, y: valid_y, keep_prob_tensor: 1.0})
    print("Accuracy:", acc1)


    confusion_matrix = metrics.confusion_matrix(_y_, _pred_, [0, 1])
    recall = metrics.recall_score(_y_, _pred_)
    specificity = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])
    uwAccuracy = (recall+specificity)/2

    print("Confusion Matrix\n", confusion_matrix)
    print("Accuracy Score\n", metrics.accuracy_score(_y_, _pred_))
    print("Unweighted Accuracy\n", uwAccuracy)
    print("F1 Score\n", metrics.f1_score(_y_, _pred_))
    print("Recall Score\n", recall)
    print("Precision Score\n", metrics.precision_score(_y_, _pred_))


