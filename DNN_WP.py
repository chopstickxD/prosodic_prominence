'''
A Multilayer Perceptron implementation example using TensorFlow library.
based on 'Author: Aymeric Damien Project: https://github.com/aymericdamien/TensorFlow-Examples/'

Now its a DNN wit 6 hidden_layers and encodes binary numbers from 1-128
and edited by Tan
'''


import tensorflow as tf
import numpy as np
import preprocessing as pre
import sklearn.metrics as metrics

# Parameters
learning_rate = 0.005
display_step = 1
batch_size = 120
hyperParam = 0.2
training_iter = 25

# Network Parameters
n_hidden_1 = 512 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_hidden_3 = 128  # 2nd layer number of features
n_hidden_4 = 64  # 2nd layer number of features
n_hidden_5 = 32  # 2nd layer number of features
n_hidden_6 = 18 # 2nd layer number of features
n_input = 13*29      # data input (7bits)
n_classes = 2  # total classes (0-128 digits)
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
numEx = len(examples)



# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


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
    # Output layer with linear activation
    out_layer = tf.matmul(layer_6, weights['out']) + biases['out']
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
        biases['b6'])

    return regularizers

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

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
    'h6': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_6])),
    'out': tf.Variable(tf.random_normal([n_hidden_6, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'b5': tf.Variable(tf.random_normal([n_hidden_5])),
    'b6': tf.Variable(tf.random_normal([n_hidden_6])),
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

# Initializing the variables
init = tf.initialize_all_variables()

#Mix the input data
_x, _y = mixExamples(examples, labels)

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_iter):
        avg_cost = 0.
        total_batch = int(numEx/batch_size)+1
        # Loop over all batches
        for i in range(total_batch):
            index = i * batch_size
            batch_x, batch_y = _x[index:(index + batch_size)], _y[index:(index + batch_size)]
            #print('y ',batch_y.shape)
            #print('x ',batch_x.shape)
            # Run optimization op (backprop) and cost op (to get loss value)
            batch_x = batch_x.reshape((batch_x.shape[0], n_input))
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                             y: batch_y})
            '''
            correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            '''

            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost)) #, " accuracy= ", "{:.9f}".format(acc))
    print ("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    _x = test_data.reshape((test_data.shape[0], n_input))
    _y = test_labels
    print(_y.shape)
    acc1, _pred_ = sess.run([accuracy, pred1], feed_dict={x: _x, y: _y})
    print("Accuracy:", acc1)

    _y_ = np.argmax(_y, 1)
    print("Confusion Matri\n", metrics.confusion_matrix(_y_, _pred_, [0, 1]))
    print("Accuracy Score\n", metrics.accuracy_score(_y_, _pred_))
    print("F1 Score\n", metrics.f1_score(_y_, _pred_))
    print("Recall Score\n", metrics.recall_score(_y_, _pred_))
    print("Precision Score\n", metrics.precision_score(_y_, _pred_))

