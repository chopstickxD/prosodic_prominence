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


#path to data and labels
pathData1 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/CNE/*16k.wav'
pathData2 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/GEN/*16k.wav'
pathData3 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/HRO/*16k.wav'
pathData4 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/ICO/*16k.wav'
pathData5 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/KTA/*16k.wav'
pathData6 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/LHE/*16k.wav'
pathData7 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/MDU/*16k.wav'
pathData8 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/SCO/*16k.wav'

pathLabel1 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_CNE.csv'
pathLabel2 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_GEN.csv'
pathLabel3 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_HRO.csv'
pathLabel4 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_ICO.csv'
pathLabel5 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_KTA.csv'
pathLabel6 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_LHE.csv'
pathLabel7 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_MDU.csv'
pathLabel8 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_SCO.csv'

pathTimes1 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/StartEndTime_as_csv/startEnd_time_CNE.csv'
pathTimes2 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/StartEndTime_as_csv/startEnd_time_GEN.csv'
pathTimes3 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/StartEndTime_as_csv/startEnd_time_HRO.csv'
pathTimes4 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/StartEndTime_as_csv/startEnd_time_ICO.csv'
pathTimes5 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/StartEndTime_as_csv/startEnd_time_KTA.csv'
pathTimes6 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/StartEndTime_as_csv/startEnd_time_LHE.csv'
pathTimes7 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/StartEndTime_as_csv/startEnd_time_MDU.csv'
pathTimes8 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/StartEndTime_as_csv/startEnd_time_SCO.csv'

#parameters for preprocessing
sampleRate = 16000
maxSentenceStep = 9*sampleRate
maxWordStep = int(0.84*sampleRate)

#reading the data and preprocessing
examples1, labels1 = pre.makeInputSeq(pathData1, pathTimes1, pathLabel1, maxSentenceStep, maxWordStep, sampleRate)
examples2, labels2 = pre.makeInputSeq(pathData2, pathTimes2, pathLabel2, maxSentenceStep, maxWordStep, sampleRate)
examples3, labels3 = pre.makeInputSeq(pathData3, pathTimes3, pathLabel3, maxSentenceStep, maxWordStep, sampleRate)
examples4, labels4 = pre.makeInputSeq(pathData4, pathTimes4, pathLabel4, maxSentenceStep, maxWordStep, sampleRate)
examples5, labels5 = pre.makeInputSeq(pathData5, pathTimes5, pathLabel5, maxSentenceStep, maxWordStep, sampleRate)
examples6, labels6 = pre.makeInputSeq(pathData6, pathTimes6, pathLabel6, maxSentenceStep, maxWordStep, sampleRate)
examples7, labels7 = pre.makeInputSeq(pathData7, pathTimes7, pathLabel7, maxSentenceStep, maxWordStep, sampleRate)

test_data, test_labels = pre.makeInputSeq(pathData8, pathTimes8, pathLabel8, maxSentenceStep, maxWordStep, sampleRate)


examples = np.concatenate((examples1, examples2, examples3, examples4, examples5, examples6, examples7))
labels = np.concatenate((labels1, labels2, labels3, labels4, labels5, labels6, labels7))

numEx = len(examples)
# Parameters
learning_rate = 0.1
display_step = 1
batch_size = 150
hyperParam = 0.2
training_iter = 75
numFrames = examples.shape[1] #frames of the mfcc

# Network Parameters
n_hidden_1 = 512 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_hidden_3 = 128  # 3rd layer number of features
n_hidden_4 = 64 # 4th layer number of features
n_hidden_5 = 32  # 5th layer number of features
n_hidden_6 = 18 # 6th layer number of features
n_hidden_7 = 6 # 6th layer number of features
n_input = 13*numFrames      # data input (13mfcc*numframes)
n_classes = 2  # total classes (0-128 digits)



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
    layer_7 = computation_layer(layer_6, weights['h7'], biases['b7'])
    # Output layer with linear activation
    out_layer = tf.matmul(layer_7, weights['out']) + biases['out']
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
        biases['b6'] + tf.nn.l2_loss(weights['h7']) + tf.nn.l2_loss(biases['b7']))

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
    'out': tf.Variable(tf.random_normal([n_hidden_7, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'b5': tf.Variable(tf.random_normal([n_hidden_5])),
    'b6': tf.Variable(tf.random_normal([n_hidden_6])),
    'b7': tf.Variable(tf.random_normal([n_hidden_7])),
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
_x, _y = mixExamples(examples, labels, numFrames, numEx)



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

    _x, _y = mixExamples(test_data, test_labels, numFrames, len(test_data))
    _x = test_data.reshape((test_data.shape[0], n_input))
    _y = test_labels
    _y_ = np.argmax(_y, 1)

    acc1, _pred_ = sess.run([accuracy, pred1], feed_dict={x: _x, y: _y})
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


