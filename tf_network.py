import tensorflow as tf
import numpy as np
import GenerateData as data
import tf_DNN_model as net
import time


'''
Getting the Data which are
numbers - binaryNumbers from 0-127
labels - labels as one hot-vectors
'''
numbers, labels = data.genData(7)
#labels = np.zeros(128)
#for i in range(128):
#    labels[i]=i
'''
Some other constants
'''
HIDDEN1 = 256
HIDDEN2 = 64
HIDDEN3 = 16
HIDDEN4 = 64
HIDDEN5 = 256
LEARNING_RATE = 0.01
'''
Define the tensor placeholders fot input and labels
'''
def placeholder_inputs():
    x = tf.placeholder(tf.float32, [128, 7]) #placeholder for input
    y_ = tf.placeholder(tf.int32, [128, 128]) #placeholder for output
    return x, y_

'''
Fills the feed_dict for training the given step.
'''
def fill_feed_dict(data_pl, labels_pl):
    feed_dict = {
        data_pl: numbers,
        labels_pl: labels,
    }
    return feed_dict


'''
Evaluation from Tutorial
'''
def do_eval(sess, eval_correct, num_placeholder, labels_placeholder):


    """Runs one evaluation against the full epoch of data.
    Args:
      sess: The session in which the model has been trained.
      eval_correct: The Tensor that returns the number of correct predictions.
      images_placeholder: The images placeholder.
      labels_placeholder: The labels placeholder.
      data_set: The set of images and labels to evaluate, from
        input_data.read_data_sets().
    """

    true_count = 0
    numData = numbers.shape[0]
    for step in range(numData):
        feed_dict = fill_feed_dict(num_placeholder,
                                   labels_placeholder)
        true_count = sess.run(eval_correct, feed_dict=feed_dict)
    precision = true_count / numData
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (numData, true_count, precision))



def run_training():
  # Get the binary numbers and labels
  numbers, labels = data.genData(7)

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    num_placeholder, labels_placeholder = placeholder_inputs()

    # Build a Graph that computes predictions from the inference model.
    logits = net.inference(num_placeholder,
                             HIDDEN1,
                             HIDDEN2,
                             HIDDEN3,
                             HIDDEN4,
                             HIDDEN5,)

    # Add to the Graph the Ops for loss calculation.
    loss = net.loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = net.training(loss, LEARNING_RATE)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = net.evaluation(logits, labels_placeholder)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Add the variable initializer Op.
    init = tf.initialize_all_variables()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.train.SummaryWriter('data', sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)

    # Start the training loop.
    for step in range(128):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = fill_feed_dict(num_placeholder,
                                 labels_placeholder)

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 100 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
      #if (step + 1) % 1000 == 0 or (step + 1) == 2000:

        saver.save(sess, 'data', global_step=step)
        # Evaluate against the training set.
        print('Training Data Eval:')
        do_eval(sess,
                eval_correct,
                num_placeholder,
                labels_placeholder)
        '''
        # Evaluate against the validation set.
        print('Validation Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.validation)
        # Evaluate against the test set.
        print('Test Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.test)
        '''
def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()