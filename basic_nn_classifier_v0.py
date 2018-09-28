import tensorflow as tf
import numpy as np
import os
import datetime
import argparse
import sys
from sklearn.model_selection import KFold

# read the argv
argv = sys.argv

#x_file = "mini_matrix.txt"
x_file = argv[1]

#y_file = "mini_label.txt"
y_file = argv[2]

# fold 
fold = int(argv[3])

# learning rate
learning_rate = float(argv[4])

# Parameters
input_dim = 345901
n_l1 = 100
n_l2 = 100
batch_size = 500
n_epochs = 1000
beta1 = 0.9
z_dim = 'NA'
results_path = './Results/Basic_NN_Classifier'
n_labels = 2


# Get data
x_data = np.loadtxt(x_file, delimiter = "\t")
x_data = x_data[:, 6:]
y_data = np.loadtxt(y_file, delimiter = "\t")
y_data = np.column_stack((y_data, 1 - y_data))

# spliting
kf = KFold(n_splits = 5, random_state = 1)
kf.get_n_splits(x_data)
k_list = []
for train_index, test_index in kf.split(x_data):
    k_list.append([train_index, test_index])    
cur_index = k_list[fold - 1]

# training data
x_train = x_data[cur_index[0], :]
y_train = y_data[cur_index[0], :]

# testing data
x_test = x_data[cur_index[1], :]
y_test = y_data[cur_index[1], :]

print("shape:\n")
print(x_train.shape[0])
n_labeled = x_train.shape[0]

print("INFO: Reading complete.\n")
# Placeholders
x_input = tf.placeholder(dtype=tf.float32, shape=[None, input_dim])
y_target = tf.placeholder(dtype=tf.float32, shape=[None, 2])

def form_results():
    """
    Forms folders for each run to store the tensorboard files, saved models and the log files.
    :return: three string pointing to tensorboard, saved models and log paths respectively.
    """
    folder_name = "/{0}_{1}_{2}_{3}_{4}_{5}_kfold{6}_Basic_NN_Classifier". \
        format(datetime.datetime.now(), z_dim, learning_rate, batch_size, n_epochs, beta1, fold)
    tensorboard_path = results_path + folder_name + '/Tensorboard'
    saved_model_path = results_path + folder_name + '/Saved_models/'
    log_path = results_path + folder_name + '/log'
    if not os.path.exists(results_path + folder_name):
        os.mkdir(results_path + folder_name)
        os.mkdir(tensorboard_path)
        os.mkdir(saved_model_path)
        os.mkdir(log_path)
    return tensorboard_path, saved_model_path, log_path


def next_batch(x, y, batch_size):
    """
    Used to return a random batch from the given inputs.
    :param x: Input vector of shape [None, 345901]
    :param y: Input labels of shape [None, 2]
    :param batch_size: integer, batch size of images and labels to return
    :return: x -> [batch_size, 345901], y-> [batch_size, 2]
    """
    index = np.arange(n_labeled)
    random_index = np.random.permutation(index)[:batch_size]
    return x[random_index], y[random_index]


def dense(x, n1, n2, name):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param n1: no. of input neurons
    :param n2: no. of output neurons
    :param name: name of the entire dense layer.
    :return: tensor with shape [batch_size, n2]
    """
    with tf.name_scope(name):
        weights = tf.Variable(tf.random_normal(shape=[n1, n2], mean=0., stddev=0.01), name='weights')
        bias = tf.Variable(tf.zeros(shape=[n2]), name='bias')
        output = tf.add(tf.matmul(x, weights), bias, name='output')
        return output


# Dense Network
def dense_nn(x):
    """
    Network used to classify MNIST digits.
    :param x: tensor with shape [batch_size, 784], input to the dense fully connected layer.
    :return: [batch_size, 10], logits of dense fully connected.
    """
    dense_1 = tf.nn.dropout(tf.nn.relu(dense(x, input_dim, n_l1, 'dense_1')), keep_prob=0.75)
    dense_2 = tf.nn.dropout(tf.nn.relu(dense(dense_1, n_l1, n_l2, 'dense_2')), keep_prob=0.75)
    dense_3 = dense(dense_2, n_l2, n_labels, 'dense_3')
    return dense_3


def train():
    """
    Used to train the autoencoder by passing in the necessary inputs.
    :return: does not return anything
    """
    dense_output = dense_nn(x_input)
    # Loss function
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dense_output, labels=y_target))
    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(loss)
    # Accuracy
    pred_op = tf.equal(tf.argmax(dense_output, 1), tf.argmax(y_target, 1))
    accuracy = tf.reduce_mean(tf.cast(pred_op, dtype=tf.float32))
    # Summary
    tf.summary.scalar(name='Loss', tensor=loss)
    tf.summary.scalar(name='Accuracy', tensor=accuracy)
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    step = 0
    with tf.Session() as sess:
        tensorboard_path, saved_model_path, log_path = form_results()
        #x_l, y_l = mnist.test.next_batch(n_labeled)
        writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=sess.graph)
        sess.run(init)
        for e in range(1, n_epochs + 1):
            n_batches = int(n_labeled / batch_size)
            #print(n_batches)
            for b in range(1, n_batches + 1):
                batch_x_l, batch_y_l = next_batch(x_train, y_train, batch_size=batch_size)
                #print("shape:")
                #print(batch_x_l.shape)
                sess.run(optimizer, feed_dict={x_input: batch_x_l, y_target: batch_y_l})
                if b % 5 == 0:
                    print(batch_x_l.shape)
                    loss_, summary = sess.run([loss, summary_op], feed_dict={x_input: batch_x_l, y_target: batch_y_l})
                    writer.add_summary(summary, step)
                    print("Epoch: {} Iteration: {}".format(e, b))
                    print("Loss: {}".format(loss_))
                    with open(log_path + '/log.txt', 'a') as log:
                        log.write("Epoch: {}, iteration: {}\n".format(e, b))
                        log.write("Loss: {}\n".format(loss_))
                step += 1
            acc = 0
            #num_batches = int(100/ batch_size)
            #for j in range(num_batches):
            # Classify unseen validation data instead of test data or train data
            #batch_x_l, batch_y_l = mnist.validation.next_batch(batch_size=batch_size)
            val_acc = sess.run(accuracy, feed_dict={x_input: x_test, y_target: y_test})
            #acc += val_acc
            #acc /= num_batches
            acc = val_acc
            print("Classification Accuracy: {}".format(acc))
            with open(log_path + '/log.txt', 'a') as log:
                log.write("Classification Accuracy: {} \n".format(acc))
            saver.save(sess, save_path=saved_model_path, global_step=step)
            
if __name__ == '__main__':    
    train()
