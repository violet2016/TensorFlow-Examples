""" Convolutional Neural Network.

Build and train a distributed convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Violet Cheng
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import json
import os
import sys
import tempfile
import time
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data


flags = tf.app.flags
flags.DEFINE_string("data_dir", "/tmp/mnist-data",
                    "Directory for storing mnist data")
flags.DEFINE_boolean("download_only", False,
                     "Only perform downloading of data; Do not proceed to "
                     "session preparation, model definition or training")
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_integer("num_gpus", 1, "Total number of gpus for each machine."
                     "If you don't use GPU, please set it to '0'")
flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
flags.DEFINE_integer("hidden_units", 100,
                     "Number of units in the hidden layer of the NN")
flags.DEFINE_integer("train_steps", 1000,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 128, "Training batch size")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_boolean(
    "sync_replicas", False,
    "Use the sync_replicas (synchronized replicas) mode, "
    "wherein the parameter updates from workers are aggregated "
    "before applied to avoid stale gradients")
flags.DEFINE_boolean(
    "existing_servers", False, "Whether servers already exists. If True, "
    "will use the worker hosts via their GRPC URLs (one client process "
    "per worker host). Otherwise, will create an in-process TensorFlow "
    "server.")
flags.DEFINE_string("ps_hosts", "localhost:2222",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None, "job name: worker or ps")

FLAGS = flags.FLAGS

# globals
IMAGE_PIXELS = 28

learning_rate = 0.001
training_steps = 200
batch_size = 128
display_step = 10


num_input = IMAGE_PIXELS * IMAGE_PIXELS
num_classes = 10
dropout = 0.75


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


def main(unused_argv):
    # Parse environment variable TF_CONFIG to get job_name and task_index

    # If not explicitly specified in the constructor and the TF_CONFIG
    # environment variable is present, load cluster_spec from TF_CONFIG.
    tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
    task_config = tf_config.get('task', {})
    task_type = task_config.get('type')
    task_index = task_config.get('index')

    FLAGS.job_name = task_type
    FLAGS.task_index = task_index

    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    if FLAGS.download_only:
        sys.exit(0)

    if FLAGS.job_name is None or FLAGS.job_name == "":
        raise ValueError("Must specify an explicit `job_name`")
    if FLAGS.task_index is None or FLAGS.task_index == "":
        raise ValueError("Must specify an explicit `task_index`")

    print("job name = %s" % FLAGS.job_name)
    print("task index = %d" % FLAGS.task_index)

    cluster_config = tf_config.get('cluster', {})
    ps_hosts = cluster_config.get('ps')
    worker_hosts = cluster_config.get('worker')

    ps_hosts_str = ','.join(ps_hosts)
    worker_hosts_str = ','.join(worker_hosts)

    FLAGS.ps_hosts = ps_hosts_str
    FLAGS.worker_hosts = worker_hosts_str

    # Construct the cluster and start the server
    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")

    # Get the number of workers.
    num_workers = len(worker_spec)

    cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})

    if not FLAGS.existing_servers:
        # Not using existing servers. Create an in-process server.
        server = tf.train.Server(
            cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
        if FLAGS.job_name == "ps":
            server.join()

    is_chief = (FLAGS.task_index == 0)
    if FLAGS.num_gpus > 0:
        # Avoid gpu allocation conflict: now allocate task_num -> #gpu
        # for each worker in the corresponding machine
        gpu = (FLAGS.task_index % FLAGS.num_gpus)
        worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)
    elif FLAGS.num_gpus == 0:
        # Just allocate the CPU to worker server
        cpu = 0
        worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, cpu)
    # The device setter will automatically place Variables ops on separate
    # parameter servers (ps). The non-Variable ops will be placed on the workers.
    # The ps use CPU and workers use corresponding GPU
    with tf.device(
        tf.train.replica_device_setter(
            worker_device=worker_device,
            ps_device="/job:ps/cpu:0",
            cluster=cluster)):
        global_step = tf.Variable(0, name="global_step", trainable=False)

        # Store layers weight & bias
        weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([1024, num_classes]))
        }

        biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([num_classes]))
        }
        # tf Graph input need in "with device"
        X = tf.placeholder(tf.float32, [None, num_input])
        Y = tf.placeholder(tf.float32, [None, num_classes])
        keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

        # Construct model
        logits = conv_net(X, weights, biases, keep_prob)
        prediction = tf.nn.softmax(logits)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y))
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

        if FLAGS.sync_replicas:
            if FLAGS.replicas_to_aggregate is None:
                replicas_to_aggregate = num_workers
            else:
                replicas_to_aggregate = FLAGS.replicas_to_aggregate

            opt = tf.train.SyncReplicasOptimizer(
                opt,
                replicas_to_aggregate=replicas_to_aggregate,
                total_num_replicas=num_workers,
                name="mnist_sync_replicas")

        train_op = opt.minimize(loss_op, global_step=global_step)

        # supervisor's ops
        if FLAGS.sync_replicas:
            local_init_op = opt.local_step_init_op
            if is_chief:
                local_init_op = opt.chief_init_op

            ready_for_local_init_op = opt.ready_for_local_init_op

            # Initial token and chief queue runners required by the sync_replicas mode
            chief_queue_runner = opt.get_chief_queue_runner()
            sync_init_op = opt.get_init_tokens_op()

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initialize the variables (i.e. assign their default value)
        init_op = tf.global_variables_initializer()
        train_dir = tempfile.mkdtemp()
        # Supervisor for distributed training
        if FLAGS.sync_replicas:
            sv = tf.train.Supervisor(
                is_chief=is_chief,
                logdir=train_dir,
                init_op=init_op,
                local_init_op=local_init_op,
                ready_for_local_init_op=ready_for_local_init_op,
                recovery_wait_secs=1,
                global_step=global_step)
        else:
            sv = tf.train.Supervisor(
                is_chief=is_chief,
                logdir=train_dir,
                init_op=init_op,
                recovery_wait_secs=1,
                global_step=global_step)

        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            device_filters=["/job:ps",
                            "/job:worker/task:%d" % FLAGS.task_index])

        # The chief worker (task_index==0) session will prepare the session,
        # while the remaining workers will wait for the preparation to complete.
        if is_chief:
            print("Worker %d: Initializing session..." % FLAGS.task_index)
        else:
            print("Worker %d: Waiting for session to be initialized..." %
                  FLAGS.task_index)

        if FLAGS.existing_servers:
            server_grpc_url = "grpc://" + worker_spec[FLAGS.task_index]
            print("Using existing server at: %s" % server_grpc_url)

            sess = sv.prepare_or_wait_for_session(
                server_grpc_url, config=sess_config)
        else:
            sess = sv.prepare_or_wait_for_session(
                server.target, config=sess_config)

        print("Worker %d: Session initialization complete." % FLAGS.task_index)

        if FLAGS.sync_replicas and is_chief:
            # Chief worker will start the chief queue runner and call the init op.
            sess.run(sync_init_op)
            sv.start_queue_runners(sess, [chief_queue_runner])

        # Perform training
        time_begin = time.time()
        print("Training begins @ %f" % time_begin)

        local_step = 0
        while True:
            # Training feed
            # put your data and train op here
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, step = sess.run([train_op, global_step],
                               feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.8})
            local_step += 1

            now = time.time()
            if local_step % display_step == 0:
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_xs,
                                                                     Y: batch_ys, keep_prob: 1.0})

                print("%f: Worker %d: training step %d done (global step: %d) Minibatch Loss=%.4f Accuracy=%.3f" %
                      (now, FLAGS.task_index, local_step, step, loss, acc))

            if step >= FLAGS.train_steps:
                break

        time_end = time.time()
        print("Training ends @ %f" % time_end)
        training_time = time_end - time_begin
        print("Training elapsed time: %f s" % training_time)

        # Validation feed
        test_len = 128
        test_data = mnist.test.images[:test_len]
        test_label = mnist.test.labels[:test_len]
        print("Testing Accuracy:",
              sess.run(accuracy, feed_dict={X: test_data, Y: test_label, keep_prob: 1.0}))


if __name__ == "__main__":
    tf.app.run()
