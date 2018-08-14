from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import tempfile
import time
import os
import json

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tflearn.layers.core import fully_connected
from tensorflow.contrib import rnn

flags = tf.app.flags
flags.DEFINE_string("data_dir", "./mnist-data",
                    "Directory for storing mnist data")
flags.DEFINE_string("job_name", None, "job name: worker or ps")
flags.DEFINE_boolean("download_only", False,
                     "Only perform downloading of data; Do not proceed to "
                     "session preparation, model definition or training")
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")

FLAGS = flags.FLAGS
num_epochs = 1000

learning_rate = 0.001
training_steps = 10000
batch_size = 128
# display_step = 200


num_input = 28
# rnn_size = 100
timesteps = 28
num_hidden = 128
num_classes = 10
num_tasks = 0


def getConfig():
    tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')

    FLAGS.job_name = tf_config.get('job_name')
    FLAGS.task_index = tf_config.get('task_index')
    return tf_config


def makeCluster():
    global num_tasks
    tf_config = getConfig()

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
    num_tasks = len(worker_spec)

    cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})

    if not FLAGS.existing_servers:
        # Not using existing servers. Create an in-process server.
        server = tf.train.Server(
            cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
        if FLAGS.job_name == "ps":
            server.join()
    return cluster


def main(unused_argv):
    cluster = makeCluster()
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    if FLAGS.download_only:
        sys.exit(0)
    if FLAGS.job_name is None or FLAGS.job_name == "":
        raise ValueError("Must specify an explicit `job_name`")
    if FLAGS.task_index is None or FLAGS.task_index == "":
        raise ValueError("Must specify an explicit `task_index`")
    task_index = FLAGS.task_index

    if FLAGS.job_name == 'ps':
        # Create and start a server for the local task.
        server = tf.train.Server(cluster,
                                 job_name="ps",
                                 task_index=task_index)
        server.join()
    else:
        server = tf.train.Server(cluster,
                                 job_name="worker",
                                 task_index=task_index)

        is_chief = (task_index == 0)
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % task_index,
                cluster=cluster)):

            global_step = tf.Variable(0, name="global_step", trainable=False)

            # tf Graph input
            img = tf.placeholder(
                tf.float32, [batch_size, timesteps, num_input])
            # rnn_size])
            labels = tf.placeholder(
                tf.float32, [batch_size, timesteps, num_input])

            num_layers = 5

            # MultiRNNCell with dropout
            stacked_lstm = rnn.MultiRNNCell(
                [
                    tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(
                        num_hidden, forget_bias=1.0), output_keep_prob=0.5)
                    for _ in range(num_layers)
                ])
            # dynamic_rnn output format is different from static_rnn
            outputs, _ = tf.nn.dynamic_rnn(
                stacked_lstm, img, dtype=tf.float32, time_major=False)

            x = tf.unstack(outputs, axis=1)
            x = [fully_connected(x[i], num_input, activation='linear')
                 for i in range(timesteps)]
            x = tf.stack(x)
            pred = tf.transpose(x, [1, 0, 2])

            # Define loss and optimizer
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels))
            opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

            replicas_to_aggregate = num_tasks

            opt = tf.train.SyncReplicasOptimizer(
                opt,
                replicas_to_aggregate=replicas_to_aggregate,
                total_num_replicas=num_tasks,
                name="mnist_sync_replicas")

            train_step = opt.minimize(loss, global_step=global_step)

            local_init_op = opt.local_step_init_op
            if is_chief:
                local_init_op = opt.chief_init_op

            ready_for_local_init_op = opt.ready_for_local_init_op

            # Initial token and chief queue runners required by the sync_replicas mode
            chief_queue_runner = opt.get_chief_queue_runner()
            sync_init_op = opt.get_init_tokens_op()

            init_op = tf.global_variables_initializer()
            train_dir = tempfile.mkdtemp()

            sv = tf.train.Supervisor(
                is_chief=is_chief,
                logdir=train_dir,
                init_op=init_op,
                local_init_op=local_init_op,
                ready_for_local_init_op=ready_for_local_init_op,
                recovery_wait_secs=1,
                global_step=global_step)

            sess_config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
                device_filters=["/job:ps", "/job:worker/task:%d" % task_index])

            # The chief worker (task_index==0) session will prepare the session,
            # while the remaining workers will wait for the preparation to complete.
            if is_chief:
                print("Worker %d: Initializing session..." % task_index)
            else:
                print("Worker %d: Waiting for session to be initialized..." %
                      task_index)

            sess = sv.prepare_or_wait_for_session(
                server.target, config=sess_config)

            print("Worker %d: Session initialization complete." % task_index)

            if is_chief:
                # Chief worker will start the chief queue runner and call the init op.
                sess.run(sync_init_op)
                sv.start_queue_runners(sess, [chief_queue_runner])

            # Perform training
            time_begin = time.time()
            print("Training begins @ %f" % time_begin)

            local_step = 0
            while True:
                # Training feed
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                train_feed = {img: batch_xs, labels: batch_ys}

                _, step = sess.run(
                    [train_step, global_step], feed_dict=train_feed)
                local_step += 1

                now = time.time()
                print("%f: Worker %d: training step %d done (global step: %d)" %
                      (now, task_index, local_step, step))

                if step >= num_epochs:
                    break

            time_end = time.time()
            print("Training ends @ %f" % time_end)
            training_time = time_end - time_begin
            print("Training elapsed time: %f s" % training_time)

            # Validation feed
            # val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
            # val_xent = sess.run(loss, feed_dict=val_feed)
            # print("After %d training step(s), validation cross entropy = %g" %
            #      (num_epochs, val_xent))


if __name__ == "__main__":
    tf.app.run()
