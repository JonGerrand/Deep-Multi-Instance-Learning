"""
    @Description:       This module contain the code responsible for
                        implementing both stage-1 and stage-2 traingin for
                        the MIL method.

    @Author:            Jonathan Gerrand
"""

import tensorflow as tf
import os.path as osp
import datetime
import models
import numpy as np
from mil_testing import ResultProcessor

# Index for Salient Class
SCREEN_INDX=1

class EarlyStoppingSupervisor(object):
    def __init__(self, stop_type='loss'):
        if stop_type == 'score':
            self.min_validation = 0
        if stop_type == 'loss':
            self.min_validation = 100
        self.stop_type = stop_type
        self.current_val_run = 0
        self.early_stop_count = 20
        self.start_delay = 30

    def process_validation_result(self, step, validation_score):
        if self.stop_type == 'score':
            if validation_score > self.min_validation \
            and step >= self.start_delay:
                #  New checkpoint reached
                self.current_val_run = 0
                self.min_validation = validation_score
                return False
            else:
                self.current_val_run += 1
                if self.current_val_run > self.early_stop_count:
                    # Early stopping criterion met
                    return True
                else:
                # Continue training
                    return None
        if self.stop_type == 'loss':
            if validation_score < self.min_validation \
            and step >= self.start_delay:
                self.current_val_run = 0
                self.min_validation = validation_score
                return False
            else:
                self.current_val_run += 1
                if self.current_val_run > self.early_stop_count:
                    return True
                else:
                    return None

def factor_training_layers(model_spec):
    """
    Separates model layers into those which need to be re-initialised
    and those which will be fine-tuned.
    Args:
        model_spec: DataSpec Object. The spec of the model being trained.
                    See 'models/helper.py'

    Returns:
        1-D list of strings - re-initialised layer names
        1-D list of strings - fine-tuned layer names
    """
    re_init_layers = []
    fine_tune_layers = []
    for v in tf.trainable_variables():
        for layer in model_spec.re_initilised_layers:
            if v.name.lower().find(layer) != -1:
                re_init_layers.append(v)
            else:
                fine_tune_layers.append(v)

    return re_init_layers, fine_tune_layers

def setup_placeholders(data_spec):
    train_lab_placehldr = tf.placeholder(tf.float32, shape=(None,
                                         data_spec.num_classes),
                                         name='Training_lab_placehldr')
    val_lab_placehldr = tf.placeholder(tf.float32, shape=(None,
                                       data_spec.num_classes),
                                       name='Validation_lab_placehldr')
    loss_placehldr = tf.placeholder(tf.float32)

    return train_lab_placehldr, val_lab_placehldr, loss_placehldr

def setup_model_params(model_name,
                       train_spec,
                       data_spec):
    img_placehldr = tf.placeholder(tf.float32, shape=(None,
                                    train_spec.model_img_len,
                                    train_spec.model_img_len,
                                    3), name='Training_img_placehldr')
    # Define network for training
    net = models.load_model_with_data(model_name,
                                      img_placehldr,
                                      num_classes=data_spec.num_classes)
    logits = net.layers['logits']

    return net, logits, img_placehldr


def create_training_ops(logits,
                        train_labels,
                        val_labels,
                        loss_placehldr,
                        train_spec,
                        model_spec,
                        dataset_spec):
    """
    Construct the loss, backprop-update and evaluation operations of the
    graph
    """
    # Create Global step counter - This can be restored and as such the
    # 'get_variable' function is used
    with tf.variable_scope("global"):
        global_step = tf.get_variable('global_step',
                                      [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)

    num_examples_per_epoch = (dataset_spec.num_datapoints()/train_spec.batch_size)
    decay_steps = int(num_examples_per_epoch * train_spec.num_epochs_per_decay)

    lr_reinit = tf.train.exponential_decay(train_spec.reinit_learn_rate,
                               global_step=global_step,
                               decay_steps=decay_steps,
                               decay_rate=train_spec.learn_rate_decay_factor,
                               staircase=True)

    lr_fine_tune = tf.train.exponential_decay(train_spec.fine_tune_learn_rate,
                                  global_step=global_step,
                                  decay_steps=decay_steps,
                                  decay_rate=train_spec.learn_rate_decay_factor,
                                  staircase=True)

    # Configure optimisers
    if train_spec.optimiser == 'Adam':
        opt_reinit = tf.train.AdamOptimizer(lr_reinit,
                                            beta1=train_spec.adam_beta1,
                                            beta2=train_spec.adam_beta2)

        opt_fine_tune = tf.train.AdamOptimizer(lr_fine_tune,
                                            beta1=train_spec.adam_beta1,
                                            beta2=train_spec.adam_beta2)

    if train_spec.optimiser == 'RMSProp':
        opt_reinit = tf.train.RMSPropOptimizer(lr_reinit,
                                            decay=train_spec.rmsprop_decay,
                                            momentum=train_spec.rmsprop_momentum)

        opt_fine_tune = tf.train.RMSPropOptimizer(lr_fine_tune,
                                            decay=train_spec.rmsprop_decay,
                                            momentum=train_spec.rmsprop_momentum)

    if train_spec.optimiser == 'SGD':
        opt_reinit = tf.train.MomentumOptimizer(lr_reinit,
                                            momentum=train_spec.sgd_momentum)

        opt_fine_tune = tf.train.MomentumOptimizer(lr_fine_tune,
                                            momentum=train_spec.sgd_momentum)

    # Configure training loss
    if train_spec.training_type == 'stage_1':
        # Running index required to gather max predictions during TF graph
        # computation. --> [0,1,..,batch_size]
        batch_t_index = tf.constant([x for x in xrange(train_spec.batch_size)],
                                  dtype=tf.int64)
        # Required for bag-wise comparison.
        # Resulting shape -> (#batch, #crops, #classes)
        reshape_t_logits = tf.reshape(logits,shape=[train_spec.batch_size,
                                                  train_spec.img_bag_count,
                                                  -1])
        # We need the absolute probabilities for softmax (GM) comparison
        soft_t_logits = tf.nn.softmax(reshape_t_logits)
        # Get the resulting max probability
        max_t_predict = tf.argmax(soft_t_logits,axis=1)
        # Create a composite index to gather max indicies
        max_t_indx = tf.stack([batch_t_index,max_t_predict[:,SCREEN_INDX]],
                            axis=1)
        # Collect max logits
        max_t_logits = tf.gather_nd(reshape_t_logits, max_t_indx)
        # Compute softmax cross entropy and average over batch
        train_logit_loss = tf.nn.softmax_cross_entropy_with_logits(
                                logits=max_t_logits, labels=train_labels)
        train_loss = tf.reduce_mean(train_logit_loss)

    if train_spec.training_type == 'stage_2':
        train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                    logits=logits,labels=train_labels))

    # Add weight decay to loss term
    if train_spec.weight_decay > 0:
        loss_vars = tf.trainable_variables()
        loss_weights = []
        for var in loss_vars:
            if var.name.lower().find('weights') != -1:
                loss_weights.append(var)

        train_loss = (train_loss + \
                     tf.add_n([tf.nn.l2_loss(v) for v in loss_weights]) *
                     train_spec.weight_decay)


    # Configure Validation metric
    if train_spec.training_type == 'stage_1' \
    and train_spec.val_metric == 'loss':

        # Tensorflow doesn't support dynamic execution as of yet,
        # therefore we create a normal and residual size val_op
        val_op = []

        # Required for bag-wise comparison.
        # Resulting shape -> (#batch, #crops, #classes)
        reshape_logits = tf.reshape(logits,shape=[-1,
                                                  train_spec.img_bag_count,
                                                  dataset_spec.num_classes])
        # We need the absolute probabilities for softmax (GM) comparison
        soft_logits = tf.nn.softmax(reshape_logits)
        # Get the resulting max probability (For class 1) for each
        max_predict = tf.argmax(soft_logits,axis=1)

        # --Normal Size--
        # Running index required to gather max predictions during TF graph
        # computation. --> [0,1,..,batch_size]
        batch_index_n = tf.constant([x for x in xrange(train_spec.batch_size)],
                                  dtype=tf.int64)
        # Create a composite index to gather max indicies
        max_indx_n = tf.stack([batch_index_n,max_predict[:,SCREEN_INDX]],
                              axis=1)
        # Collect max logits
        max_logits_n = tf.gather_nd(reshape_logits, max_indx_n)
        # Compute softmax cross entropy and average over batch
        val_logit_loss_n = tf.nn.softmax_cross_entropy_with_logits(
                                logits=max_logits_n, labels=val_labels)
        val_op.append(tf.reduce_mean(val_logit_loss_n))

        # --Residual Size--
        batch_index_r = tf.constant(
            [x for x in xrange(dataset_spec.num_val%train_spec.batch_size)],
            dtype=tf.int64)
        max_indx_r = tf.stack([batch_index_r,max_predict[:,SCREEN_INDX]],
                              axis=1)
        max_logits_r = tf.gather_nd(reshape_logits, max_indx_r)
        val_logit_loss_r = tf.nn.softmax_cross_entropy_with_logits(
                                logits=max_logits_r, labels=val_labels)
        val_op.append(tf.reduce_mean(val_logit_loss_r))

    if train_spec.val_metric == 'score':
        softmax_val = tf.nn.softmax(logits)
        val_op = tf.nn.top_k(softmax_val,
                             k=dataset_spec.num_classes,
                             sorted=True)

    # Configure optimization steps
    train_op_reinit = opt_reinit.minimize(train_loss,
                                          var_list=model_spec.reinit_params,
                                          global_step=global_step)
    train_op_fine_tune = opt_fine_tune.minimize(train_loss,
                                          var_list=model_spec.fine_tune_params,
                                          global_step=global_step)

    train_op = tf.group(train_op_reinit, train_op_fine_tune)

    # Capture training loss for a model
    tf.summary.scalar('training/Training_loss', train_loss)

    return train_loss, val_op, train_op


def setup_train_summary_writers(summary_dir,
                                session):
    """
    Initilise variable watchers for training.
    Args:
        summary_dir: Directory where logfiles will be stored
        session:
    Returns:

    """

    all_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(summary_dir + '/train',
                                         session.graph)
    return all_summaries, train_writer

def get_val_error(session,
                  val_op,
                  img_data,
                  img_placehldr,
                  label_data,
                  label_placehldr,
                  dropout_placehldr,
                  train_spec,
                  data_spec,
                  val_calc):
     print ("Performing validation step..")
     val_calc.flush_vals()
     val_total = 0.0
     val_counter = 0
     res_op = 0
     batch_size = train_spec.batch_size
     for begin in xrange(0, data_spec.num_val, train_spec.batch_size):
         val_counter += 1
         end = begin + train_spec.batch_size

         if end > data_spec.num_val:
             end = data_spec.num_val
             batch_size = end - begin
             res_op = 1

         temp_img_bag = img_data[begin:end]
         temp_lab_bag = label_data[begin:end]

         if train_spec.val_metric == 'loss':
             # Reshape batches for model insertion
             temp_img_bag = np.reshape(temp_img_bag,
                newshape=(batch_size*train_spec.img_bag_count,
                       train_spec.model_img_len,
                       train_spec.model_img_len,3))
             temp_img_bag = np.float32(temp_img_bag)
             temp_lab_bag = np.float32(temp_lab_bag)

             feed = {img_placehldr:temp_img_bag,
                     label_placehldr: temp_lab_bag,
                     dropout_placehldr:False}
             # Compute validation loss over a single batch
             val_loss = session.run([val_op[res_op]], feed_dict=feed)
             print (val_loss)
             val_total += val_loss[0]

         if train_spec.val_metric == 'score':
             temp_img_bag = np.reshape(temp_img_bag,
              newshape=(batch_size*train_spec.img_bag_count,
                        train_spec.model_img_len,
                        train_spec.model_img_len,3))
             temp_img_bag = np.float32(temp_img_bag)

             feed = {img_placehldr:temp_img_bag,
                     dropout_placehldr:False}

             # Compute validation loss over a single batch
             logit_values, logit_indices = session.run(val_op,
                                                 feed_dict=feed)

             # Reshape for calc ingestion
             logit_values = np.reshape(logit_values,
                                       newshape=(-1, train_spec.img_bag_count,
                                       data_spec.num_classes))
             logit_indices = np.reshape(logit_indices,
                                       newshape=(-1, train_spec.img_bag_count,
                                       data_spec.num_classes))

            #  temp_lab_bag = np.repeat(temp_lab_bag,2)
             for i in xrange(len(logit_values)):
                val_calc.ingest_result(logit_values=logit_values[i],
                                       logit_indices=logit_indices[i],
                                       bag_label=temp_lab_bag[i])

     if train_spec.val_metric == 'loss':
         return val_total/val_counter
     if train_spec.val_metric == 'score':
         return val_calc.get_f1_score()


def train_model(model,
                train_op,
                train_loss,
                val_op,
                img_placehldr,
                t_lab_placehldr,
                v_lab_placehldr,
                train_img_bags,
                train_labels,
                val_img_bags,
                val_labels,
                data_spec,
                train_spec,
                checkpoint_path,
                model_data_path):

    # Path validation
    if not osp.exists(checkpoint_path):
        raise Exception('Checkpoint path: {} is invalid'.format(checkpoint_path))

    if not osp.exists(model_data_path):
        raise Exception('Model data path: {} is invalid'.format(model_data_path))

    val_checker = EarlyStoppingSupervisor(stop_type=train_spec.val_metric)
    val_calc = ResultProcessor(num_datapoints=data_spec.num_datapoints(),
                               num_test_classes=data_spec.num_classes,
                               pos_class=SCREEN_INDX,
                               pos_class_thresh=train_spec.val_thresh,
                               testing_type='indicator',
                               debug=False)

    dropout_placeholder = model.use_dropout
    val_point = data_spec.num_train/train_spec.batch_size

    with tf.Session() as sess:

        # Initilise the graph
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

        # Create a variable saver for the Session
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

        # Determine graph state based on training type
        if train_spec.init_type == 'Fine-tune':
            # Load the data model from original data
            print ('Loading model data from: {}'.format(model_data_path))
            model.load(model_data_path, sess, ignore_missing=True)
            step = 0

        if train_spec.init_type == 'Multi-fine-tune':
            # Collect variables to restore
            train_vars = []
            for var in tf.trainable_variables():
                if var.name.find('logits') == -1:
                    train_vars.append(var)
            restorer = tf.train.Saver(train_vars, max_to_keep=2)

            # Load the model from a previous checkpoint
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=model_data_path)

            if ckpt and ckpt.model_checkpoint_path:
                restorer.restore(sess, ckpt.model_checkpoint_path)

                # Retrieve step number from checkpoint path
                step = 0
                print ('Successfully loaded model from {} at step {}'.format(
                                            ckpt.model_checkpoint_path,step))

            else:
                print ('No checkpoint file found at {}'.format(checkpoint_path))
                return

            # Re-initilise the last training layer
            with tf.variable_scope("logits", reuse=True):
                old_weights = tf.get_variable("weights")
                old_biases = tf.get_variable("biases")

            # Use xavier initilisation of new weights
            new_weights = tf.get_variable("reinit_weights",
                            shape=old_weights.get_shape(),
                            initializer=tf.contrib.layers.xavier_initializer())
            new_biases = tf.get_variable("reinit_biases",
                            shape=old_biases.get_shape(),
                            initializer=tf.contrib.layers.xavier_initializer())

            new_init_op = tf.variables_initializer([new_weights, new_biases],
                                                    name="logit_init")
            sess.run(new_init_op)

            reset_weights = old_weights.assign(new_weights)
            reset_biases = old_biases.assign(new_biases)
            sess.run(reset_weights)
            sess.run(reset_biases)

            # Re-initilise learning rates
            with tf.variable_scope("global", reuse=True):
                old_global_step = tf.get_variable('global_step')
            reset_global_step = old_global_step.assign(0)
            sess.run(reset_global_step)

        if train_spec.init_type == 'Restore':
            # Load the model from a previous checkpoint
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=model_data_path)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # Retrieve step number from checkpoint path
                step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
                print ('Successfully loaded model from {} at step {}'.format(
                                    ckpt.model_checkpoint_path,step))
            else:
                raise IOError('No checkpoint file found at {}'.format(
                                                        checkpoint_path))

        if train_spec.init_type == 'Raw':
            print ('Initialising model from scratch...')
            step = 0

        # Create a training summary writer
        sum_op, train_writer = setup_train_summary_writers(checkpoint_path, sess)
        checkpoint_file = osp.join(checkpoint_path,
                                  '{}.model.ckpt'.format(model.name))

        # Begin data preparation
        print ('Initialising batch queues...')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print ('Commencing Training:')
        print ('Training for {} epochs'.format(train_spec.num_epochs))
        print ('=' * 60)

        # Save the graph and Network State - Parameter selection
        print ('Saving model checkpoint at {}'.format(checkpoint_file))
        saver.save(sess, checkpoint_file, global_step=step)

        # Main training loop
        print ("Preparing images...")
        epoch_val = 0
        while epoch_val < train_spec.num_epochs:

            # Validation step
            if step % (np.round(train_spec.val_frequency*val_point)) == 0:
                val_error = get_val_error(session=sess,
                                          val_op=val_op,
                                          img_data=val_img_bags,
                                          img_placehldr=img_placehldr,
                                          label_data=val_labels,
                                          label_placehldr=v_lab_placehldr,
                                          dropout_placehldr=dropout_placeholder,
                                          train_spec=train_spec,
                                          data_spec=data_spec,
                                          val_calc=val_calc)
                summary = tf.Summary()
                summary.value.add(tag='training/Validation_Accuracy',
                                  simple_value=val_error)
                train_writer.add_summary(summary, step)
                if train_spec.val_metric == 'score':
                    print ('Validation F1-score: {}'.format(val_error))
                if train_spec.val_metric == 'loss':
                    print ('Validation Loss: {}'.format(val_error))

                # can be 'True', 'False' or 'None'
                stopping_decision = val_checker.process_validation_result(
                                    step=step, validation_score=val_error)

                if stopping_decision is False:
                    # Save the graph and Network State - Early stopping
                    print ('Saving model checkpoint at {}'.format(checkpoint_file))
                    saver.save(sess, checkpoint_file, global_step=step)

                elif stopping_decision is True:
                    print ('Early-stopping criterion met - Training completed')
                    coord.request_stop()
                    coord.join(threads=threads, stop_grace_period_secs=2)
                    sess.close()
                    break

            # Fetch a batch of training bags
            img_batch, label_batch = sess.run([train_img_bags,
                                               train_labels])

            if train_spec.training_type == 'stage_1':
                # Reshape batches for img bag insertion
                img_batch = np.reshape(img_batch,
                    newshape=(train_spec.batch_size*train_spec.img_bag_count,
                              train_spec.model_img_len,
                              train_spec.model_img_len,3))

            feed = {img_placehldr:img_batch,
                    t_lab_placehldr:label_batch,
                    dropout_placeholder: True}

            # Compute training params for the current forward pass
            summary, net_loss, _ = sess.run([sum_op, train_loss, train_op],
                                            feed_dict=feed)


            # Display results for current iteration
            curr_time = datetime.datetime.now()
            epoch_val = round((float(step) * float(train_spec.batch_size)) \
                        / float(data_spec.num_train),2)

            print ('Step: {}, Epoch: {}, Time: {} Loss: {}'.format(
                step, epoch_val, curr_time.strftime("%H:%M:%S"), net_loss))

            step += 1 #Increment global training step


    # Terminate the session
    print ("Training completed")
    coord.request_stop()
    coord.join(threads=threads)
    sess.close()
