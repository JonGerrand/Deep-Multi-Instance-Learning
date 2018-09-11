"""
    @Description:       This module contains testing functionality
                        for the evaluation of both stage-1 and stage-2 of
                        the MIL method.

                        Several evaluation methods will be employed in the
                        classification procedure:
    @Author:            Jonathan Gerrand
"""

from input_pipeline import InputPipeline
from dataset import Dataset
from training_spec import TrainingSpec
from argparse import ArgumentParser

import argparse
import training
import models
import tensorflow as tf
import os.path as osp
import numpy as np
import models

class ResultProcessor(object):

    def __init__(self,
                 num_datapoints=1,
                 testing_type='discrim',
                 num_test_classes=2,
                 pos_class=1,
                 pos_class_thresh=0.7,
                 debug=True):

        self.num_incorrect=0.0
        self.num_correct=0.0
        self.TP_count=0.0
        self.TN_count=0.0
        self.FP_count=0.0
        self.FN_count=0.0
        self.sensitivity = 0.0
        self.specificity = 0.0
        self.f1_score = 0.0
        self.auc = 0.0
        self.pos_score = [] # List for holding positive prediction vals
        self.label_list = [] # Used for ROC calculation
        self.debug=debug
        self.testing_type=testing_type
        self.num_datapoints=num_datapoints
        self.num_classes=num_test_classes
        self.pos_class=pos_class
        self.pos_class_thresh=pos_class_thresh

    # Adapter method
    def _analyse_bag(self, vals, indices, bag_label):
        if self.testing_type == 'greatest':
            self._greatest_value_analysis(vals, indices, bag_label)
        if self.testing_type == 'indicator':
            self._pos_indicator_analysis(vals, indices, bag_label)
        pass

    def flush_vals(self):
        self.num_correct=0.0
        self.num_incorrect=0.0
        self.TP_count=0.0
        self.TN_count=0.0
        self.FP_count=0.0
        self.FN_count=0.0
        self.sensitivity = 0.0
        self.specificity = 0.0
        self.f1_score = 0.0
        self.auc = 0.0

    def _pos_indicator_analysis(self, vals, indices, bag_label):

        temp_vals = vals[::-1]
        temp_indices = indices[::-1]
        # Search for positive class in bag
        for patch in xrange(len(temp_vals)):
            if temp_indices[patch] == self.pos_class \
            and temp_vals[patch] >= self.pos_class_thresh:
                if temp_indices[patch] == bag_label:
                    self.num_correct += 1
                    self.TP_count += 1
                    if self.debug:
                        print (temp_vals)
                        print (temp_indices)
                        print ("Num correct: {}".format(self.num_correct))
                    return 0
                else:
                    self.num_incorrect += 1
                    self.FP_count += 1
                    if self.debug:
                        print (temp_vals)
                        print (temp_indices)
                        print ("Num incorrect: {}".format(self.num_incorrect))
                    return 0
        # All negative prediction
        if self.pos_class not in temp_indices:
            if bag_label != self.pos_class:
                self.num_correct += 1
                self.TN_count += 1
                if self.debug:
                    print (temp_vals)
                    print (temp_indices)
                    print ("Num correct: {}".format(self.num_correct))
            if bag_label == self.pos_class:
                self.num_incorrect += 1
                self.FN_count += 1
                if self.debug:
                    print (temp_vals)
                    print (temp_indices)
                    print ("Num incorrect: {}".format(self.num_incorrect))
        # Positive predictions below threshold
        elif bag_label != self.pos_class \
        and temp_vals[np.where(temp_indices == self.pos_class)[0][0]] \
            < self.pos_class_thresh:
            self.num_correct += 1
            self.TN_count += 1
            if self.debug:
                print (temp_vals)
                print (temp_indices)
                print ("Num correct: {}".format(self.num_correct))
        # False Positives above threshold
        else:
            self.num_incorrect += 1
            self.FN_count += 1
            if self.debug:
                print (temp_vals)
                print (temp_indices)
                print ("Num incorrect: {}".format(self.num_incorrect))

    def _greatest_value_analysis(self, vals, indices, bag_label, debug=True):
        for patch in reversed(xrange(len(vals))):
            if indices[patch] != 2:
                if indices[patch] == bag_label:
                    self.num_correct += 1
                    if debug:
                        print (vals[-15:])
                        print (indices[-15:])
                        print ("Num correct: {}".format(self.num_correct))
                    break
                else:
                    self.num_incorrect += 1
                    if debug:
                        print (vals[-15:])
                        print (indices[-15:])
                        print ("Num incorrect: {}".format(self.num_incorrect))
                    break

    def ingest_result(self,
                      logit_values,
                      logit_indices,
                      bag_label):
        temp_vals = np.array([x[0] for x in logit_values])
        temp_indices = np.array([x[0] for x in logit_indices])
        inds = temp_vals.argsort()
        temp_vals = temp_vals[inds]
        temp_indices = temp_indices[inds]

        # Append for ROC calc
        self._extract_roc_vals(logit_values, logit_indices, bag_label)

        # Process testing result
        self._analyse_bag(temp_vals, temp_indices, bag_label)
        pass

    def _extract_roc_vals(self, logit_values, logit_indices, bag_label):
        # Select index of largest pos value in bag in an iterative manner
        bag_pos_score = 0
        for patch in xrange(len(logit_values)):
            indicator_indx = (len(logit_indices[patch])-1) - \
            logit_indices[patch].tolist()[::-1].index(self.pos_class)
            patch_pos_score = logit_values[patch][indicator_indx]
            if patch_pos_score > bag_pos_score:
                bag_pos_score = patch_pos_score

        self.pos_score.append(bag_pos_score)
        self.label_list.append(bag_label)

    def _calc_sensitivity(self):
        try:
            self.sensitivity =  np.round(self.TP_count/(self.TP_count + self.FN_count),5)
        except ZeroDivisionError:
            self.sensitivity = 0.0

    def _calc_specificity(self):
        try:
            self.specificity = np.round(self.TN_count/(self.TN_count + self.FP_count),5)
        except ZeroDivisionError:
            self.specificity = 0.0

    def _calc_f1_score(self):
        self._calc_sensitivity()
        self._calc_specificity()
        try:
            self.f1_score = 2*((self.sensitivity*self.specificity)/ \
                              (self.sensitivity + self.specificity))
        except ZeroDivisionError:
            self.f1_score = 0.0

    def produce_roc_values(self, file_path):
        from sklearn import metrics
        import csv

        # Calculate True-positive rate and False-positive rate
        fpr, tpr, _ = metrics.roc_curve(np.array(self.label_list),
                                        np.array(self.pos_score),
                                        pos_label=self.pos_class)

        self.auc = metrics.auc(fpr, tpr)

        # Write results to file
        # ROC curve
        with open(osp.join(file_path, 'ROC.csv'), 'w') as csvfile:
            fieldNames = ['fpr', 'tpr']
            writer = csv.DictWriter(csvfile, fieldnames=fieldNames)
            writer.writeheader()
            for i in xrange(0, len(fpr)):
                writer.writerow({'fpr': fpr[i] , 'tpr': tpr[i]})
            csvfile.close()
        #AUC
        with open(osp.join(file_path, 'AUC.txt'), 'w') as txtFile:
            txtFile.write('AUC: {}'.format(self.auc))
        txtFile.close()

    def get_sensitivity(self):
        self._calc_sensitivity()
        return self.sensitivity

    def get_specificity(self):
        self._calc_specificity()
        return self.specificity

    def get_f1_score(self):
        self._calc_f1_score()
        return self.f1_score

    def get_accuracy(self):
        return (self.num_correct / self.num_datapoints)

def launch_testing(model_name,
                   img_data_path,
                   model_data_path,
                   subset_type,
                   raw_img_len,
                   model_img_len,
                   crop_step_size,
                   num_test_classes,
                   patch_type,
                   testing_type='discrim',
                   use_gpu=False,
                   thresh_val=0.7,
                   use_roc=False):

    import time

    # Argument validation
    if not osp.isdir(img_data_path):
        raise IOError("Invalid img dir: {}".format(img_data_path))
    if not osp.isdir(model_data_path):
        raise IOError("Invalid model data dir: {}".format(model_data_path))
    if testing_type not in ['greatest','indicator']:
        raise IOError("Testing method not supported: {}".format(testing_type))

    # Load testing dependencies
    train_spec = TrainingSpec(batch_size=1,
                              raw_img_len=raw_img_len,
                              model_img_len=model_img_len,
                              crop_step_size=crop_step_size,
                              patch_type=patch_type)
    data_spec = Dataset(root_dir=img_data_path,
                        dataset_type=subset_type)
    res_calc = ResultProcessor(num_datapoints=data_spec.num_datapoints(),
                               testing_type=testing_type,
                               num_test_classes=num_test_classes,
                               pos_class_thresh=thresh_val)

    # Create placeholders
    img_node = tf.placeholder(tf.float32,
                shape=(None, train_spec.model_img_len,
                       train_spec.model_img_len,3))

    # Setup batch queues
    pipeline = InputPipeline(training_spec=train_spec,
                             dataset_spec=data_spec,
                             pipeline_type='testing',
                             patch_type=patch_type)

    img_queue, sparse_labs, img_paths = \
                            pipeline.generate_input_batches()

    # Load model
    net = models.load_model_with_data(model_name=model_name,
                                      input_data=img_node,
                                      num_classes=num_test_classes,
                                      train=False)

    dropout_placehldr = net.use_dropout
    probs = net.get_output()
    pred_op = tf.nn.top_k(probs,
                          k=num_test_classes,
                          sorted=True)

    # Model parameter restoration
    saver = tf.train.Saver(tf.global_variables())

    # Simultaneous testing/training
    if use_gpu is False:
        config = tf.ConfigProto(device_count={'GPU':0})
    else:
        config = tf.ConfigProto(device_count={'GPU':1})

    # Commence extraction
    with tf.Session(config=config) as sess:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

        # Restore model from previous checkpoint
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=model_data_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

            step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print ("Successfully loaded model from {} \
                    at step {}".format(ckpt.model_checkpoint_path, step))

        else:
            raise IOError('No checkpoint file found at: \
                           {}'.format(model_data_path))

        # Initilise counters
        # Start queue runners
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print ('Starting evaluation...')

        for i in xrange(data_spec.num_datapoints()):

            img_batch, label_s, img_path = sess.run([img_queue,
                                                     sparse_labs,
                                                     img_paths])
            img_batch = np.reshape(img_batch,
                newshape=(train_spec.batch_size*train_spec.img_bag_count,
                          train_spec.model_img_len,
                          train_spec.model_img_len,
                          3))

            feed = {img_node:img_batch,
                    dropout_placehldr:False}

            start = time.time()

            logit_values, logit_indices = sess.run(pred_op,
                                            feed_dict=feed)
            end = time.time()

            print ("Processing image bag: {}".format(img_path[0]))
            # Extract discriminative classes and non-discriminative class
            res_calc.ingest_result(logit_values=logit_values,
                                   logit_indices=logit_indices,
                                   bag_label=label_s[0])

        print ("Accuracy: {}".format(res_calc.get_accuracy()))
        print ("Sensitivity @ class 1: {}".format(res_calc.get_sensitivity()))
        print ("Specificity @ class 1: {}".format(res_calc.get_specificity()))

        if use_roc:
            res_calc.produce_roc_values(model_data_path)
            print ("AUC: {}".format(res_calc.auc))

        # Wrap-up testing
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

def main():
    parser = ArgumentParser(description='Perform model testing for MIL')

    parser.add_argument('--model_name',
                        default='GoogleNet',
                        help='(Options: GoogleNet, NIN, BodyNet)')

    parser.add_argument('--train_data_path',
                        default='',
                        help='Default: " "')

    parser.add_argument('--model_data_path',
                        default='',
                        help='Default: " "')

    parser.add_argument('--subset_type',
                        default='train',
                        help='(Options: train, val, test)')

    parser.add_argument('--testing_type',
                        default='indicator',
                        help='(Options: greatest, indicator)')

    parser.add_argument('--raw_img_len',
                        default=376,
                        type=int,
                        help='Default: 376')

    parser.add_argument('--crop_step_size',
                        default=38,
                        type=int,
                        help='Default: 38')

    parser.add_argument('--model_img_len',
                        default=224,
                        type=int,
                        help='Default: 224')

    parser.add_argument('--num_test_classes',
                        default=3,
                        type=int,
                        help='Default: 3')

    parser.add_argument('--patch_type',
                        default='grid',
                        help='Default: grid '\
                        '(Options: grid, horizontal, vertical)')

    parser.add_argument('--thresh',
                        default=0.7,
                        type=float,
                        help='Default: 0.7')

    parser.add_argument('--use_gpu', dest='gpu', action='store_true')

    parser.add_argument('--roc_curve', dest='roc', action='store_true')

    parser.set_defaults(gpu=False, roc=False)

    args = parser.parse_args()

    launch_testing(model_name=args.model_name,
                   img_data_path=args.train_data_path,
                   model_data_path=args.model_data_path,
                   subset_type=args.subset_type,
                   testing_type=args.testing_type,
                   raw_img_len=args.raw_img_len,
                   crop_step_size=args.crop_step_size,
                   num_test_classes=args.num_test_classes,
                   model_img_len=args.model_img_len,
                   patch_type=args.patch_type,
                   thresh_val=args.thresh,
                   use_gpu=args.gpu,
                   use_roc=args.roc)

if __name__ == '__main__':
    main()
