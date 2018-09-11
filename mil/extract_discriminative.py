"""
    @Description: Module to extract Discriminative and non-discriminative
                  images from a stage-1 MIL trained model.
    @Author: Jonathan Gerrand
"""

from argparse import ArgumentParser
from training_spec import TrainingSpec
from dataset import Dataset
from input_pipeline import InputPipeline
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

import subprocess
import tensorflow as tf
import os
import os.path as osp
import numpy as np
import models
import cv2

# Constants
SCREEN_CLASS = 1
USE_GPU=1

def process_img_bag(logit_values, logit_indices, img_label,
                    img_path, img_data, save_root_path,
                    print_results=False, extract_count=4,
                    thresh_val=0.6):

    # Sort the lists based-off logit values
    logit_values = np.array([x[0] for x in logit_values])
    logit_indices = np.array([x[0] for x in logit_indices])
    inds = logit_values.argsort()

    logit_values = logit_values[inds]
    logit_indices = logit_indices[inds]
    img_data = img_data[inds]

    # For inspection purposes
    if print_results:
        print (logit_values)
        print (logit_indices)

    # Format image name
    img_name = img_path[0].split('/')[-1].split('.')[0]

    for j in reversed(xrange(len(logit_values))):

        # Non-discriminative -> False positive
        if (logit_indices[j] == SCREEN_CLASS) \
        and (logit_indices[j] != img_label[0]):
            file_img_name = "#".join([img_name, str(logit_values[j]),
                            str(logit_indices[j]), str(img_label[0]),".jpg"])

            cv2.imwrite(osp.join(save_root_path,'d_n',file_img_name),
                            img_data[j])

        # High Conf Non-discriminative
        if (logit_indices[j] != SCREEN_CLASS) \
        and (logit_values[j] > thresh_val) \
        and (logit_indices[j] != img_label[0]):
            file_img_name = "#".join([img_name, str(logit_values[j]),
                            str(logit_indices[j]), str(img_label[0]),".jpg"])

            cv2.imwrite(osp.join(save_root_path,'d_n',file_img_name),
                                img_data[j])

        #Low Conf Negative -> Classify as positive
        if (logit_indices[j] != SCREEN_CLASS) \
        and (logit_values[j] <= thresh_val) \
        and (logit_indices[j] != img_label[0]):
            file_img_name = "#".join([img_name, str(logit_values[j]),
                            str(logit_indices[j]), str(img_label[0]),".jpg"])
            cv2.imwrite(osp.join(save_root_path,'d_1',file_img_name),
                                img_data[j])


        # True positive/negative
        if (logit_indices[j] == img_label[0]):
            file_img_name = "#".join([img_name, str(logit_values[j]),
                            str(logit_indices[j]), str(img_label[0]),".jpg"])
            cv2.imwrite(osp.join(save_root_path,'d_{}'.format(img_label[0]),
                                 file_img_name),img_data[j])


def extract_img_patches(model_name,
                        img_data_path,
                        model_data_path,
                        save_root_path,
                        subset_type,
                        raw_img_len,
                        model_img_len,
                        crop_step_size,
                        patch_type,
                        extract_count,
                        thresh_val):

    # Argument validation
    if not osp.isdir(img_data_path):
        raise IOError("Invalid img dir: {}".format(img_data_path))
    if not osp.isdir(model_data_path):
        raise IOError("Invalid model data dir: {}".format(model_data_path))
    if not osp.isdir(save_root_path):
        raise IOError("Invalid save dir: {}".format(save_root_path))

    # Create class folders
    classes = ['d_0', 'd_1', 'd_n']

    for img_class in classes:
        cmd = "mkdir {}".format(osp.join(save_root_path,img_class))
        subprocess.call(cmd, shell=True)

    # Load extraction dependencies
    train_spec = TrainingSpec(batch_size=1,
                              raw_img_len=raw_img_len,
                              model_img_len=model_img_len,
                              crop_step_size=crop_step_size,
                              patch_type=patch_type)
    data_spec = Dataset(root_dir=img_data_path,
                        dataset_type=subset_type)

    with tf.Graph().as_default() as g:

        # Create placeholders
        img_node = tf.placeholder(tf.float32,
                  shape=(None,train_spec.model_img_len,
                         train_spec.model_img_len,3))
        label_node = tf.placeholder(tf.float32,
                                    shape=(None, data_spec.num_classes))

        # Setup batch queues
        pipeline = InputPipeline(training_spec=train_spec,
                                 dataset_spec=data_spec,
                                 pipeline_type='testing',
                                 patch_type=patch_type)

        img_queue, sparse_labs, img_path_queue = \
                                pipeline.generate_input_batches()

        # Load model
        net = models.load_model_with_data(model_name=model_name,
                                          input_data=img_node,
                                          num_classes=data_spec.num_classes,
                                          train=False)

        dropout_placehldr = net.use_dropout
        probs = net.get_output()
        pred_op = tf.nn.top_k(probs,
                              k=data_spec.num_classes,
                              sorted=True)

        # Model parameter restoration
        saver = tf.train.Saver(tf.global_variables())

        # Commence extraction
        config = tf.ConfigProto(device_count={'GPU':USE_GPU})
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

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            print ('Starting evaluation...')
            for i in xrange(data_spec.num_datapoints() -2):

                img_batch, label_s, img_path = sess.run([img_queue,
                                                         sparse_labs,
                                                         img_path_queue])
                img_batch = np.reshape(img_batch,
                    newshape=(train_spec.batch_size*train_spec.img_bag_count,
                              train_spec.model_img_len,
                              train_spec.model_img_len,
                              3))

                feed = {img_node:img_batch,
                        dropout_placehldr:False}

                logit_values, logit_indices = sess.run(pred_op,
                                                feed_dict=feed)

                print ("Processing image bag: {}".format(img_path[0]))

                process_img_bag(logit_values=logit_values,
                                logit_indices=logit_indices,
                                img_label=label_s,
                                img_path=img_path,
                                img_data=img_batch,
                                save_root_path=save_root_path,
                                extract_count=extract_count,
                                thresh_val=thresh_val)

            # Wrap-up testing
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

def main():
    parser = ArgumentParser(description='Extract discriminative img patches\
                                         for MIL model training')
    parser.add_argument('--model_name',
                        default='GoogleNet',
                        help='(Options: GoogleNet, NIN)')

    parser.add_argument('--img_data_path',
                        default='',
                        help='Default: " "')

    parser.add_argument('--model_data_path',
                        default='',
                        help='Default: " "')

    parser.add_argument('--save_root_path',
                        default='~/temp/',
                        help='Default: "~/temp/"')

    parser.add_argument('--subset_type',
                        default='train',
                        help='(Options: train, val, test)')

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

    parser.add_argument('--extract_count',
                        default=1,
                        type=int,
                        help='Default: 1')

    parser.add_argument('--patch_type',
                        default='grid',
                        help='Default: grid '\
                        '(Options: grid, horizontal, vertical)')

    parser.add_argument('--thresh_val',
                        default=0.6,
                        type=float,
                        help='Default: 0.6')

    args = parser.parse_args()

    extract_img_patches(model_name=args.model_name,
                        img_data_path=args.img_data_path,
                        model_data_path=args.model_data_path,
                        save_root_path=args.save_root_path,
                        subset_type=args.subset_type,
                        raw_img_len=args.raw_img_len,
                        model_img_len=args.model_img_len,
                        crop_step_size=args.crop_step_size,
                        patch_type=args.patch_type,
                        extract_count=args.extract_count,
                        thresh_val=args.thresh_val)

if __name__ == '__main__':
    main()
