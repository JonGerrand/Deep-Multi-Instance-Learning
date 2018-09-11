"""
    @Description: Class responsible for feeding data during the training,
                  validation and testing of stage-1 and stage-2 of the
                  MIL method.
    @Author: Jonathan Gerrand
"""

import tensorflow as tf
import os
import numpy as np

# Run config
USE_GPU=1

class InputPipeline(object):

    def __init__(self,
                 training_spec,
                 dataset_spec,
                 pipeline_type='stage_1',
                 num_readers=25,
                 patch_type='grid'):

        # Input validation
        if pipeline_type not in ['stage_1', 'stage_1_val', \
                                 'stage_2', 'stage_2_val', \
                                 'testing', 'saliency']:
             raise IOError ("Pipeline type not supported: \
                            {}".format(pipeline_type))

        self.train_spec = training_spec
        self.dataset_spec = dataset_spec
        self.bounding_box = self._create_bounding_box(pipeline_type)
        self.pipeline_type = pipeline_type
        self.num_readers = num_readers
        self.patch_type = patch_type
        self.img_patch_coords = self._get_img_coords()
        self.patch_ids = tf.constant([0 for i in xrange(0,
                                     self.train_spec.img_bag_count)],
                                     dtype=tf.int32)

    def _get_img_coords(self):

        if self.patch_type == 'grid_cent':
            img_boxes = []
            # Top left
            img_boxes.append([0.0,0.0,0.5,0.5])
            # Top right
            img_boxes.append([0.0,0.5,0.5,1.0])
            # Mid left
            img_boxes.append([0.3,0.0,0.7,0.5])
            # Mid right
            img_boxes.append([0.3,0.5,0.7,1.0])
            # Bottom left
            img_boxes.append([0.5,0.0,1.0,0.5])
            # Bottom right
            img_boxes.append([0.5,0.5,1.0,1.0])
            img_boxes = np.array(img_boxes)
            return tf.constant(img_boxes,dtype=tf.float32)

        if self.patch_type == 'grid':
            x1, y1 = 0, 0
            x2, y2 = self.train_spec.model_img_len, self.train_spec.model_img_len
            img_boxes = []
            # Calculate number of image patches to extract
            row_box_count = self.train_spec.row_box_count
            # Extrapulate to grid
            for col_val in xrange(0, row_box_count):
                for row_val in xrange(0, row_box_count):
                    if (x2 <= self.train_spec.raw_img_len) \
                    and (y2 <= self.train_spec.raw_img_len):
                        img_boxes.append([y1,x1,y2,x2])

                    # Reach end of Row
                    if (x2 >= self.train_spec.raw_img_len):
                        x1 = 0
                        x2 = self.train_spec.model_img_len
                    else:
                        x1 += self.train_spec.crop_step_size
                        x2 += self.train_spec.crop_step_size

                y1 += self.train_spec.crop_step_size
                y2 += self.train_spec.crop_step_size

        if self.patch_type == 'horizontal':
            x1, y1 = 0, 0
            x2, y2 = self.train_spec.model_img_len, self.train_spec.raw_img_len
            img_boxes = []
            # Calculate number of image patches to extract
            row_box_count = self.train_spec.row_box_count
            # Extrapulate horizontally
            for row_val in xrange(0, row_box_count):
                img_boxes.append([y1,x1,y2,x2])
                x1 += self.train_spec.crop_step_size
                x2 += self.train_spec.crop_step_size

        if self.patch_type == 'vertical':
            x1, y1 = 0, 0
            x2, y2 = self.train_spec.raw_img_len, self.train_spec.model_img_len
            img_boxes = []
            # Calculate number of image patches to extract
            row_box_count = self.train_spec.row_box_count
            # Extrapulate horizontally
            for col_val in xrange(0, row_box_count):
                img_boxes.append([y1,x1,y2,x2])
                y1 += self.train_spec.crop_step_size
                y2 += self.train_spec.crop_step_size

        img_boxes = np.array(img_boxes)/float(self.train_spec.raw_img_len)
        return tf.constant(img_boxes,dtype=tf.float32)

    def _create_bounding_box(self, training_type):
        """
        Creates a bounding box covering the entire area of an image.
        Returns:
            3-D Tensor specifying the dimensions of an image to which cropping
            will be applied.
        """
        if training_type == "testing":
            xmin = tf.expand_dims([0.0], 0)
            ymin = tf.expand_dims([0.0], 0)
            xmax = tf.expand_dims([1.0], 0)
            ymax = tf.expand_dims([1.0], 0)
        else:
            xmin = tf.expand_dims([0.1], 0)
            ymin = tf.expand_dims([0.1], 0)
            xmax = tf.expand_dims([0.9], 0)
            ymax = tf.expand_dims([0.9], 0)
        bbox = tf.concat(axis=0, values=[ymin, xmin, ymax, xmax])
        bbox = tf.expand_dims(bbox, 0)
        bbox = tf.transpose(bbox, [0, 2, 1])
        return bbox

    def decode_image(self, img_string):
        return tf.image.decode_image(contents=img_string,
                                     channels=3,
                                     name="Image_decode")

    @staticmethod
    def read_csv(filename_queue):
        """
        Decode the contents of a .csv file
        """
        reader = tf.TextLineReader(skip_header_lines=1)
        # Read the file
        _, value = reader.read(filename_queue)
        # Decode the output of the reader
        record_defaults = [[0], ['']]
        ml_class, img_path = tf.decode_csv(records=value,
                                           record_defaults=record_defaults)
        ml_class = tf.to_int32(ml_class)
        return ml_class, img_path

    def process_image(self, image_path):
        """
        Adapter function for various image processing techniques used in
        the pipeline
        """
        if self.pipeline_type == 'stage_1':
            return self.create_image_bag(image_path,
                            inline_augmentation=False)
        if self.pipeline_type == 'stage_1_val':
            return self.create_image_bag(image_path,
                            inline_augmentation=False)
        if self.pipeline_type == 'stage_2': \
            return self.create_training_image(image_path,
                                inline_augmentation=False)
        if self.pipeline_type == 'stage_2_val':
            return self.create_image_bag(image_path,
                        inline_augmentation=False)
        if self.pipeline_type == 'testing' \
        or self.pipeline_type == 'saliency':
            return self.create_image_bag(image_path,
                          inline_augmentation=False)

    def process_label(self, sparse_label):
        """
        Adapter function for various label processing
        """
        if self.pipeline_type == 'stage_1':
            return tf.sparse_to_dense(sparse_indices=sparse_label,
                                  output_shape=[self.dataset_spec.num_classes],
                                  sparse_values=1.0,
                                  default_value=0.0)

        if self.pipeline_type == 'stage_1_val':
            if self.train_spec.val_metric == 'score':
                return sparse_label
            if self.train_spec.val_metric == 'loss':
                return tf.sparse_to_dense(sparse_indices=sparse_label,
                                      output_shape=[self.dataset_spec.num_classes],
                                      sparse_values=1.0,
                                      default_value=0.0)

        if self.pipeline_type == 'stage_2':
            # Perform hot-encoding of the labels
            return tf.sparse_to_dense(sparse_indices=sparse_label,
                                  output_shape=[self.dataset_spec.num_classes],
                                  sparse_values=1.0,
                                  default_value=0.0)

        if self.pipeline_type == 'testing'\
        or self.pipeline_type == 'stage_2_val':
            return sparse_label

        if self.pipeline_type == 'saliency':
            dense_label = tf.sparse_to_dense(sparse_indices=sparse_label,
                                      output_shape=[self.dataset_spec.num_classes],
                                      sparse_values=1.0,
                                      default_value=0.0)
            dense_label = [dense_label for i in xrange(self.train_spec.img_bag_count)]
            return [sparse_label, dense_label]

    def create_image_bag(self, img_path, inline_augmentation=False):
        """
        Divide a single image into an image-bag
        """
        img_data = tf.read_file(img_path)
        img = self.decode_image(img_data)

        if inline_augmentation:
            # Random crop
            bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
                tf.shape(img),
                bounding_boxes=self.bounding_box,
                min_object_covered=self.train_spec.crop_ratio,
                aspect_ratio_range=[0.75, 1.33],
                area_range=[0.05, 1.0],
                max_attempts=100,
                use_image_if_no_bounding_boxes=True)

            img = tf.slice(img, bbox_begin, bbox_size)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_hue(img, max_delta=0.1)

        # Uint8 kernel not currently supported for img bag
        img = tf.cast(img, tf.float32)
        img = tf.expand_dims(img,0)

        # Subdivide img into patches
        img_patches =  tf.image.crop_and_resize(image=img,
                            boxes=self.img_patch_coords,
                            box_ind=self.patch_ids,
                            crop_size=[self.train_spec.model_img_len,
                                       self.train_spec.model_img_len])
        return img_patches

    def create_training_image(self, image_path, inline_augmentation=False):
        """
        Creates a minor distroted image for training
        """
        img_data = tf.read_file(image_path)
        img = self.decode_image(img_data)

        # Inline Augmentations
        if inline_augmentation:
            # Random crop
            distorted_bounding_box = tf.image.sample_distorted_bounding_box(
                tf.shape(img),
                bounding_boxes=self.bounding_box,
                min_object_covered=self.train_spec.crop_ratio,
                aspect_ratio_range=[0.75, 1.33],
                area_range=[0.05, 1.0],
                max_attempts=100,
                use_image_if_no_bounding_boxes=True)
            bbox_begin, bbox_size, distort_bbox = distorted_bounding_box
            img = tf.slice(img, bbox_begin, bbox_size)
            img = tf.image.resize_images(img,
                                        [self.train_spec.model_img_len,
                                         self.train_spec.model_img_len])
            img.set_shape([self.train_spec.model_img_len,
                           self.train_spec.model_img_len, 3])
            # Image flipping
            img = tf.image.random_flip_up_down(img)
            # Hue and contrast alteration
            img = tf.image.random_hue(img, max_delta=0.1)

            return img

        # Resize image to original height and width
        img = tf.expand_dims(img, 0)
        img = tf.image.resize_bilinear(img, [self.train_spec.model_img_len,
                                             self.train_spec.model_img_len],
                                             align_corners=False)
        img = tf.squeeze(img)
        return img

    def create_validation_image(self, image_path):
        """
            Creates a centrally cropped, single image for stage 2 validation
        """
        img_data = tf.read_file(image_path)
        img = self.decode_image(img_data)
        # Resize image to original height and width
        img = tf.expand_dims(img, 0)
        img = tf.image.resize_bilinear(img, [self.train_spec.model_img_len,
                                             self.train_spec.model_img_len],
                                             align_corners=False)
        img = tf.squeeze(img)
        return img

    def prepare_files(self,filename_queue):
        """
        Reads a single file instance from a queue to be consumed and returns a
        [processed_image_bag, processed_label] pair.
        Args:
            filename_queue: A Tensor Queue object containing the files to be
                            consumed.
        Returns:
            4-D Tensor of processed_image (float32)
            1-D Tensor of processed_label (float32)
        """
        img_class, img_path = self.read_csv(filename_queue)

        processed_image = self.process_image(img_path)
        processed_label = self.process_label(img_class)

        if self.pipeline_type == 'testing' \
        or self.pipeline_type == 'saliency':
            return processed_image, processed_label, img_path
        else:
            return processed_image, processed_label

    def generate_input_batches(self):
        """
        Generates queue objects which can be continuosly pulled from during
        training/validation/testing.
        """
        # Constrain to CPU processing
        with tf.device('/cpu'):

            if self.pipeline_type == 'stage_1':
                min_after_dequeue = self.dataset_spec.num_train
                capacity = min_after_dequeue + self.train_spec.batch_size * 5

                # Create file queue
                filename_queue = tf.train.string_input_producer(
                                [self.dataset_spec.train_file],
                                num_epochs=None,
                                shuffle=True,
                                name='train_CSV')
                img_bag, label = self.prepare_files(filename_queue=filename_queue)
                img_queue, label_queue = tf.train.shuffle_batch(
                                    [img_bag, label],
                                    batch_size=self.train_spec.batch_size,
                                    capacity=capacity,
                                    num_threads=self.num_readers,
                                    shapes=([self.train_spec.img_bag_count,
                                            self.train_spec.model_img_len,
                                            self.train_spec.model_img_len,3],
                                           [self.dataset_spec.num_classes]),
                                           min_after_dequeue=min_after_dequeue)

                return img_queue, label_queue

            if self.pipeline_type == 'stage_1_val':
                if self.train_spec.val_metric == 'loss':
                    lab_struct = [self.dataset_spec.num_classes]
                if self.train_spec.val_metric == 'score':
                    lab_struct = []

                # Create file queue
                filename_queue = tf.train.string_input_producer(
                                [self.dataset_spec.validation_file],
                                 num_epochs=None,
                                 shuffle=False,
                                 name='val_CSV')
                img_bag, label = self.prepare_files(filename_queue=filename_queue)
                img_queue, label_queue = tf.train.batch([img_bag, label],
                                 batch_size=1,
                                 num_threads=self.num_readers,
                                 shapes=([self.train_spec.img_bag_count,
                                        self.train_spec.model_img_len,
                                        self.train_spec.model_img_len,
                                        3],
                                        lab_struct))

                # Fully preprocessed and static set. Therefore we exhaust the
                # queue of val/test files and load them into memory.
                config = tf.ConfigProto(device_count={'GPU':USE_GPU})
                with tf.Session(config=config) as prep_sess:
                    print ("Preparing Validation data...")
                    # Begin queue runners
                    init_op = tf.group(tf.global_variables_initializer(),
                                       tf.local_variables_initializer())
                    prep_sess.run(init_op)
                    prep_coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=prep_sess,
                                                           coord=prep_coord)

                    # Create static lists to be populated
                    img_batch_queue = []
                    label_batch_queue = []
                    for i in xrange(self.dataset_spec.num_val):
                        print ("Processing image {} of {}".format(i,
                                            self.dataset_spec.num_val))
                        img, label = prep_sess.run([img_queue,label_queue])
                        img_batch_queue.append(img[0])
                        label_batch_queue.append(label[0])

                    print ("Preparation completed")
                    prep_coord.request_stop()
                    prep_coord.join(threads=threads, stop_grace_period_secs=10)
                    prep_sess.close()

                return img_batch_queue, label_batch_queue

            if self.pipeline_type == 'stage_2':
                min_after_dequeue = self.dataset_spec.num_train -1
                capacity = round(self.dataset_spec.num_train * 2)
                # Create file queue
                filename_queue = tf.train.string_input_producer(
                                [self.dataset_spec.train_file],
                                num_epochs=None,
                                shuffle=True,
                                name='train_CSV')
                img_bag, label = self.prepare_files(filename_queue=filename_queue)
                img_queue, label_queue = tf.train.shuffle_batch(
                                    [img_bag, label],
                                    batch_size=self.train_spec.batch_size,
                                    capacity=capacity,
                                    num_threads=self.num_readers,
                                    shapes=([self.train_spec.model_img_len,
                                             self.train_spec.model_img_len,3],
                                             [self.dataset_spec.num_classes]),
                                    min_after_dequeue=min_after_dequeue)
                return img_queue, label_queue

            if self.pipeline_type == 'stage_2_val':
                # Create file queue
                filename_queue = tf.train.string_input_producer(
                                [self.dataset_spec.validation_file],
                                 num_epochs=None,
                                 shuffle=False,
                                 name='val_CSV')
                val_img, label = self.prepare_files(
                                filename_queue=filename_queue)
                img_queue, label_queue = tf.train.batch([val_img, label],
                                 batch_size=1,
                                 num_threads=self.num_readers,
                                 shapes=([self.train_spec.img_bag_count,
                                          self.train_spec.model_img_len,
                                          self.train_spec.model_img_len,3],
                                          []))
                # Fully preprocessed and static set. Therefore we exhaust the
                # queue of val/test files and load them into memory.
                config = tf.ConfigProto(device_count={'GPU':USE_GPU})
                with tf.Session(config=config) as prep_sess:
                    print ("Preparing Validation data...")
                    # Begin queue runners
                    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                    prep_sess.run(init_op)
                    prep_coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=prep_sess, coord=prep_coord)

                    # Create static lists to be populated
                    img_batch_queue = []
                    label_batch_queue = []
                    for i in xrange(self.dataset_spec.num_val):
                        print ("Processing image {} of {}".format(i,
                                            self.dataset_spec.num_val))
                        img, label = prep_sess.run([img_queue,label_queue])
                        img_batch_queue.append(img[0])
                        label_batch_queue.append(label[0])

                    print ("Preparation completed")
                    prep_coord.request_stop()
                    prep_coord.join(threads=threads, stop_grace_period_secs=10)
                    prep_sess.close()

                return img_batch_queue, label_batch_queue

            if self.pipeline_type == 'testing':
                # Create file queue
                filename_queue = tf.train.string_input_producer(
                                [self.dataset_spec.get_subset()],
                                 num_epochs=1,
                                 shuffle=False,
                                 name='test_CSV')
                img_bag, label, img_path = self.prepare_files(
                                 filename_queue=filename_queue)
                img_queue, label_queue, path_queue = tf.train.batch(
                                 [img_bag, label, img_path],
                                 batch_size=1,
                                 num_threads=self.num_readers,
                                 shapes=([self.train_spec.img_bag_count,
                                          self.train_spec.model_img_len,
                                          self.train_spec.model_img_len,3],
                                          [],[]))

                return img_queue, label_queue, path_queue

            if self.pipeline_type == 'saliency':
                # Create file queue
                print (self.dataset_spec.get_subset())
                filename_queue = tf.train.string_input_producer(
                                [self.dataset_spec.get_subset()],
                                 num_epochs=1,
                                 shuffle=False,
                                 name='salient_CSV')
                img_bag, label, img_path = self.prepare_files(
                                 filename_queue=filename_queue)
                img_queue, sparse_labs, dense_labs, path_queue = tf.train.batch(
                                 [img_bag, label[0], label[1], img_path],
                                 batch_size=1,
                                 num_threads=self.num_readers,
                                 shapes=([self.train_spec.img_bag_count,
                                          self.train_spec.model_img_len,
                                          self.train_spec.model_img_len,3],
                                          [],
                                          [self.train_spec.img_bag_count,
                                           self.dataset_spec.num_classes],
                                          []))

                return img_queue, sparse_labs, dense_labs, path_queue
