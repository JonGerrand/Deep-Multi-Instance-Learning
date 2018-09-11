class TrainingSpec(object):
    """
    Class to encapsulate all training information for a given session.
    """

    def __init__(self,
                 batch_size=8,
                 num_epochs=80,
                 raw_img_len=376,
                 model_img_len=224,
                 crop_step_size=38,
                 crop_ratio=0.8,
                 reinit_learn_rate=0.1,
                 fine_tune_learn_rate=0.001,
                 optimiser='SGD',
                 val_metric='loss',
                 init_type='Fine-tune',
                 training_type='stage_1',
                 val_frequency=1,
                 weight_decay=0.0001,
                 patch_type='grid',
                 val_thresh=0.5):

         """
         Constructor
         """

         #Parameter validation
         assert (batch_size > 0)
         self.batch_size = batch_size

         if ((raw_img_len - model_img_len) % crop_step_size != 0):
             if patch_type != 'grid_cent':
                 raise ValueError("Invalid crop, img and step sizes.")
         self.raw_img_len = raw_img_len
         self.model_img_len = model_img_len
         self.crop_step_size = crop_step_size
         self.crop_ratio = crop_ratio

         assert num_epochs > 0
         self.num_epochs = num_epochs

         # Standard training values
         self.num_epochs_per_decay=40.0
         self.learn_rate_decay_factor=0.1
         self.rmsprop_momentum=0.9
         self.rmsprop_decay=0.9
         self.adam_beta1=0.9
         self.adam_beta2=0.999
         self.sgd_momentum=0.9
         self.weight_decay=weight_decay

         if reinit_learn_rate < 0:
             raise ValueError("Invalid reinit learn rate: {}".format(
                                                    reinit_learn_rate))
         self.reinit_learn_rate=reinit_learn_rate

         if fine_tune_learn_rate < 0:
             raise ValueError("Invalid fine-tune learn rate: {}".format(
                                                    fine_tune_learn_rate))
         self.fine_tune_learn_rate=fine_tune_learn_rate

         if optimiser not in ['SGD', 'Adam', 'RMSProp']:
             raise ValueError("Selected optimiser is not supported: {}".format(
                                                                    optimiser))
         self.optimiser=optimiser

         if val_metric not in ['loss', 'score']:
             raise ValueError("Selected validation metric is \
                               not supported: {}".format(val_metric))
         self.val_metric=val_metric

         if init_type not in ['Fine-tune', 'Restore','Multi-fine-tune',\
                              'Raw']:
             raise ValueError("Selected initilisation type is not \
                               supported: {}".format(init_type))
         self.init_type=init_type

         if training_type not in ['stage_1', 'stage_2', 'exp_mil']:
             raise ValueError("Selected training type is not \
                               supported: {}".format(training_type))
         self.training_type = training_type

         if patch_type not in ['grid', 'vertical', 'horizontal', 'grid_cent']:
             raise ValueError ("Selected patch_type is not \
                                supported: {}".format(patch_type))
         self.patch_type = patch_type

         if val_frequency < 0:
             raise ValueError("Validation interval is invalid: {}".format(
                                                            val_frequency))
         self.val_frequency=val_frequency

         self.val_thresh = val_thresh
         self._calculate_bag_properties()


    def _calculate_bag_properties(self):
        self.row_box_count = ((self.raw_img_len - self.model_img_len) / \
                              self.crop_step_size) + 1
        if self.patch_type == "grid":
            self.img_bag_count = self.row_box_count*self.row_box_count
        else:
            self.img_bag_count = self.row_box_count

        # Manual region
        if self.patch_type == 'grid_cent':
            self.row_box_count = 2
            self.img_bag_count = 6

        print ("Image bag size: {}".format(self.img_bag_count))
        pass
