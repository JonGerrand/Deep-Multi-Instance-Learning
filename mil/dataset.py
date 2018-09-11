"""
    @Description: Dataset Class used to store dataset statistics during
                  training, validation and testing
    @Author: Jonathan Gerrand
"""

import os.path as osp

class Dataset(object):

    def __init__(self,root_dir,dataset_type='train',num_classes=None):

        if dataset_type not in ['train', 'validation', 'test']:
            raise IOError('Incorrect subset chosen: {}'.format(dataset_type))
        self.root_dir = root_dir
        self.type = dataset_type
        self._read_db_metadata()
        # Overide Number of classes manually - needed for saliency mapping
        if num_classes is not None:
            self.num_classes = num_classes

    def _read_db_metadata(self):
        meta_file_location = osp.join(self.root_dir, 'meta_data.txt')
        if not osp.exists(meta_file_location):
            raise IOError('No meta file found at: {}'.format(
                           meta_file_location))
        self.train_file = osp.join(self.root_dir, 'class_labels_train.csv')
        self.validation_file = osp.join(self.root_dir, 'class_labels_validation.csv')
        self.test_file = osp.join(self.root_dir, 'class_labels_test.csv')

        with open(meta_file_location, 'r') as metaFile:
            metaData = metaFile.read()
            self.db_name = metaData.split(',')[0] #Name of the Dataset
            self.file_format = metaData.split(',')[1] #Image type - jpeg/png
            self.num_classes = int(metaData.split(',')[2]) #Num of classes
            self.num_train = int(metaData.split(',')[3]) #Num training points
            self.num_val = int(metaData.split(',')[4]) #Num validation points
            self.num_test = int(metaData.split(',')[5]) #Num testing points

        metaFile.close()

    def num_datapoints(self):
        if self.type == 'train':
            return self.num_train
        if self.type == 'validation':
            return self.num_val
        if self.type == 'test':
            return self.num_test

    def get_subset(self):
        if self.type == 'train':
            return self.train_file
        if self.type == 'validation':
            return self.validation_file
        if self.type == 'test':
            return self.test_file
