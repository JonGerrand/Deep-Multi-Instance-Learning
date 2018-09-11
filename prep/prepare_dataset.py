import os
import csv
# from shutil import move
from shutil import copyfile
import argparse
import numpy as np
import math
import subprocess

def get_lowest_volume(root_dir,ex_classes=[]):
    # Argument validation
    if os.path.isdir(root_dir) is not True:
        raise IOError("Invalid root dir: {}".format(root_dir))

    min_data_points = np.inf
    for class_folder in os.listdir(root_dir):
        if class_folder not in ex_classes:
            if len(os.listdir(os.path.join(root_dir,class_folder))) < \
                    min_data_points:
                min_data_points = len(os.listdir(os.path.join(root_dir,
                                                          class_folder)))

    return min_data_points

def subsample_class(className, sourceDir, targetDir, num_samples):
    # Gather all files in class Dir
    fileQueue = []
    for file in os.listdir(os.path.join(sourceDir, className)):
        fileQueue.append(file)

    # Shuffle file Queue
    np.random.shuffle(fileQueue)

    # Draw samples until the sub-sample limit

    for i in xrange(0, num_samples):
        try:
            # Thrown for smallest class
            fileQueue[i]
        except IndexError:
            break

        copyfile(os.path.join(sourceDir, className, fileQueue[i]),
                 os.path.join(targetDir, className, fileQueue[i]))

def write_class_csv_files(root_dir, class_labels):
    for dir in os.listdir(root_dir):
        if dir != 'raw' and dir != 'meta_data.txt':
            with open(os.path.join(root_dir,
                'class_labels_{}.csv'.format(dir)),'wb') as csvfile:
                field_names = ['class', 'path']
                writer = csv.DictWriter(csvfile, fieldnames=field_names)
                writer.writeheader()
                for sub_dir in os.listdir(os.path.join(root_dir,dir)):
                    for file in os.listdir(os.path.join(root_dir, dir, sub_dir)):
                        writer.writerow({'class': class_labels.index(sub_dir),
                         'path': os.path.join(root_dir, dir, sub_dir, file)})

def produce_meta_file(root_dir, datasplits, DB_name, stage='stage_1'):
    raw_dir = os.path.join(root_dir,'raw')
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'validation')
    test_dir = os.path.join(root_dir, 'test')

    DB_name = DB_name
    num_classes = len(os.listdir(train_dir))
    num_train = 0
    for data_class in os.listdir(train_dir):
        for data_file in os.listdir(os.path.join(train_dir,data_class)):
            num_train += 1

    if stage == 'stage_1':
        num_val = 0
        for data_class in os.listdir(val_dir):
            for data_file in os.listdir(os.path.join(val_dir,data_class)):
                num_val += 1

        num_test = 0
        for data_class in os.listdir(test_dir):
            for data_file in os.listdir(os.path.join(test_dir,data_class)):
                num_test += 1

    if stage == 'stage_2':
        num_test = 0
        parent_dir = os.path.abspath("{}".format(os.path.join(
                                            root_dir, os.pardir)))
        parent_val_dir = os.path.join(parent_dir, 'validation')
        num_val = 0
        for data_class in os.listdir(parent_val_dir):
            for data_file in os.listdir(os.path.join(parent_val_dir,data_class)):
                num_val += 1

    # Write the meta file
    with open(os.path.join(root_dir, 'meta_data.txt'),'w') as txtFile:
        txtFile.write("{},{},{},{},{},{}".format(DB_name, 'jpeg', num_classes,
                      num_train, num_val, num_test))

    pass

def splitDatainDir(className, baseDir, trainDir, valDir, testDir, datasplits):
    fileQueue = []
    for file in os.listdir(os.path.join(baseDir, className)):
        fileQueue.append(file)

    # Shuffle file Queue
    train_split = math.floor(len(fileQueue) * datasplits[0])
    val_split = math.floor(len(fileQueue) * datasplits[1])
    test_split = len(fileQueue) - train_split - val_split
    np.random.shuffle(fileQueue)

    # Create new Dir
    cmd = "mkdir {}".format(os.path.join(trainDir, className))
    subprocess.call(cmd, shell=True)

    cmd = "mkdir {}".format(os.path.join(valDir, className))
    subprocess.call(cmd, shell=True)

    cmd = "mkdir {}".format(os.path.join(testDir, className))
    subprocess.call(cmd, shell=True)

    # Move sets of data around
    for j in range(int(train_split)):
        copyfile(os.path.join(baseDir, className, fileQueue[j]), os.path.join(trainDir, className, fileQueue[j]))

    for j in range(int(train_split + 1), int(train_split + val_split)):
        copyfile(os.path.join(baseDir, className, fileQueue[j]), os.path.join(valDir, className, fileQueue[j]))

    for j in range(int(train_split + val_split + 1), int(train_split + val_split + test_split)):
        copyfile(os.path.join(baseDir, className, fileQueue[j]), os.path.join(testDir, className, fileQueue[j]))

    pass

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir',
                        default="data/dataset")

    parser.add_argument('--dataset_name',
                        default="dataset")

    parser.add_argument('--stage',
            default='stage_1',
            help="""
            Dataset preparation for either 'stage_1' or 'stage_2'
            of MIL. Default = 'stage_1'
            """)

    parser.add_argument('--split_ratio',
            default="0.6,0.2,0.2",
            help="""
            Comma-seperated string of Train, Val, Test data
            splits. (Default: 0.6,0.2,0.2)
            """)

    args = parser.parse_args()

    raw_dir = os.path.join(args.root_dir,'raw')
    train_dir = os.path.join(args.root_dir,'train')
    val_dir = os.path.join(args.root_dir,'validation')
    test_dir = os.path.join(args.root_dir,'test')
    data_splits = [float(ratio) for ratio in args.split_ratio.split(',')]

    if args.stage == 'stage_1':
        for subset in ['train', 'validation', 'test']:
            cmd = "mkdir {}".format(os.path.join(args.root_dir,subset))
            subprocess.call(cmd, shell=True)

        for img_class in os.listdir(raw_dir):
            print ("Splitting class '{}'".format(img_class))
            splitDatainDir(img_class, raw_dir, train_dir, val_dir, test_dir, data_splits)

        write_class_csv_files(args.root_dir, ['d_0', 'd_1'])

        produce_meta_file(args.root_dir, data_splits, args.dataset_name,
                          stage='stage_1')

    if args.stage == 'stage_2':
        sub_number = get_lowest_volume(raw_dir)
        cmd = "mkdir {}".format(os.path.join(args.root_dir,'train'))
        subprocess.call(cmd, shell=True)
        save_dir = os.path.join(args.root_dir,'train')

        for img_class in os.listdir(raw_dir):
            cmd = "mkdir {}".format(os.path.join(save_dir, img_class))
            subprocess.call(cmd, shell=True)
            print ("Subsampling class {}".format(img_class))
            subsample_class(img_class, raw_dir, save_dir, sub_number)

        write_class_csv_files(args.root_dir, ['d_0', 'd_1', 'd_n'])

        parent_dir = os.path.abspath("{}".format(os.path.join(
                                                 args.root_dir,
                                                 os.pardir
                                                 )))
        copyfile(os.path.join(parent_dir, 'class_labels_validation.csv'),
                 os.path.join(args.root_dir, 'class_labels_validation.csv'))

        produce_meta_file(args.root_dir, data_splits, args.dataset_name,
                          stage='stage_2')

    # Write code



if __name__ == '__main__':
    main()
