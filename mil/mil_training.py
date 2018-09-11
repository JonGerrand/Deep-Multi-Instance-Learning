"""
    @Description:       This module is responsible for setting up and launching
                        MIL training. 

    @Author:            Jonathan Gerrand
"""

from input_pipeline import InputPipeline
from dataset import Dataset
from training_spec import TrainingSpec

import argparse
import training
import models

def launch_training(model_name,
                    train_data_dir,
                    model_data_dir,
                    checkpoint_dir,
                    reinit_learn_rate,
                    fine_tune_learn_rate,
                    num_epochs,
                    optimiser,
                    weight_decay,
                    init_type,
                    train_type,
                    val_metric,
                    batch_size,
                    raw_img_len,
                    model_img_len,
                    crop_step_size,
                    patch_type,
                    val_thresh):

    train_spec = TrainingSpec(batch_size=batch_size,
                              num_epochs=num_epochs,
                              reinit_learn_rate=reinit_learn_rate,
                              fine_tune_learn_rate=fine_tune_learn_rate,
                              optimiser=optimiser,
                              val_metric=val_metric,
                              init_type=init_type,
                              training_type=train_type,
                              weight_decay=weight_decay,
                              raw_img_len=raw_img_len,
                              model_img_len=model_img_len,
                              crop_step_size=crop_step_size,
                              patch_type=patch_type,
                              val_frequency=0.20,
                              val_thresh=val_thresh)

    data_spec = Dataset(root_dir=train_data_dir,
                        dataset_type='train')

    model_spec = models.ModelSpec(re_initilised_layers=['logits'])

    print ('')
    print ('=' * 60)
    print ('Batch size: ', train_spec.batch_size)
    print ('Data Path: ', data_spec.train_file)

    # Setup model values
    net, logits, in_imgs = training.setup_model_params(model_name=model_name,
                                                       train_spec=train_spec,
                                                       data_spec=data_spec)

    train_l_holder, val_l_holder, loss_holder = training.setup_placeholders(
                                                        data_spec=data_spec)

    reinit_params, fine_tune_params = training.factor_training_layers(model_spec)
    model_spec.get_factorised_layers(reinit_params=reinit_params,
                                     fine_tune_params=fine_tune_params)

    train_loss, val_op, train_op = training.create_training_ops(logits=logits,
                                                train_labels=train_l_holder,
                                                val_labels=val_l_holder,
                                                loss_placehldr=loss_holder,
                                                train_spec=train_spec,
                                                model_spec=model_spec,
                                                dataset_spec=data_spec)

    # Setup data pipelines
    t_pipeline = InputPipeline(training_spec=train_spec,
                               dataset_spec=data_spec,
                               pipeline_type=train_type,
                               patch_type=patch_type)
    v_pipeline = InputPipeline(training_spec=train_spec,
                               dataset_spec=data_spec,
                               pipeline_type=train_type + '_val',
                               patch_type=patch_type)
    t_img_queue, t_label_queue = t_pipeline.generate_input_batches()
    v_img_queue, v_label_queue = v_pipeline.generate_input_batches()

    # Commence training
    training.train_model(model=net,
                        train_op=train_op,
                        train_loss=train_loss,
                        val_op=val_op,
                        img_placehldr=in_imgs,
                        t_lab_placehldr=train_l_holder,
                        v_lab_placehldr=val_l_holder,
                        train_img_bags=t_img_queue,
                        train_labels=t_label_queue,
                        val_img_bags=v_img_queue,
                        val_labels=v_label_queue,
                        data_spec=data_spec,
                        train_spec=train_spec,
                        checkpoint_path=checkpoint_dir,
                        model_data_path=model_data_dir)

def main():
    # Parse all input arguments
    parser = argparse.ArgumentParser(description='Launch MIL training for \
                                                  a given model')

    parser.add_argument('--model_name',
                        default='GoogleNet',
                        help='Default: GoogleNet. \
                        (Options: GoogleNet, NIN)')

    parser.add_argument('--train_data_path',
                        default='',
                        help='Default: ""')

    parser.add_argument('--model_data_path',
                        default='data/models/googlenet/GoogleNet.npy',
                        help='Default: data/models/googlenet/GoogleNet.npy')

    parser.add_argument('--checkpoint_path',
                        default='',
                        help='Default: ""')

    parser.add_argument('--train_type',
                        default='stage_1',
                        help='Default: stage_1\
                        (Options: stage_1, stage_2)')

    parser.add_argument('--init_type',
                        default='Fine-tune',
                        help='Default: Fine-tune\
                        (Options: Fine-tune, Multi-fine-tune, Raw, Restore)')

    parser.add_argument('--optimiser',
                        default='RMSProp',
                        help='Default is: RMSProp. '\
                        '(Options: RMSProp, Adam, SGD)')

    parser.add_argument('--reinit_learn_rate',
                        default=0.01,
                        type=float,
                        help='Default: 0.01')

    parser.add_argument('--fine_tune_learn_rate',
                        default=0.001,
                        type=float,
                        help='Default: 0.001')

    parser.add_argument('--batch_size',
                        default=8,
                        type=int,
                        help='Default: 8')

    parser.add_argument('--num_epochs',
                        default=50,
                        type=int,
                        help='Default: 50')

    parser.add_argument('--weight_decay',
                        default=0.0001,
                        type=float,
                        help='Default: 0.0001')

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

    parser.add_argument('--val_metric',
                        default='loss',
                        help='Default: loss '\
                        '(Options: score, loss)')

    parser.add_argument('--patch_type',
                        default='grid',
                        help='Default: grid '\
                        '(Options: grid, grid_cent, horizontal, vertical)')

    parser.add_argument('--val_thresh',
                        default=0.5,
                        type=float,
                        help='Default: 0.5')

    args = parser.parse_args()

    launch_training(model_name=args.model_name,
                    train_data_dir=args.train_data_path,
                    model_data_dir=args.model_data_path,
                    checkpoint_dir=args.checkpoint_path,
                    reinit_learn_rate=args.reinit_learn_rate,
                    fine_tune_learn_rate=args.fine_tune_learn_rate,
                    num_epochs=args.num_epochs,
                    optimiser=args.optimiser,
                    weight_decay=args.weight_decay,
                    init_type=args.init_type,
                    train_type=args.train_type,
                    val_metric=args.val_metric,
                    batch_size=args.batch_size,
                    raw_img_len=args.raw_img_len,
                    model_img_len=args.model_img_len,
                    crop_step_size=args.crop_step_size,
                    patch_type=args.patch_type,
                    val_thresh=args.val_thresh)

if __name__ == '__main__':
    main()
