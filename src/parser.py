# coding=utf-8
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    
    # str config of model
    parser.add_argument('-data', '--dataset',
                        type=str,
                        help='dataset to use',
                        default='MiniImagenet')
    
    parser.add_argument('-root', '--dataset_root',
                        type=str,
                        help='path to dataset',
                        default='../dataset/MiniImagenet')

    parser.add_argument('-exp', '--experiment_root',
                        type=str,
                        help='root where to store models, losses and accuracies',
                        default='../output')
    parser.add_argument('-fm', '--feature_mode',
                        type=str,
                        help='use weighted or unweighted or cat feature',
                        default='unweighted')
    parser.add_argument('-stg', '--stage',
                        type=str,
                        help='train or test',
                        default='train')
    
    # numerical config of model
    parser.add_argument('-nep', '--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=200)

    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.001',
                        default=0.001)
    
    parser.add_argument('-wdcay', '--weight_decay',
                        type=float,
                        help='weight penalty for the model, default=0.0005',
                        default=5e-4)
    parser.add_argument('-lrS', '--lr_scheduler_step',
                        type=int,
                        help='StepLR learning rate scheduler step, default=20',
                        default=25)

    parser.add_argument('-lrG', '--lr_scheduler_gamma',
                        type=float,
                        help='StepLR learning rate scheduler gamma, default=0.5',
                        default=0.7)

    parser.add_argument('-its', '--iterations',
                        type=int,
                        help='number of episodes per epoch, default=10',
                        default=400)

    parser.add_argument('-cTr', '--classes_per_it_tr',
                        type=int,
                        help='number of random classes per episode for training, default=60',
                        default=10)

    parser.add_argument('-nsTr', '--num_support_tr',
                        type=int,
                        help='number of samples per class to use as support for training, default=5',
                        default=1)

    parser.add_argument('-cVa', '--classes_per_it_val',
                        type=int,
                        help='number of random classes per episode for validation, default=5',
                        default=5)

    parser.add_argument('-nsVa', '--num_support_val',
                        type=int,
                        help='number of samples per class to use as support for validation, default=5',
                        default=1)

    parser.add_argument('-btz', '--batch_size',
                        type=int,
                        help='number of per training iteration for querying, default=1',
                        default=150)
    parser.add_argument('-seed', '--manual_seed',
                        type=int,
                        help='input for the manual seeds initializations',
                        default=7)
    parser.add_argument('-a', '--augment',
                        type=int,
                        help='intensity of data augmentation',
                        default=1)

    parser.add_argument('-g', '--gpu',
                        help='which gpu to use',
                        default='0')
    

    
    # bool config of model
    parser.add_argument('--cuda',
                        action='store_false',
                        help='enables cuda')
    parser.add_argument('-l',dest = 'load',
                        action='store_true',
                        help='load pretrained model')
    parser.add_argument('-r',dest = 'regular',
                        action='store_true',
                        help='load pretrained model')
    parser.add_argument('-rela',dest = 'relation',
                        action='store_true',
                        help='use relation network')
    parser.add_argument('-swt',dest = 'switch',
                        action='store_true',
                        help='switch image size per epoch (default between 100 and 84)')
    parser.add_argument('-spl',dest = 'simplest',
                        action='store_true',
                        help='use simplest form of attention module')
    parser.add_argument('-scl',dest = 'scale',
                        action='store_false',
                        help='use scale (temperature parameter proposed in TADAM)')
    parser.add_argument('-res',dest = 'resnet',
                        action='store_true',
                        help='use resnet-12 like feature extractor')
    parser.add_argument('-d',dest = 'double',
                        action='store_true',
                        help='double reweighting module')
    parser.add_argument('-up',dest = 'use_perm',
                        action='store_true',
                        help='use perm loss')
    parser.add_argument('-uic',dest = 'use_inter_class',
                        action='store_true',
                        help='use inter class loss')
    parser.add_argument('-tune',dest = 'fine_tune',
                        action='store_true',
                        help='fine tune while testing')
    parser.add_argument('-ta',dest = 'test_augmentation',
                        action='store_false',
                        help='augment support set while testing')
    parser.add_argument('-vis',dest = 'visulize',
                        action='store_true',
                        help='visulize attention map')
    parser.add_argument('-proto',dest = 'prototypical',
                        action='store_true',
                        help='train or test baseline prototypical network')
    return parser
if __name__ == '__main__':
    options = get_parser().parse_args()
