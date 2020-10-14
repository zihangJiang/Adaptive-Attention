# coding=utf-8
import logging
import datetime
import numpy as np
import torch
import os, random
from omniglot_dataset import OmniglotDataset
from cub_dataset import CUBDataset
from fc100_dataset import FC100Dataset
from car_dataset import StanfordCarDataset
from dog_dataset import StanfordDogDataset
from mini_imagenet_dataset import MiniImagenetDataset, AugmentedMiniImagenetDataset
from reweighter_batch_sampler import ReweighterBatchSampler, SpecialReweighterBatchSampler
from cfrnet import CRFNet, ProtoNet, RelationNet
default_device = 'cuda:0'

def init_log_file(opt, prefix = 'LOG_INFO_'):
    logger = logging.getLogger(__name__)

    strHandler = logging.StreamHandler()
    formatter = logging.Formatter(
            '%(asctime)s - %(name)-8s - %(levelname)-6s - %(message)s')
    strHandler.setFormatter(formatter)
    logger.addHandler(strHandler)
    logger.setLevel(logging.INFO)

    log_dir = os.path.join(opt.experiment_root, 'logs')
    if (not os.path.isdir(log_dir)):
        os.makedirs(log_dir)

    now_str = datetime.datetime.now().__str__().replace(' ','_')

    log_file = os.path.join(log_dir, prefix+now_str+'.txt')
    log_fileHandler = logging.FileHandler(log_file)
    log_fileHandler.setFormatter(formatter)
    logger.addHandler(log_fileHandler)
    
    return logger
    
def init_seed(opt):
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
def init_dataset(opt, mode):
    if 'train' in mode:
        mode = 'train'
    if opt.dataset == 'Omniglot':
        if 'Omniglot' not in opt.dataset_root:
            print('automatically switch to Omniglot dataset')
            opt.dataset_root = '../dataset/Omniglot'
        dataset = OmniglotDataset(mode=mode, root=opt.dataset_root)
    elif opt.dataset == 'MiniImagenet':
        if opt.augment == 1:
            dataset = MiniImagenetDataset(mode=mode, root=opt.dataset_root)
        else:
            dataset = AugmentedMiniImagenetDataset(mode=mode, root=opt.dataset_root)
    elif opt.dataset == 'CUB':
        if 'CUB' not in opt.dataset_root:
            print('automatically switch to CUB dataset')
            opt.dataset_root = '../dataset/CUB'
        dataset = CUBDataset(mode=mode, root=opt.dataset_root)
    elif opt.dataset == 'StanfordDog':
        if 'StanfordDog' not in opt.dataset_root:
            print('automatically switch to StanfordDog dataset')
            opt.dataset_root = '../dataset/StanfordDog'
        dataset = StanfordDogDataset(mode=mode, root=opt.dataset_root)
    elif opt.dataset == 'StanfordCar':
        if 'StanfordCar' not in opt.dataset_root:
            print('automatically switch to StanfordCar dataset')
            opt.dataset_root = '../dataset/StanfordCar'
        dataset = StanfordCarDataset(mode=mode, root=opt.dataset_root)
    elif opt.dataset == 'FC100':
        if 'FC100' not in opt.dataset_root:
            print('automatically switch to FC100 dataset')
            opt.dataset_root = '../dataset/FC100'
        dataset = FC100Dataset(mode=mode, root=opt.dataset_root)

    n_classes = len(np.unique(dataset.y))
    if (n_classes < opt.classes_per_it_tr and mode=='train') or (n_classes < opt.classes_per_it_val and not mode=='train'):
        raise(Exception('There are not enough classes in the dataset in order ' +
                        'to satisfy the chosen classes_per_it. Decrease the ' +
                        'classes_per_it_{tr/val} option and try again.'))
    return dataset

def init_sampler(opt, labels, mode):
    if mode=='train' :
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.batch_size
        spc = opt.num_support_tr
        iters = opt.iterations
    else:
        classes_per_it = opt.classes_per_it_val
        if opt.stage=='train':
            num_samples = opt.batch_size*classes_per_it//opt.classes_per_it_tr
        else:
            num_samples = opt.batch_size
        spc = opt.num_support_val
        if opt.stage=='train':
            iters = opt.iterations*2
        else:
            iters = opt.iterations

    return ReweighterBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=iters,
                                    sample_per_class = spc,
                                    regular = opt.regular)

def init_dataloader(opt, mode, agument = False):
    dataset = init_dataset(opt, mode)
    sampler = init_sampler(opt, dataset.y, mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler,num_workers=3)
    return dataloader


def init_crfnet(opt):
    '''
    Initialize the crfNet
    '''
    device = default_device if torch.cuda.is_available() and opt.cuda else 'cpu'
    model = CRFNet(opt = opt).to(device)
    return model

def init_protonet(opt):
    '''
    Initialize the crfNet
    '''
    device = default_device if torch.cuda.is_available() and opt.cuda else 'cpu'
    model = ProtoNet(opt = opt).to(device)
    return model
def init_relationnet(opt):
    '''
    Initialize the crfNet
    '''
    device = default_device if torch.cuda.is_available() and opt.cuda else 'cpu'
    model = RelationNet(opt = opt).to(device)
    return model
def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate, weight_decay=opt.weight_decay)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)