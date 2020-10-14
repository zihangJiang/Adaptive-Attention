# coding=utf-8
from __future__ import print_function
import torch.utils.data as data
import numpy as np
import torch
import os
import argparse
import glob
import cv2
import csv
import random
from shutil import copytree
from tqdm import tqdm
from copy import deepcopy
from torchvision import transforms
from PIL import Image
'''
Inspired by https://github.com/pytorch/vision/pull/46
and
https://github.com/y2l/mini-imagenet-tools
'''

IMG_CACHE = {}
class CUBGenerator(object):
    image_url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz'
    annos_url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200/annotations.tgz'
    def __init__(self, input_args):
        self.input_args = input_args
        if self.input_args.tar_dir is not None:
            print('Untarring CUB image package')
            self.image_dir = '../dataset/CUB/images'
            if not os.path.exists(self.image_dir):
                os.makedirs(self.image_dir)
            os.system('tar xvf ' + str(self.input_args.tar_dir) + ' -C ' + self.image_dir)
        elif self.input_args.image_dir is not None:
            self.image_dir = self.input_args.image_dir
        else:
            print('You need to specify the images.tgz source file path')


    def process_original_files(self):
        all_classes = os.listdir(self.image_dir)
        assert len(all_classes)==200
        self.processed_img_dir = '../dataset/CUB/processed_images'
        split_lists = ['train', 'val', 'test']
        split_numbers = {'train':130, 'val':20, 'test':50}
        if not os.path.exists(self.processed_img_dir):
            os.makedirs(self.processed_img_dir)
        for stage in split_lists:
            stage_dataset = set()
            if not os.path.exists(self.processed_img_dir+'/'+stage):
                os.makedirs(self.processed_img_dir+'/'+stage)
            with open('../dataset/CUB/csv_files/{}.csv'.format(stage)) as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',')
                next(csv_reader, None)
                for row in csv_reader:
                    stage_dataset.add(row[1])
                assert len(stage_dataset) == split_numbers[stage]
            for class_name in tqdm(stage_dataset):
                class_dir = self.image_dir+'/'+class_name
                copytree(class_dir, self.processed_img_dir+'/'+stage+'/'+class_name)



        #sorted(glob.glob(self.image_dir+'/*'))
        #random.shuffle(all_classes)
        # classes = {}
        # classes['train'] = all_classes[:130]
        # classes['val'] = all_classes[130:150]
        # classes['test'] = all_classes[150:]
        # for stage in split_lists:
        #     print('generating {} dataset'.format(stage))
        #     for class_dir in tqdm(classes[stage]):
        #         class_name = class_dir.split('/')[-1]
        #         copytree(class_dir, self.processed_img_dir+'/'+stage+'/'+class_name)


class CUBDataset(data.Dataset):
    raw_folder = 'images'
    processed_folder = 'processed_images'
    def __init__(self, mode='train', root='../CUB', transform=None, target_transform=None):
        '''
        The items are (filename,category). The index of all the categories can be found in self.idx_classes
        Args:
        - root: the directory where the dataset will be stored
        - transform: how to transform the input
        - target_transform: how to transform the target
        - download: need to download the dataset
        '''
        super(CUBDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.image_size = 84
        self.mode = mode
        if transform == None:
            if not self.mode=='train':
                self.transform = transforms.Compose([
                    transforms.Resize(self.image_size),
                    transforms.CenterCrop(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]],
                                         std=[x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]])
                ])
            else:
                self.transform = transforms.Compose([
                        transforms.Resize(self.image_size),
                        transforms.RandomCrop(self.image_size, padding=8),
                        transforms.RandomHorizontalFlip(),
                        lambda x: np.asarray(x),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]],
                                             std=[x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]])
                ])
        self.target_transform = target_transform


        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        self.classes = sorted(os.listdir(os.path.join(self.root, self.processed_folder, mode)))
        self.all_items = find_items(os.path.join(
            self.root, self.processed_folder, mode), self.classes)

        self.idx_classes = index_classes(self.all_items)
        paths, self.y = zip(*[self.get_path_label(pl)
                              for pl in range(len(self))])
        self.x = paths

    def __getitem__(self, idx):
        file_path = self.x[idx]
        # if self.image_size > 100:
            # switch image size>100 to use higher resolution image

        x = Image.open(file_path).convert('RGB')
        #x = self.x[idx]
        if self.transform:
            x = self.transform(deepcopy(x))
        return x, self.y[idx]

    def __len__(self):
        return len(self.all_items)

    def switch_image_size(self, size = 0):
        if self.image_size == 84:
            self.image_size = 224
        else:
            self.image_size = 84
        if size>0:
            self.image_size = size
        if not self.mode=='train':
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]],
                                     std=[x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]])
            ])
        else:
            self.transform = transforms.Compose([
                    transforms.Resize(self.image_size),
                    #transforms.Pad(16,padding_mode='reflect'),
                    transforms.RandomCrop(self.image_size, padding=8),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]],
                                         std=[x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]])
            ])

    def get_path_label(self, index):
        filename = self.all_items[index][0]
        img = str.join('/', [self.all_items[index][2], filename])
        target = self.idx_classes[self.all_items[index][1]]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder))


def find_items(root_dir, classes):
    retour = []
    for (root, dirs, files) in sorted(os.walk(root_dir)):
        for f in sorted(files):
            r = root.split('/')
            lr = len(r)
            label = r[lr - 1]
            
            if label in classes and (f.endswith("jpg")):
                retour.extend([(f, label, root)])
    print("== Dataset: Found %d items " % len(retour))
    return retour


def index_classes(items):
    idx = {}
    for i in items:
        if (not i[1] in idx):
            idx[i[1]] = len(idx)
    print("== Dataset: Found %d classes" % len(idx))
    return idx


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--tar_dir',  type=str, help='tar CUB dir')
    parser.add_argument('--image_dir',  type=str, default = '../dataset/CUB/images', help='untar cub dir')
    args = parser.parse_args()
    dataset_generator = CUBGenerator(args)
    dataset_generator.process_original_files()