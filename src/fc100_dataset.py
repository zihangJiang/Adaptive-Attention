# coding=utf-8
from __future__ import print_function
import torch.utils.data as data
import numpy as np
import torch
import os
import argparse
import csv
import glob
import cv2
from shutil import copyfile
from tqdm import tqdm
from copy import deepcopy
from torchvision import transforms
from torchvision.datasets.utils import download_url, check_integrity
from PIL import Image
import pickle
'''
Inspired by https://github.com/pytorch/vision/pull/46
and
https://github.com/y2l/mini-imagenet-tools
'''

IMG_CACHE = {}
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict
def save_pickle(dicts,file):

    with open(file, 'wb') as fo:
        pickle.dump(dicts, fo)

class FC100Generator(object):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    def __init__(self, input_args, download=True):
        self.input_args = input_args
        self.image_dir = self.input_args.image_dir
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        self.data = []
        self.labels = []
        self.super_labels = []
        self.filenames = []
        for fentry in self.train_list+self.test_list:
            f = fentry[0]
            file = os.path.join(self.image_dir, self.base_folder, f)
            fo = open(file, 'rb')
            entry = pickle.load(fo, encoding='latin1')
            self.data.append(entry['data'])

            self.super_labels += entry['coarse_labels']
            self.labels += entry['fine_labels']
            self.filenames += entry['filenames']
            fo.close()
        self.data = np.concatenate(self.data)
        self.data = self.data.reshape((60000, 3, 32, 32))
        self.data = self.data.transpose((0, 2, 3, 1))
        #img = Image.fromarray(data, 'RGB')
    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.image_dir
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
    def _check_integrity(self):
        root = self.image_dir
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def process_original_files(self):
        self.processed_img_dir = '../dataset/FC100/processed_images'
        split_lists = ['train', 'val', 'test']
        super_class_split = {'train':[1, 2, 3, 4, 5, 6, 9, 10, 15, 17, 18, 19], 'val':[8, 11, 13, 16], 'test':[0,7,12,14]}
        if not os.path.exists(self.processed_img_dir):
            os.makedirs(self.processed_img_dir)
        # split data
        # idxs = {'train':[], 'val':[], 'test':[]}
        # data = {'train':[], 'val':[], 'test':[]}
        # label = {'train':[], 'val':[], 'test':[]}
        # filenames = {'train':[], 'val':[], 'test':[]}
        for idx, super_label in tqdm(enumerate(self.super_labels)):
            for stage in split_lists:
                if super_label in super_class_split[stage]:
                    file_dir = os.path.join(self.processed_img_dir,stage,str(self.labels[idx]))
                    file_path = os.path.join(file_dir,self.filenames[idx])
                    if not os.path.exists(file_dir):
                        os.makedirs(file_dir)
                    cv2.imwrite(file_path, self.data[idx])
                    # data[stage].append(self.data[idx:idx+1])
                    # label[stage].append(self.labels[idx])
                    # filenames[stage].append(self.filenames[idx])
        # train_pickle = {'data':np.concatenate(data['train']), 'label':label['train'], 'filenames':filenames['train']}
        # val_pickle = {'data':np.concatenate(data['val']), 'label':label['val'], 'filenames':filenames['val']}
        # test_pickle = {'data':np.concatenate(data['test']), 'label':label['test'], 'filenames':filenames['test']}
        # save_pickle(train_pickle, self.processed_img_dir+'/train')
        # save_pickle(val_pickle, self.processed_img_dir+'/val')
        # save_pickle(test_pickle, self.processed_img_dir+'/test')


class FC100Dataset(data.Dataset):
    processed_folder = 'processed_images'
    def __init__(self, mode='train', root='../FC100', transform=None, target_transform=None):
        '''
        The items are (filename,category). The index of all the categories can be found in self.idx_classes
        Args:
        - root: the directory where the dataset will be stored
        - transform: how to transform the input
        - target_transform: how to transform the target
        - download: need to download the dataset
        '''
        super(FC100Dataset, self).__init__()
        self.root = root
        self.transform = transform
        self.image_size = 32
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
                        transforms.RandomCrop(self.image_size, padding=4),
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
    parser.add_argument('--image_dir',  type=str, default = '../dataset/FC100', help='untar cifar dir')
    parser.add_argument('--image_resize',  type=int,  default=84)
    args = parser.parse_args()
    dataset_generator = FC100Generator(args)
    dataset_generator.process_original_files()