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
from PIL import Image
'''
Inspired by https://github.com/pytorch/vision/pull/46
and
https://github.com/y2l/mini-imagenet-tools
'''

IMG_CACHE = {}
class MiniImageNetGenerator(object):
    def __init__(self, input_args):
        self.input_args = input_args
        if self.input_args.tar_dir is not None:
            print('Untarring ILSVRC2012 package')
            self.imagenet_dir = '../dataset/Imagenet'
            if not os.path.exists(self.imagenet_dir):
                os.mkdir(self.imagenet_dir)
            os.system('tar xvf ' + str(self.input_args.tar_dir) + ' -C ' + self.imagenet_dir)
        elif self.input_args.imagenet_dir is not None:
            self.imagenet_dir = self.input_args.imagenet_dir
        else:
            print('You need to specify the ILSVRC2012 source file path')
        self.mini_dir = '../dataset/MiniImagenet/raw'
        if not os.path.exists(self.mini_dir):
            os.mkdir(self.mini_dir)
        self.image_resize = self.input_args.image_resize
        
    def untar_mini(self):
        self.mini_keys = ['n02110341', 'n01930112', 'n04509417', 'n04067472', 'n04515003', 'n02120079', 'n03924679', 'n02687172', 'n03075370', 'n07747607', 'n09246464', 'n02457408', 'n04418357', 'n03535780', 'n04435653', 'n03207743', 'n04251144', 'n03062245', 'n02174001', 'n07613480', 'n03998194', 'n02074367', 'n04146614', 'n04243546', 'n03854065', 'n03838899', 'n02871525', 'n03544143', 'n02108089', 'n13133613', 'n03676483', 'n03337140', 'n03272010', 'n01770081', 'n09256479', 'n02091244', 'n02116738', 'n04275548', 'n03773504', 'n02606052', 'n03146219', 'n04149813', 'n07697537', 'n02823428', 'n02089867', 'n03017168', 'n01704323', 'n01532829', 'n03047690', 'n03775546', 'n01843383', 'n02971356', 'n13054560', 'n02108551', 'n02101006', 'n03417042', 'n04612504', 'n01558993', 'n04522168', 'n02795169', 'n06794110', 'n01855672', 'n04258138', 'n02110063', 'n07584110', 'n02091831', 'n03584254', 'n03888605', 'n02113712', 'n03980874', 'n02219486', 'n02138441', 'n02165456', 'n02108915', 'n03770439', 'n01981276', 'n03220513', 'n02099601', 'n02747177', 'n01749939', 'n03476684', 'n02105505', 'n02950826', 'n04389033', 'n03347037', 'n02966193', 'n03127925', 'n03400231', 'n04296562', 'n03527444', 'n04443257', 'n02443484', 'n02114548', 'n04604644', 'n01910747', 'n04596742', 'n02111277', 'n03908618', 'n02129165', 'n02981792']
        
        for idx, keys in enumerate(self.mini_keys):
            print('Untarring ' + keys)
            os.system('tar xvf ' + self.imagenet_dir + '/' + keys + '.tar -C ' + self.mini_dir)
        print('All the tar files are untarred')

    def process_original_files(self):
        self.processed_img_dir = '../dataset/MiniImagenet/processed_images'
        if not self.image_resize==84:
            self.processed_img_dir = '../dataset/MiniImagenet/processed_images_{}'.format(self.image_resize)
        split_lists = ['train', 'val', 'test']
        csv_files = ['../dataset/MiniImagenet/csv_files/train.csv','../dataset/MiniImagenet/csv_files/val.csv', '../dataset/MiniImagenet/csv_files/test.csv']

        if not os.path.exists(self.processed_img_dir):
            os.makedirs(self.processed_img_dir)

        for this_split in split_lists:
            filename = '../dataset/MiniImagenet/csv_files/' + this_split + '.csv'
            this_split_dir = self.processed_img_dir + '/' + this_split
            if not os.path.exists(this_split_dir):
                os.makedirs(this_split_dir)
            with open(filename) as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',')
                next(csv_reader, None)
                images = {}
                print('Reading IDs....')

                for row in tqdm(csv_reader):
                    if row[1] in images.keys():
                        images[row[1]].append(row[0])
                    else:
                        images[row[1]] = [row[0]]

                print('Writing photos....')
                for cls in tqdm(images.keys()):
                    this_cls_dir = this_split_dir + '/' + cls        
                    if not os.path.exists(this_cls_dir):
                        os.makedirs(this_cls_dir)

                    lst_files = []
                    for file in glob.glob(self.mini_dir + "/*"+cls+"*"):
                        lst_files.append(file)

                    lst_index = [int(i[i.rfind('_')+1:i.rfind('.')]) for i in lst_files]
                    index_sorted = sorted(range(len(lst_index)), key=lst_index.__getitem__)

                    index_selected = [int(i[i.index('.') - 4:i.index('.')]) for i in images[cls]]
                    selected_images = np.array(index_sorted)[np.array(index_selected) - 1]
                    for i in np.arange(len(selected_images)):
                        if self.image_resize==0:
                            copyfile(lst_files[selected_images[i]],os.path.join(this_cls_dir, images[cls][i]))
                        else:
                            im = cv2.imread(lst_files[selected_images[i]])
                            im_resized = cv2.resize(im, (self.image_resize, self.image_resize), interpolation=cv2.INTER_AREA)
                            cv2.imwrite(os.path.join(this_cls_dir, images[cls][i]),im_resized)


class MiniImagenetDataset(data.Dataset):
    splits_folder = os.path.join('csv_files')
    raw_folder = 'raw'
    processed_folder = 'processed_images'
    def __init__(self, mode='train', root='../MiniImagenet', transform=None, target_transform=None):
        '''
        The items are (filename,category). The index of all the categories can be found in self.idx_classes
        Args:
        - root: the directory where the dataset will be stored
        - transform: how to transform the input
        - target_transform: how to transform the target
        - download: need to download the dataset
        '''
        super(MiniImagenetDataset, self).__init__()
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
        if self.image_size > 100:
            # switch image size>100 to use higher resolution image
            file_path = transform_filename(file_path, self.image_size)
        x = Image.open(file_path).convert('RGB')
        #x = self.x[idx]
        if self.transform:
            x = self.transform(deepcopy(x))
        return x, self.y[idx]

    def __len__(self):
        return len(self.all_items)

    def switch_image_size(self, size = 0):
        if self.image_size == 84:
            # default change to 224
            self.image_size = 224
        else:
            # switch back to 84
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

def transform_filename(filename, image_size):
    return filename.replace(MiniImagenetDataset.processed_folder, MiniImagenetDataset.processed_folder+'_'+str(image_size))

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


class AugmentedMiniImagenetDataset(MiniImagenetDataset):
    def __init__(self, mode='train', root='../MiniImagenet', transform=None, target_transform=None):
        '''
        The items are (filename,category). The index of all the categories can be found in self.idx_classes
        Args:
        - root: the directory where the dataset will be stored
        - transform: how to transform the input
        - target_transform: how to transform the target
        - download: need to download the dataset
        '''
        super(AugmentedMiniImagenetDataset, self).__init__(mode=mode, root=root, transform=transform, target_transform=target_transform)
        from torchvision import transforms
        self.transform_sets = [
        transforms.Compose([transforms.Resize(self.image_size)]),
        transforms.Compose([transforms.Resize(self.image_size+16), transforms.CenterCrop(self.image_size)]),
        transforms.Compose([transforms.Resize(self.image_size), transforms.RandomCrop(self.image_size, padding=8)]),
        transforms.Compose([transforms.Resize(self.image_size), transforms.RandomHorizontalFlip(1)]),
        ]
        self.transform = transforms.Compose([
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                transforms.Normalize(mean=[x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]],
                                     std=[x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]])
        ])
    def __getitem__(self, idx):
        x = Image.open(self.x[idx]).convert('RGB')

        items = torch.stack([self.transform(trans(deepcopy(x))) for trans in self.transform_sets])
        return items ,self.y[idx]

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--tar_dir',  type=str, help='tar imagenet dir (not recommended)')
    parser.add_argument('--imagenet_dir',  type=str, default = '../dataset/ILSVRC2012_img_train', help='untar imagenet dir (recommended)')
    parser.add_argument('--image_resize',  type=int,  default=84)
    args = parser.parse_args()
    dataset_generator = MiniImageNetGenerator(args)
    dataset_generator.untar_mini()
    dataset_generator.process_original_files()