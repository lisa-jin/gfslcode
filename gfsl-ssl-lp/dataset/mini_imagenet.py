import os
import pickle
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import scipy.io as sio

import csv, torchvision, random, os

from torch.utils.data import Sampler, Dataset, DataLoader, BatchSampler, SequentialSampler, RandomSampler, Subset
from torchvision import transforms, datasets
from collections import defaultdict

class ImageNet(Dataset):
    def __init__(self, args, partition='train', transform=None):
        super(Dataset, self).__init__()
        self.args = args
        self.data_root = args.data_root
        self.partition = partition
        self.data_aug = args.data_aug
        self.partion = args.partion
        self.mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        self.std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        # self.pretrain = pretrain

        if transform is None:
            if self.partition == 'train' and self.data_aug:
                self.transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.RandomCrop(84, padding=8),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
            else:
                self.transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
        else:
            self.transform = transform

        if self.partition=='train' or self.partition=='trainval':
            self.file_pattern = 'miniImageNet_category_split_train_phase_%s.pickle'
        else:
            self.file_pattern = 'miniImageNet_category_split_%s.pickle'

        self.data = {}
        with open(os.path.join(self.data_root, self.file_pattern % partition), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            images = data['data']
            labels = data['labels']

            if self.partition == 'train' or self.partition =='trainval' or self.partition == 'auxtest':

                unique_lab = np.unique(labels)
                select_index = []
                is_first = True
                for i in range(len(unique_lab)):
                    idx = (np.array(labels) == i).nonzero()[0]
                    select_num = round(len(idx) * self.partion)   ###
                    selected_ins = np.random.choice(idx, select_num, False)

                    if is_first:
                        select_index = selected_ins
                        is_first = False
                    else:
                        select_index = np.hstack((select_index, selected_ins))

                self.imgs = images[select_index]
                self.labels = np.array(labels)[select_index]

            else:
                self.imgs = images
                self.labels = labels

    def transform_sample(self, img, indx=None):
        if indx is not None:
            out = transforms.functional.resized_crop(img, indx[0], indx[1], indx[2], indx[3], (84, 84))
        else:
            out = img
        out = self.color_transform(out)
        out = transforms.RandomHorizontalFlip()(out)
        out = transforms.functional.to_tensor(out)
        out = self.normalize(out)
        return out

    def __getitem__(self, item):
        img = np.asarray(self.imgs[item]).astype('uint8')
        if self.args.method == 'in-eq':
            if self.partition == 'train':
                img = transforms.RandomCrop(84, padding=8)(Image.fromarray(img))
            else:
                img = Image.fromarray(img)

            img2 = self.transform_sample(img, [np.random.randint(28), 0, 56, 84])
            img3 = self.transform_sample(img, [0, np.random.randint(28), 84, 56])
            img4 = self.transform_sample(img, [np.random.randint(28), np.random.randint(28), 56, 56])

            if self.partition == 'train':
                img = self.transform_sample(img)
            else:
                img = transforms.functional.to_tensor(img)
                img = self.normalize(img)
            target = self.labels[item] - min(self.labels)
            return img, img2, img3, img4, target, item
        else:
            img = self.transform(img)
            target = self.labels[item] - min(self.labels)
            return img, target
        
    def __len__(self):
        return len(self.labels)


class MetaBaseImageNettrain(ImageNet):
    def __init__(self, args, partition='train', train_transform=None, test_transform=None, fix_seed=True):
        super(MetaBaseImageNettrain, self).__init__(args, partition, False)
        self.fix_seed = fix_seed
        self.n_shots = 1

        self.classes = list(self.data.keys())
        self.n_test_runs = args.n_test_runs
        self.partition = partition
        if train_transform is None:
            self.train_transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.RandomCrop(84, padding=8),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.train_transform = train_transform

        if test_transform is None:
            self.test_transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.test_transform = test_transform

        self.data = {}
        for idx in range(self.imgs.shape[0]):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        self.classes = list(self.data.keys())

    def __getitem__(self, item):
        if self.fix_seed:
            np.random.seed(item)

        cls_sampled = self.classes
        support_xs = []
        support_ys = []

        for idx, cls in enumerate(cls_sampled):
            imgs = np.asarray(self.data[cls]).astype('uint8')
            support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]), self.n_shots, False)
            support_xs.append(imgs[support_xs_ids_sampled])
            support_ys.append([cls] * self.n_shots)

        support_xs, support_ys = np.array(support_xs), np.array(support_ys)
        num_ways, num_shots, height, width, channel = support_xs.shape
        support_ys = support_ys.reshape((num_ways * self.n_shots,))
        support_xs = support_xs.reshape((-1, height, width, channel))
        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)

        support_xs = torch.stack(list(map(lambda x: self.train_transform(x.squeeze()), support_xs)))

        return support_xs, support_ys


class TSNEBaseImageNet(ImageNet):

    def __init__(self, args, partition='train', train_transform=None, test_transform=None, fix_seed=True):
        super(TSNEBaseImageNet, self).__init__(args, partition, False)
        self.fix_seed = fix_seed
        self.n_shots = args.n_queries
        self.n_ways = args.n_base_ways
        self.classes = list(self.data.keys())
        self.n_test_runs = args.n_test_runs
        self.partition = partition
        if train_transform is None:
            self.train_transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.RandomCrop(84, padding=8),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.train_transform = train_transform

        if test_transform is None:
            self.test_transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.test_transform = test_transform

        self.data = {}
        for idx in range(self.imgs.shape[0]):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        self.classes = list(self.data.keys())

    def __getitem__(self, item):
        if self.fix_seed:
            np.random.seed(item)

        cls_sampled = np.random.choice(self.classes, self.n_ways, False)
        support_xs = []
        support_ys = []

        for idx, cls in enumerate(cls_sampled):
            imgs = np.asarray(self.data[cls]).astype('uint8')
            support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]), self.n_shots, False)
            support_xs.append(imgs[support_xs_ids_sampled])
            support_ys.append([idx] * self.n_shots)

        support_xs, support_ys = np.array(support_xs), np.array(support_ys)
        num_ways, num_shots, height, width, channel = support_xs.shape
        support_ys = support_ys.reshape((num_ways * self.n_shots,))
        support_xs = support_xs.reshape((-1, height, width, channel))
        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)

        support_xs = torch.stack(list(map(lambda x: self.test_transform(x.squeeze()), support_xs)))

        return support_xs, support_ys

    def __len__(self):
        return self.n_test_runs


class MetaBaseImageNet(ImageNet):

    def __init__(self, args, partition='auxtest', train_transform=None, test_transform=None, fix_seed=True):
        super(MetaBaseImageNet, self).__init__(args, partition, False)
        self.fix_seed = fix_seed
        self.n_shots = 5

        self.classes = list(self.data.keys())
        self.n_test_runs = args.n_test_runs
        self.partition = partition
        if train_transform is None:
            self.train_transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.RandomCrop(84, padding=8),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.train_transform = train_transform

        if test_transform is None:
            self.test_transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.test_transform = test_transform

        self.data = {}
        for idx in range(self.imgs.shape[0]):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        self.classes = list(self.data.keys())

    def __getitem__(self, item):
        if self.fix_seed:
            np.random.seed(item)

        cls_sampled = self.classes
        support_xs = []
        support_ys = []

        for idx, cls in enumerate(cls_sampled):
            imgs = np.asarray(self.data[cls]).astype('uint8')
            support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]), self.n_shots, False)
            support_xs.append(imgs[support_xs_ids_sampled])
            support_ys.append([cls] * self.n_shots)

        support_xs, support_ys = np.array(support_xs), np.array(support_ys)
        num_ways, num_shots, height, width, channel = support_xs.shape
        support_ys = support_ys.reshape((num_ways * self.n_shots,))
        support_xs = support_xs.reshape((-1, height, width, channel))
        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)

        support_xs = torch.stack(list(map(lambda x: self.test_transform(x.squeeze()), support_xs)))

        return support_xs, support_ys

    def __len__(self):
        return self.n_test_runs


class MetaImageNet(ImageNet):

    def __init__(self, args, partition='train', train_transform=None, test_transform=None, fix_seed=True):
        super(MetaImageNet, self).__init__(args, partition, False)
        self.fix_seed = fix_seed
        self.n_ways = args.n_ways
        self.n_shots = args.n_test_shots

        self.n_queries = args.n_queries
        self.classes = list(self.data.keys())
        self.n_test_runs = args.n_test_runs
        self.partition = partition
        self.n_aug_support_samples = args.n_aug_support_samples
        if train_transform is None:
            self.train_transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.RandomCrop(84, padding=8),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.train_transform = train_transform

        if test_transform is None:
            self.test_transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.test_transform = test_transform

        self.data = {}
        for idx in range(self.imgs.shape[0]):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        self.classes = list(self.data.keys())

    def __getitem__(self, item):
        if self.fix_seed:
            np.random.seed(item)

        cls_sampled = np.random.choice(self.classes, self.n_ways, False)
        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []
        for idx, cls in enumerate(cls_sampled):
            imgs = np.asarray(self.data[cls]).astype('uint8')
            support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]), self.n_shots, False)
            support_xs.append(imgs[support_xs_ids_sampled])
            support_ys.append([idx] * self.n_shots)
            query_xs_ids = np.setxor1d(np.arange(imgs.shape[0]), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
            query_xs.append(imgs[query_xs_ids])
            query_ys.append([idx] * query_xs_ids.shape[0])
        support_xs, support_ys, query_xs, query_ys = np.array(support_xs), \
                    np.array(support_ys), np.array(query_xs), np.array(query_ys)
        num_ways, n_queries_per_way, height, width, channel = query_xs.shape
        query_xs = query_xs.reshape((num_ways * n_queries_per_way, height, width, channel))
        support_ys = support_ys.reshape((num_ways * self.n_shots,))
        query_ys = query_ys.reshape((num_ways * n_queries_per_way,))
        support_xs = support_xs.reshape((-1, height, width, channel))
        if self.n_aug_support_samples > 1:
            support_xs = np.tile(support_xs, (self.n_aug_support_samples, 1, 1, 1))
            support_ys = np.tile(support_ys.reshape((-1, )), (self.n_aug_support_samples))
        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
        query_xs = query_xs.reshape((-1, height, width, channel))
        query_xs = np.split(query_xs, query_xs.shape[0], axis=0)

        support_xs = torch.stack(list(map(lambda x: self.train_transform(x.squeeze()), support_xs)))
        query_xs = torch.stack(list(map(lambda x: self.test_transform(x.squeeze()), query_xs)))

        return support_xs, support_ys, query_xs, query_ys
        # return support_xs, support_ys, query_xs, query_ys, cls_sampled

    def __len__(self):
        return self.n_test_runs


