import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class TieredImageNet(Dataset):
    def __init__(self, args, partition='train', pretrain=True, is_sample=False, k=4096,
                 transform=None):
        super(Dataset, self).__init__()
        self.args = args
        self.data_root = args.data_root
        self.partition = partition
        self.data_aug = args.data_aug
        self.mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        self.std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        self.pretrain = pretrain

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

        if self.pretrain:
            self.image_file_pattern = '%s_images.npz'
            self.label_file_pattern = '%s_labels.pkl'
        else:
            self.image_file_pattern = '%s_images.npz'
            self.label_file_pattern = '%s_labels.pkl'

        self.data = {}

        # modified code to load tieredImageNet
        image_file = os.path.join(self.data_root, self.image_file_pattern % partition)
        self.imgs = np.load(image_file)['images']
        label_file = os.path.join(self.data_root, self.label_file_pattern % partition)
        self.labels = self._load_labels(label_file)['labels']

        # pre-process for contrastive sampling
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            self.labels = np.asarray(self.labels)
            self.labels = self.labels - np.min(self.labels)
            num_classes = np.max(self.labels) + 1

            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(len(self.imgs)):
                self.cls_positive[self.labels[i]].append(i)

            self.cls_negative = [[] for _ in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]
            self.cls_positive = np.asarray(self.cls_positive)
            self.cls_negative = np.asarray(self.cls_negative)

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

    @staticmethod
    def _load_labels(file):
        try:
            with open(file, 'rb') as fo:
                data = pickle.load(fo)
            return data
        except:
            with open(file, 'rb') as f:
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                data = u.load()
            return data


class MetaBaseTieredtrain(TieredImageNet):
    def __init__(self, args, partition='train', train_transform=None, test_transform=None, fix_seed=True):
        super(MetaBaseTieredtrain, self).__init__(args, partition, False)
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

        support_xs = torch.stack(list(map(lambda x: self.test_transform(x.squeeze()), support_xs)))

        return support_xs, support_ys

    def __len__(self):
        return self.n_test_runs


class TSNEBaseTieredImageNet(TieredImageNet):

    def __init__(self, args, partition='auxtest', train_transform=None, test_transform=None, fix_seed=True):
        super(TSNEBaseTieredImageNet, self).__init__(args, partition, False)
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


class MetaBaseTieredImageNet(TieredImageNet):

    def __init__(self, args, partition='auxtest', train_transform=None, test_transform=None, fix_seed=True):
        super(MetaBaseTieredImageNet, self).__init__(args, partition, False)
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


class MetaTieredImageNet(TieredImageNet):

    def __init__(self, args, partition='train', train_transform=None, test_transform=None, fix_seed=True):
        super(MetaTieredImageNet, self).__init__(args, partition, False)
        self.fix_seed = fix_seed
        self.n_ways = args.n_ways
        self.n_shots = args.n_test_shots
        self.n_queries = args.n_queries
        self.classes = list(self.data.keys())
        self.n_test_runs = args.n_test_runs
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
        support_xs, support_ys, query_xs, query_ys = np.array(support_xs), np.array(support_ys), np.array(
            query_xs), np.array(query_ys)
        num_ways, n_queries_per_way, height, width, channel = query_xs.shape
        query_xs = query_xs.reshape((num_ways * n_queries_per_way, height, width, channel))
        query_ys = query_ys.reshape((num_ways * n_queries_per_way,))
        support_xs = support_xs.reshape((-1, height, width, channel))
        support_ys = support_ys.reshape((num_ways * self.n_shots,))
        if self.n_aug_support_samples > 1:
            support_xs = np.tile(support_xs, (self.n_aug_support_samples, 1, 1, 1))
            support_ys = np.tile(support_ys.reshape((-1, )), (self.n_aug_support_samples))
        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
        query_xs = query_xs.reshape((-1, height, width, channel))
        query_xs = np.split(query_xs, query_xs.shape[0], axis=0)

        support_xs = torch.stack(list(map(lambda x: self.train_transform(x.squeeze()), support_xs)))
        query_xs = torch.stack(list(map(lambda x: self.test_transform(x.squeeze()), query_xs)))

        return support_xs, support_ys, query_xs, query_ys

    def __len__(self):
        return self.n_test_runs
