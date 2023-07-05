from dataset.mini_imagenet import ImageNet, MetaImageNet, MetaBaseImageNet  #, DatasetWrapper, PairBatchSampler
from dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet, MetaBaseTieredImageNet
from dataset.cifar import CIFAR100, MetaCIFAR100, MetaBaseCIFAR
from dataset.cub import CUB, MetaCUB, MetaBaseCUB
from dataset.dog import Dog, MetaDog, MetaBaseDog
from torch.utils.data import DataLoader, Sampler
from dataset.transform_cfg import transforms_test_options, transforms_options, transforms_list
from torch.utils.data import Dataset
import csv, torchvision, random, os

from torchvision import transforms, datasets
from collections import defaultdict
import numpy as np

random.seed(100)

class PairBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_iterations=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iterations = num_iterations

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for k in range(len(self)):
            if self.num_iterations is None:
                offset = k*self.batch_size
                batch_indices = indices[offset:offset+self.batch_size]
            else:
                batch_indices = random.sample(range(len(self.dataset)),
                                              self.batch_size)

            yield batch_indices

    def __len__(self):
        if self.num_iterations is None:
            return len(self.dataset) // self.batch_size
        else:
            return self.num_iterations


class DatasetWrapper(Dataset):

    def __init__(self, dataset, indices=None):
        self.base_dataset = dataset
        if indices is None:
            self.indices = list(range(len(dataset)))
        else:
            self.indices = indices

        self.classwise_indices = defaultdict(list)
        for i in range(len(self)):
            y = self.base_dataset.labels[self.indices[i]]
            self.classwise_indices[y].append(i)
        self.num_classes = max(self.classwise_indices.keys())+1

    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)

    def get_class(self, i):
        return self.base_dataset.labels[self.indices[i]]


def get_train_loader(opt):
    # dataloader
    if opt.dataset=='miniImageNet' or opt.dataset=='tieredImageNet' or opt.task_mode != 'gfsl':
        train_partition = 'trainval' if opt.use_trainval else 'train'
    else:
        train_partition = 'base_80'
    # if opt.mode=='train' or opt.dataset=='miniImageNet' or opt.dataset=='tieredImageNet':
    #     train_partition = 'trainval' if opt.use_trainval else 'train'
    # else:
    #     train_partition = 'base_80'
    if opt.dataset == 'miniImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        opt.partion = 1
        trainset = DatasetWrapper(ImageNet(args=opt, partition=train_partition, transform=train_trans))
        get_sampler = lambda d: PairBatchSampler(d, opt.batch_size)
        train_loader = DataLoader(trainset, batch_sampler=get_sampler(trainset), num_workers=opt.num_workers)
        num_sample = len(trainset)
        # opt.partion = 0.1
        # testset = DatasetWrapper(ImageNet(args=opt, partition=test_partition, transform=test_trans))
        # # get_test_sampler = lambda d: PairBatchSampler(d, 256)
        # base_test_loader = DataLoader(testset, batch_sampler=get_sampler(testset), num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64
    elif opt.dataset == 'tieredImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        trainset = DatasetWrapper(TieredImageNet(args=opt, partition=train_partition, transform=train_trans))
        get_train_sampler = lambda d: PairBatchSampler(d, opt.batch_size)
        train_loader = DataLoader(trainset,batch_sampler=get_train_sampler(trainset), num_workers=opt.num_workers)
        num_sample = len(trainset)
        if opt.use_trainval:
            n_cls = 448
        else:
            n_cls = 351
    elif opt.dataset == 'CIFAR-FS':
        train_trans, test_trans = transforms_options['D']

        trainset = DatasetWrapper(CIFAR100(args=opt, partition=train_partition, transform=train_trans))
        get_train_sampler = lambda d: PairBatchSampler(d, opt.batch_size)

        train_loader = DataLoader(trainset, batch_sampler=get_train_sampler(trainset),
                                  num_workers=opt.num_workers)
        num_sample = len(trainset)
        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64

    elif opt.dataset == 'CUB':
        train_trans, test_trans = transforms_options[opt.transform]

        trainset = DatasetWrapper(CUB(args=opt, partition=train_partition, transform=train_trans))
        get_train_sampler = lambda d: PairBatchSampler(d, opt.batch_size)
        train_loader = DataLoader(trainset, batch_sampler=get_train_sampler(trainset), num_workers=opt.num_workers)
        num_sample = len(trainset)
        if opt.use_trainval:
            n_cls = 150
        else:
            n_cls = 100


    elif opt.dataset == 'Dog':
        train_trans, test_trans = transforms_options[opt.transform]

        trainset = DatasetWrapper(Dog(args=opt, partition=train_partition, transform=train_trans))
        get_train_sampler = lambda d: PairBatchSampler(d, opt.batch_size)
        train_loader = DataLoader(trainset, batch_sampler=get_train_sampler(trainset), num_workers=opt.num_workers)
        num_sample = len(trainset)
        if opt.use_trainval:
            n_cls = 90
        else:
            n_cls = 70

    else:
        raise NotImplementedError(opt.dataset)

    if opt.method == 'in-eq':
        return train_loader, n_cls, num_sample
    else:
        return train_loader, n_cls,


def get_metatrain_loader(opt):
    if opt.dataset == 'CUB':
        train_support_trans, train_query_trans = transforms_options[opt.transform]
        opt.n_test_runs = opt.train_tasks
        train_set = MetaCUB(args=opt, partition='train', train_transform=train_support_trans,
                            test_transform=train_query_trans)

        # val_support_trans, val_query_trans = transforms_test_options[opt.transform]
        opt.n_test_runs = opt.val_tasks
        # val_set = MetaCUB(args=opt, partition='val', train_transform=val_support_trans, test_transform=val_query_trans)
        #
        # test_support_trans, test_query_trans = transforms_test_options[opt.transform]
        # opt.n_test_runs = opt.test_tasks
        # test_set = MetaCUB(args=opt, partition='test', train_transform=test_support_trans,
        #                    test_transform=test_query_trans)

        train_loader = DataLoader(train_set, batch_size=opt.batch_tasks, shuffle=True, drop_last=False,
                                  num_workers=opt.num_workers)
        n_cls = 100
        # val_loader = DataLoader(val_set, batch_size=opt.batch_val_tasks, shuffle=False, drop_last=False, num_workers=opt.num_workers)
        # test_loader = DataLoader(test_set, batch_size=opt.batch_val_tasks, shuffle=False, drop_last=False, num_workers=opt.num_workers)
    elif opt.dataset == 'miniImageNet':
        train_support_trans, train_query_trans = transforms_options[opt.transform]
        opt.n_test_runs = opt.train_tasks
        train_set = MetaImageNet(args=opt, partition='train', train_transform=train_support_trans,
                                 test_transform=train_query_trans)

        # val_support_trans, val_query_trans = transforms_test_options[opt.transform]
        opt.n_test_runs = opt.val_tasks
        # val_set = MetaImageNet(args=opt, partition='val', train_transform=val_support_trans, test_transform=val_query_trans)

        # test_support_trans, test_query_trans = transforms_test_options[opt.transform]
        # opt.n_test_runs = opt.test_tasks
        # test_set = MetaImageNet(args=opt, partition='test', train_transform=test_support_trans, test_transform=test_query_trans)

        train_loader = DataLoader(train_set, batch_size=opt.batch_tasks, shuffle=True, drop_last=False,
                                  num_workers=opt.num_workers)
        n_cls=64
        # val_loader = DataLoader(val_set, batch_size=opt.batch_val_tasks, shuffle=False, drop_last=False,
        #                         num_workers=opt.num_workers)
        # test_loader = DataLoader(test_set, batch_size=opt.batch_val_tasks, shuffle=False, drop_last=False,
        #                          num_workers=opt.num_workers)
    elif opt.dataset == 'tieredImageNet':
        train_support_trans, train_query_trans = transforms_options[opt.transform]
        opt.n_test_runs = opt.train_tasks
        train_set = MetaTieredImageNet(args=opt, partition='train', train_transform=train_support_trans, test_transform=train_query_trans)
        opt.n_test_runs = opt.val_tasks
        train_loader = DataLoader(train_set, batch_size=opt.batch_tasks, shuffle=True, drop_last=False,
                                  num_workers=opt.num_workers)
        n_cls = 351
    elif opt.dataset == 'CIFAR-FS':
        train_support_trans, train_query_trans = transforms_options['D']
        opt.n_test_runs = opt.train_tasks
        train_set = MetaCIFAR100(args=opt, partition='train', train_transform=train_support_trans, test_transform=train_query_trans)
        opt.n_test_runs = opt.val_tasks
        train_loader = DataLoader(train_set, batch_size=opt.batch_tasks, shuffle=True, drop_last=False,
                                  num_workers=opt.num_workers)
        n_cls = 64
    elif opt.dataset == 'Dog':
        train_support_trans, train_query_trans = transforms_options[opt.transform]
        opt.n_test_runs = opt.train_tasks
        train_set = MetaDog(args=opt, partition='train', train_transform=train_support_trans, test_transform=train_query_trans)
        opt.n_test_runs = opt.val_tasks
        train_loader = DataLoader(train_set, batch_size=opt.batch_tasks, shuffle=True, drop_last=False,
                                  num_workers=opt.num_workers)
        n_cls = 70
    else:
        raise NotImplementedError(opt.dataset)
    return train_loader, n_cls



def get_test_loader(opt):

    if opt.dataset == 'miniImageNet':
        train_trans, test_trans = transforms_test_options[opt.transform]
        # if opt.use_trainval:
        #     opt.n_ways = 80
        # else:
        #     opt.n_ways = 64

        meta_baseloader = DataLoader(MetaBaseImageNet(args=opt, partition='auxtest',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans,
                                                  fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)

    elif opt.dataset == 'tieredImageNet':
        train_trans, test_trans = transforms_test_options[opt.transform]
        if opt.use_trainval:
            n_cls = 448
        else:
            n_cls = 351
        meta_baseloader = DataLoader(MetaBaseTieredImageNet(args=opt, partition='aux',
                                                        train_transform=train_trans,
                                                        test_transform=test_trans,
                                                        fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)


    elif opt.dataset == 'CIFAR-FS':
        train_trans, test_trans = transforms_test_options['D']
        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64
        meta_baseloader = DataLoader(MetaBaseCIFAR(args=opt, partition='base_20',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans,
                                                  fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)


    elif opt.dataset == 'CUB':
        train_trans, test_trans = transforms_test_options[opt.transform]
        meta_baseloader = DataLoader(MetaBaseCUB(args=opt, partition='base_20',
                                                      train_transform=train_trans,
                                                      test_transform=test_trans),
                                         batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                         num_workers=opt.num_workers)

        if opt.use_trainval:
            n_cls = 150
        else:
            n_cls = 100

    elif opt.dataset == 'Dog':
        train_trans, test_trans = transforms_test_options[opt.transform]
        meta_baseloader = DataLoader(MetaBaseDog(args=opt, partition='base_20',
                                                      train_transform=train_trans,
                                                      test_transform=test_trans),
                                         batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                         num_workers=opt.num_workers)

        if opt.use_trainval:
            n_cls = 90
        else:
            n_cls = 70
    else:
        raise NotImplementedError(opt.dataset)

    return meta_baseloader







def test_loader(opt):

    if opt.dataset == 'miniImageNet':
        # train_partition = 'trainval' if opt.use_trainval else 'train'
        train_trans, test_trans = transforms_test_options[opt.transform]
        meta_testloader = DataLoader(MetaImageNet(args=opt, partition=opt.test_state,
                                                  train_transform=train_trans,
                                                  test_transform=test_trans,
                                                  fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)

        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64
    elif opt.dataset == 'tieredImageNet':
        train_trans, test_trans = transforms_test_options[opt.transform]
        meta_testloader = DataLoader(MetaTieredImageNet(args=opt, partition=opt.test_state,
                                                        train_transform=train_trans,
                                                        test_transform=test_trans,
                                                        fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)

        if opt.use_trainval:
            n_cls = 448
        else:
            n_cls = 351
    elif opt.dataset == 'CIFAR-FS':
        train_trans, test_trans = transforms_test_options['D']
        meta_testloader = DataLoader(MetaCIFAR100(args=opt, partition=opt.test_state,
                                                  train_transform=train_trans,
                                                  test_transform=test_trans,
                                                  fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64

    elif opt.dataset == 'CUB':
        train_trans, test_trans = transforms_test_options[opt.transform]
        meta_testloader = DataLoader(MetaCUB(args=opt, partition=opt.test_state,
                                                      train_transform=train_trans,
                                                      test_transform=test_trans),
                                         batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                         num_workers=opt.num_workers)

        if opt.use_trainval:
            n_cls = 150
        else:
            n_cls = 100

    elif opt.dataset == 'Dog':
        train_trans, test_trans = transforms_test_options[opt.transform]
        meta_testloader = DataLoader(MetaDog(args=opt, partition=opt.test_state,
                                                      train_transform=train_trans,
                                                      test_transform=test_trans),
                                         batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                         num_workers=opt.num_workers)

        if opt.use_trainval:
            n_cls = 90
        else:
            n_cls = 70
    else:
        raise NotImplementedError(opt.dataset)

    return meta_testloader, n_cls



