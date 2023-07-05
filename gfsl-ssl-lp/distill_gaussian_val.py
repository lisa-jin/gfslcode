

import os
import shutil
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from models.util import create_model
from torch.utils.data import DataLoader
from models import model_pool
from dataset.transform_cfg import transforms_options, transforms_list

from models.criterion import DistillKL
from models.model import Classifier, Classifier_GFSL, Scatter, Classifier_GFSL_2

from dataset.get_loader import get_train_loader, test_loader, get_test_loader

from comm.util import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings('ignore')
torch.multiprocessing.set_sharing_strategy('file_system')

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=64, help='batch_size 50')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=80, help='number of training epochs')
    parser.add_argument('--seed', type=int, default=100)

    # optimization
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,70', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--dataset', type=str, default='CUB',
                        choices=['miniImageNet', 'Dog','CIFAR-FS', 'CUB', 'tieredImageNet'])

    parser.add_argument('--method', type=str, default='dis-gau_2',
                        choices=['baseline', 'protonet', 'ours', 'distill', 'distill-gaussian', 'dis-gau_2'])
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--task_mode', type=str, default='gfsl')
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    parser.add_argument('--use_trainval', default=False, help='use trainval set')
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    parser.add_argument('--lamb_rot', type=float, default=1.0)
    parser.add_argument('--lamb_kl', type=float, default=0.0, help='hyper-param')
    parser.add_argument('--lamb_bk', type=float, default=0.0)
    parser.add_argument('--alpha', type=float, default=1.0, help='hyper-param')
    parser.add_argument('--beta_rot', type=float, default=0.5)
    parser.add_argument('--beta_dis', type=float, default=0.5)
    parser.add_argument('--beta_3', type=float, default=0.5)
    parser.add_argument('--beta_4', type=float, default=0.5)
    parser.add_argument('--orig_imsize', type=int, default=-1)

    # model parameter
    parser.add_argument('--fea_dim', type=int, default=640, help='feature dimension')
    parser.add_argument('--partion', default=1.0, type=float, help='radio of training instances')

    # specify folder
    parser.add_argument('--model_path', type=str, default='./save_ori', help='path to save model')
    # parser.add_argument('--tb_path', type=str, default='./tensorboard', help='path to tensorboard')
    parser.add_argument('--data_root', type=str, default='/data/datasets', help='path to data root')
    parser.add_argument('--model_pretrained', type=str, default='./save1')

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=300, help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5,
                        help='Number of classes for doing each classification run')
    # parser.add_argument('--n_shots', type=int, default=1,
    #                     help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15,
                        help='Number of query in test')
    parser.add_argument('--test_batch_size', type=int, default=1,help='Size of test batch)')
    parser.add_argument('--test_state', type=str, default='test')

    opt = parser.parse_args()
    if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        opt.transform = 'D'
    opt.n_aug_support_samples = 1


    opt.data_aug = True

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}'.format(opt.dataset, opt.method, opt.model)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name, opt.mode)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    opt.tb_path = os.path.join(opt.save_folder, 'tensorboard')
    if os.path.exists(opt.tb_path):
        shutil.rmtree(opt.tb_path)
    os.mkdir(opt.tb_path)

    opt.n_gpu = torch.cuda.device_count()
    return opt


def Ours(opt):

    writer = SummaryWriter(opt.tb_path)
    if opt.seed == 0:
        print('Random mode.')
        torch.backends.cudnn.benchmark = True
    else:
        import random
        print('Fixed random seed:', opt.seed)
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    fout_path = os.path.join(opt.save_folder, 'val_info.txt')
    fout_file = open(fout_path, 'a+')
    print_func(opt, fout_file)

    ##
    train_loader, n_cls = get_train_loader(opt)
    opt.n_cls = n_cls
    # model
    model = create_model(opt.model, opt.dataset)
    classifier_block3 = Classifier_GFSL_2(320, n_cls)
    classifier_block4 = Classifier_GFSL_2(640, n_cls)
    # classifier_block3 = Classifier(320, n_cls)
    # classifier_block4 = Classifier(640, n_cls)
    scatter = Scatter(opt.fea_dim, opt.n_cls * 16)

    criterion_cls = nn.CrossEntropyLoss(reduction='none')
    criterion_div = DistillKL(opt.kd_T)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)
    criterion_list.append(criterion_div)

    if opt.n_gpu > 1:
        model = nn.DataParallel(model)
        classifier_block3, classifier_block4 = nn.DataParallel(classifier_block3), nn.DataParallel(classifier_block4)
        scatter = nn.DataParallel(scatter)

    if torch.cuda.is_available():
        model = model.cuda()
        classifier_block3, classifier_block4 = classifier_block3.cuda(), classifier_block4.cuda()
        scatter = scatter.cuda()
        cudnn.benchmark = True

    module_list = nn.ModuleList([])
    module_list.append(model)
    module_list.append(classifier_block3)
    module_list.append(classifier_block4)
    module_list.append(scatter)


    ## optimizer
    # optimizer = optim.SGD(module_list.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    best_fsl_1 = 0.0
    best_fsl_1_epoch = 0
    best_fsl_2 = 0.0
    best_fsl_2_epoch = 0
    best_hm_1 = 0.0
    best_hm_1_epoch = 0
    best_hm_2 = 0.0
    best_hm_2_epoch = 0
    for epoch in range(20, opt.epochs + 1):
        print_func('epoch {}'.format(epoch), fout_file)
        ckpt_path = os.path.join(opt.model_pretrained, 'epoch_{}.pth'.format(epoch))
        params = torch.load(ckpt_path)
        if opt.n_gpu > 1:
            model.module.load_state_dict(params['embedding'])
            classifier_block4.module.load_state_dict(params['classifier'])
            clf_base = classifier_block4.module.weight_base
        else:
            model.load_state_dict(params['embedding'])
            classifier_block4.load_state_dict(params['classifier'])
            clf_base = classifier_block4.weight_base


        # clf_base = classifier_block4.module.weight_base
        # clf_base = get_clf(train_loader, module_list, opt, fout_file)

        print_func('======Traditional Few-Shot Classification Result======', fout_file)
        opt.n_test_shots = 1
        novel_acc, novel_std = fsl_test(module_list, opt)
        print_func('{}-way, {}-shot: novel_acc:{:.4f},novel_std:{:.4f}'.format(opt.n_ways, opt.n_test_shots, novel_acc,
                                                                               novel_std), fout_file)
        writer.add_scalar('Accuracy/fsl1shot', novel_acc, epoch)
        if novel_acc > best_fsl_1:
            best_fsl_1 = novel_acc
            best_fsl_1_epoch = epoch
            print_func('** 5-way, 1-shot, best_acc: test_acc:{:.4f},test_std:{:.4f}'.format(novel_acc, novel_std),
                       fout_file)

        opt.n_test_shots = 5
        novel_acc, novel_std = fsl_test(module_list, opt)
        print_func('{}-way, {}-shot: novel_acc:{:.4f},novel_std:{:.4f}'.format(opt.n_ways, opt.n_test_shots, novel_acc,
                                                                               novel_std), fout_file)
        writer.add_scalar('Accuracy/fsl5shot', novel_acc, epoch)
        if novel_acc > best_fsl_2:
            best_fsl_2 = novel_acc
            best_fsl_2_epoch = epoch
            print_func('** 5-way, 5-shot, best_acc: test_acc:{:.4f},test_std:{:.4f}'.format(novel_acc, novel_std),
                       fout_file)

        ## =====================Generalized Few-Shot Classification============
        opt.n_test_shots = 1
        print_func('======Generalized Few-Shot Classification Result======', fout_file)
        base_acc, base_std, novel_acc, novel_std, base_own, novel_own = gfsl_test(module_list, clf_base, opt)
        print_func(
            '{}-way, {}-shot: base_acc:{:.4f},base_std:{:.4f}'.format(opt.n_ways, opt.n_test_shots, base_acc, base_std),
            fout_file)
        print_func('{}-way, {}-shot: novel_acc:{:.4f},novel_std:{:.4f}'.format(opt.n_ways, opt.n_test_shots, novel_acc,
                                                                               novel_std), fout_file)
        print_func(
            '{}-way, {}-shot: base_own_acc:{:.4f}, novel_own_acc:{:.4f}'.format(opt.n_ways, opt.n_test_shots, base_own,
                                                                                novel_own), fout_file)
        hm_acc = (2 * base_acc * novel_acc) / (base_acc + novel_acc)
        print_func('{}-way, {}-shot: hm_acc:{:.4f}'.format(opt.n_ways, opt.n_test_shots, hm_acc), fout_file)
        writer.add_scalar('gfsl1shot-base', base_acc, epoch)
        writer.add_scalar('gfsl1shot-novel', novel_acc, epoch)
        writer.add_scalar('gfsl1shot-hm', hm_acc, epoch)
        if hm_acc > best_hm_1:
            best_hm_1 = hm_acc
            best_hm_1_epoch = epoch
            print_func('** {}-way, {}-shot: best_hm1_acc:{:.4f}'.format(opt.n_ways, opt.n_test_shots, hm_acc),
                       fout_file)


        ##===5-way 5-shot===
        opt.n_test_shots = 5
        base_acc, base_std, novel_acc, novel_std, base_own, novel_own = gfsl_test(module_list, clf_base, opt)
        print_func(
            '{}-way, {}-shot: base_acc:{:.4f},base_std:{:.4f}'.format(opt.n_ways, opt.n_test_shots, base_acc, base_std),
            fout_file)
        print_func('{}-way, {}-shot: novel_acc:{:.4f},novel_std:{:.4f}'.format(opt.n_ways, opt.n_test_shots, novel_acc,
                                                                               novel_std), fout_file)
        print_func(
            '{}-way, {}-shot: base_own_acc:{:.4f}, novel_own_acc:{:.4f}'.format(opt.n_ways, opt.n_test_shots, base_own,
                                                                                novel_own), fout_file)
        hm_acc = (2 * base_acc * novel_acc) / (base_acc + novel_acc)
        print_func('{}-way, {}-shot: hm_acc:{:.4f}'.format(opt.n_ways, opt.n_test_shots, hm_acc), fout_file)
        writer.add_scalar('gfsl5shot-base', base_acc, epoch)
        writer.add_scalar('gfsl5shot-novel', novel_acc, epoch)
        writer.add_scalar('gfsl5shot-hm', hm_acc, epoch)
        if hm_acc > best_hm_2:
            best_hm_2 = hm_acc
            best_hm_2_epoch = epoch
            print_func('** {}-way, {}-shot: best_hm2_acc:{:.4f}'.format(opt.n_ways, opt.n_test_shots, hm_acc),
                       fout_file)

    print_func('FSL: 1shot_best_epoch:{}, 5shot_best_epoch:{}'.format(best_fsl_1_epoch, best_fsl_2_epoch), fout_file)
    print_func('GFSL: 1shot_best_epoch:{}, 5shot_best_eoch:{}'.format(best_hm_1_epoch, best_hm_2_epoch), fout_file)




def get_clf(train_loader, module_list, opt, fout_file):
    model = module_list[0]
    model.eval()

    base_classifier = torch.randn(opt.n_cls, opt.fea_dim).cuda()
    tbar = tqdm(train_loader)

    class_lab = []
    with torch.no_grad():
        for idx, (input, target) in enumerate(tbar):
            input = input.float()
            if torch.cuda.is_available():
                input, target = input.cuda(), target.cuda()

            feature = model(input)

            # feature = feature.detach()
            for i in range(len(target)):
                label = target[i].tolist()
                class_lab.append(label)
                base_classifier[label] += feature[i]

    ####
    class_lab = np.array(class_lab)
    classifier = torch.randn(opt.n_cls, opt.fea_dim).cuda()
    for l in range(0, opt.n_cls):
        loc = np.where(class_lab == l)
        classifier[l, :] = base_classifier[l, :] / loc[0].shape[0]
    ####

    # classifier = classifier_block4.fc.weight
    return classifier


def fsl_test(module_list, opt):
    meta_loader, cls = test_loader(opt)
    model = module_list[0]
    model.eval()

    acc = []
    top1 = AverageMeter()
    with torch.no_grad():
        for idx, data in tqdm(enumerate(meta_loader)):
            support_x, support_y, query_x, query_y = data
            query_y = query_y.squeeze(0)
            if torch.cuda.is_available():
                support_x = support_x.cuda()
                query_x = query_x.cuda()
                query_y = query_y.cuda()

            img_shape = support_x.shape[-3:]
            support_x = support_x.view(-1, *img_shape)
            query_x = query_x.view(-1, *img_shape)

            support_fea = model(support_x)
            query_fea = model(query_x)

            classifier = torch.mean(support_fea.view(opt.n_ways, opt.n_test_shots, opt.fea_dim), 1)

            classifier = F.normalize(classifier)
            query_fea = F.normalize(query_fea)
            s_logits = torch.mm(query_fea, classifier.t())
            s_logits = F.softmax(s_logits, dim=0)

            acc1, _ = accuracy(s_logits, query_y, topk=(1,5))
            top1.update(acc1[0], query_x.size(0))

            acc.append(top1.avg.cpu())
            acc.append(acc1[0].item())
    return mean_confidence_interval(acc)


def gfsl_test(module_list, b_classifier, opt):
    meta_loader, cls = test_loader(opt)
    base_loader = get_test_loader(opt)

    model = module_list[0]
    model.eval()

    # classifier_block4 = module_list[-1]
    # classifier_block4.eval()

    base_top1 = AverageMeter()
    novel_top1 = AverageMeter()
    base_own_top1 = AverageMeter()
    novel_own_top1 = AverageMeter()
    base_acc = []
    novel_acc = []

    tqdm_meta = tqdm(meta_loader)
    with torch.no_grad():
        for idx, (data, (input, target)) in enumerate(zip(tqdm_meta, base_loader), 1):
            support_x, support_y, query_x, query_y = data
            query_y = query_y.squeeze(0) + cls
            input = input.float()
            target = target.squeeze(0)
            if torch.cuda.is_available():
                support_x, query_x, query_y = support_x.cuda(), query_x.cuda(), query_y.cuda()
                input, target = input.cuda(), target.cuda()

            img_shape = support_x.shape[-3:]
            support_x = support_x.view(-1, *img_shape)
            query_x = query_x.view(-1, *img_shape)
            input = input.view(-1, *img_shape)

            support_fea = model(support_x)
            query_fea = model(query_x)

            novel_classifier = torch.mean(support_fea.view(opt.n_ways, opt.n_test_shots, opt.fea_dim), 1)
            all_classifier = torch.cat((b_classifier, novel_classifier), 0)

            all_classifier = F.normalize(all_classifier)

            ##
            feature = model(input)
            feature = F.normalize(feature)
            base_logit = torch.mm(feature, all_classifier.t())
            base_logit = F.normalize(base_logit, 0)
            acc1, _ = accuracy(base_logit, target, topk=(1, 5))
            acc1_base_own, _ = accuracy(base_logit[:, :cls], target, topk=(1,5))

            base_top1.update(acc1[0], input.size(0))
            base_acc.append(base_top1.avg.cpu())
            base_acc.append(acc1[0].item())
            base_own_top1.update(acc1_base_own[0], 1)

            ##
            query_fea = F.normalize(query_fea)
            novel_logit = torch.mm(query_fea, all_classifier.t())
            novel_logit = F.normalize(novel_logit, 0)
            n_acc1, _ = accuracy(novel_logit, query_y, topk=(1,5))
            n_acc1_own, _ = accuracy(novel_logit[:, cls:], query_y-cls, topk=(1,5))

            novel_top1.update(n_acc1[0], query_x.size(0))
            novel_acc.append(novel_top1.avg.cpu())
            novel_acc.append(n_acc1[0].item())
            novel_own_top1.update(n_acc1_own[0], 1)

    base_acc, base_std = mean_confidence_interval(base_acc)
    novel_acc, novel_std = mean_confidence_interval(novel_acc)

    return base_acc, base_std, novel_acc, novel_std, base_own_top1.avg, novel_own_top1.avg




if __name__ == '__main__':
    opt = parse_option()
    Ours(opt)

