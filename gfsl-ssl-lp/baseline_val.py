

import os
import torch.optim as optim
import torch.backends.cudnn as cudnn
from models.util import create_model

from models.criterion import DistillKL
from models.model import Classifier, Classifier_GFSL

from dataset.get_loader import get_train_loader, get_test_loader, test_loader

from comm.util import *
from tqdm import tqdm
from comm.options import parse_option
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings('ignore')
torch.multiprocessing.set_sharing_strategy('file_system')


def Baseline(opt):

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
    # classifier = Classifier(640, n_cls)
    classifier = Classifier_GFSL(640, n_cls)
    model.eval()
    classifier.eval()

    if opt.n_gpu > 1:
        model = nn.DataParallel(model)
        classifier = nn.DataParallel(classifier)

    if torch.cuda.is_available():
        model = model.cuda()
        classifier = classifier.cuda()
        cudnn.benchmark = True

    module_list = nn.ModuleList([])
    module_list.append(model)
    module_list.append(classifier)

    best_fsl_1 = 0.0
    best_fsl_1_epoch = 0
    best_fsl_2 = 0.0
    best_fsl_2_epoch = 0
    best_hm_1 = 0.0
    best_hm_1_epoch = 0
    best_hm_2 = 0.0
    best_hm_2_epoch = 0
    for epoch in range(20, opt.epochs+1):
        print_func('epoch {}'.format(epoch), fout_file)
        ckpt_path = os.path.join(opt.model_pretrained, 'epoch_{}.pth'.format(epoch))
        params = torch.load(ckpt_path)
        if opt.n_gpu >1:
            model.module.load_state_dict(params['embedding'])
            classifier.module.load_state_dict(params['classifier'])
            clf_base = classifier.module.weight_base
        else:
            model.load_state_dict(params['embedding'])
            classifier.load_state_dict(params['classifier'])
            clf_base = classifier.weight_base

        # clf_base = get_clf(train_loader, module_list, opt, fout_file)


        print_func('======Traditional Few-Shot Classification Result======', fout_file)
        opt.n_test_shots = 1
        novel_acc, novel_std = fsl_test(module_list, opt)
        print_func('{}-way, {}-shot: novel_acc:{:.4f},novel_std:{:.4f}'.format(opt.n_ways, opt.n_test_shots, novel_acc,
                                                                               novel_std), fout_file)
        writer.add_scalar('Accuracy/fsl1shot', novel_acc, epoch)
        if novel_acc > best_fsl_1:
            best_fsl_1 = novel_acc
            best_fsl_1_epoch =  epoch
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
            print_func('** {}-way, {}-shot: best_hm1_acc:{:.4f}'.format(opt.n_ways, opt.n_test_shots, hm_acc), fout_file)

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
            print_func('** {}-way, {}-shot: best_hm2_acc:{:.4f}'.format(opt.n_ways, opt.n_test_shots, hm_acc), fout_file)

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
    Baseline(opt)

