

import os
import torch.optim as optim
import torch.backends.cudnn as cudnn
from models.util import create_model

from models.criterion import DistillKL
from models.model import Classifier, Classifier_GFSL

from dataset.get_loader import get_train_loader, test_loader

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

    # result_name = '{}_{}'.format(opt.dataset, opt.model)
    # save_path = os.path.join(opt.save_folder, result_name)
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    fout_path = os.path.join(opt.save_folder, 'train_info.txt')
    fout_file = open(fout_path, 'a+')
    print_func(opt, fout_file)

    ##
    train_loader, n_cls = get_train_loader(opt)

    # model
    model = create_model(opt.model, opt.dataset)
    # classifier = Classifier(640, n_cls)
    classifier = Classifier_GFSL(640, n_cls)

    criterion_cls = nn.CrossEntropyLoss()

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)


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

    ## optimizer
    optimizer = optim.SGD(module_list.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)


    for epoch in range(1, opt.epochs + 1):
        print_func('epoch {}'.format(epoch), fout_file)

        acc, loss = train(train_loader, module_list, criterion_list, optimizer, opt, fout_file)
        # print('epoch {}, total time {:.2f}'.format(epoch, time2-time1))
        writer.add_scalar('Accuracy/train', acc, epoch)
        writer.add_scalar('Loss/train', loss, epoch)

        state = {
            'epoch': epoch,
            'embedding': model.state_dict() if opt.n_gpu <= 1 else model.module.state_dict(),
            'classifier': classifier.state_dict() if opt.n_gpu <= 1 else classifier.module.state_dict()
        }
        save_file = os.path.join(opt.save_folder, 'epoch_{}.pth'.format(epoch))
        torch.save(state, save_file)

        adjust_learning_rate(epoch, opt, optimizer)



def train(train_loader, module_list, criterion, optimizer, opt, fout_file):

    """One epoch training"""
    model = module_list[0]
    model.train()

    classifier = module_list[1]
    classifier.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    criterion_cls = criterion[0]

    tbar = tqdm(train_loader)

    for idx, (input, target) in enumerate(tbar):
        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        f_map3, f_map4, feature = model(input, map=True)
        logit = classifier(feature)
        cls_loss = criterion_cls(logit, target)

        loss = cls_loss
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ===================meters=====================
        acc1, acc5 = accuracy(logit, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0)//2)
        top5.update(acc5[0], input.size(0)//2)

        tbar.set_description(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    print_func(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5), fout_file)

    return top1.avg, losses.avg




if __name__ == '__main__':

    opt = parse_option()
    Baseline(opt)

