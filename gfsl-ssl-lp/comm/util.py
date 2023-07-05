import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import scipy
from scipy.stats import t
import operator


def divide_fea(f_map):
    _, _, w, h = f_map.shape
    half_w, half_h = int(w / 2), int(h / 2)

    f_map_ll = f_map[:, :, 0:half_w, 0:half_h]
    f_map_lr = f_map[:, :, 0:half_w, half_h:]
    f_map_rl = f_map[:, :, half_w:, 0:half_h]
    f_map_rr = f_map[:, :, half_w:, half_h:]

    fea_block = torch.cat(
        [f_map_ll.mean([-1, -2]), f_map_lr.mean([-1, -2]), f_map_rl.mean([-1, -2]), f_map_rr.mean([-1, -2])], dim=0)

    return fea_block

def rotrate_concat(inputs):
    out = None
    for x in inputs:
        x_90 = x.transpose(2,3).flip(2)
        x_180 = x.flip(2).flip(3)
        x_270 = x.flip(2).transpose(2,3)
        if out is None:
            out = torch.cat((x, x_90, x_180, x_270),0)
        else:
            out = torch.cat((out, x, x_90, x_180, x_270),0)
    return out


def block_fea(f_map):

    fea_block = divide_fea(f_map)
    # fea_block4 = divide_fea(f_map4)
    return fea_block


def get_div_loss(logit_set, logit, criterion_div):
    bs, num_cls = logit.shape

    loss = criterion_div(logit_set[0:bs], logit.detach()) +  criterion_div(logit_set[bs:bs * 2], logit.detach()) +\
           criterion_div(logit_set[bs * 2:bs* 3], logit.detach()) + criterion_div(logit_set[bs* 3:bs * 4], logit.detach())
    return loss



def binarize_and_smooth_labels(T, nb_classes, smoothing_const=0):
    import sklearn.preprocessing
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(
        T, classes=range(0, nb_classes)
    )
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    T = torch.FloatTensor(T).cuda()

    return T


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None, num_classes=64):
        super(BCEWithLogitsLoss, self).__init__()
        self.num_classes = num_classes
        self.criterion = nn.BCEWithLogitsLoss(weight=weight,
                                              size_average=size_average,
                                              reduce=reduce,
                                              reduction=reduction,
                                              pos_weight=pos_weight)

    def forward(self, input, target):
        target_onehot = F.one_hot(target, num_classes=self.num_classes)
        return self.criterion(input, target_onehot)


def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.lr * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1 + confidence) / 2., n - 1)
    return m, h


def class_count(data):
    labels = np.array(data.dataset.labels)
    class_data_num = []
    for l in np.unique(labels):
        class_data_num.append(len(labels[labels == l]))
    return class_data_num


def rotation(input):
    input_90 = input.transpose(2, 3).flip(2)
    input_180 = input.flip(2).flip(3)
    input_270 = input.flip(2).transpose(2, 3)
    input_data = torch.cat((input, input_90, input_180, input_270), 0)
    return input_data


def rot_lab(opt, num_cls):

    unit_lab = torch.arange(num_cls)

    rot_labels = unit_lab.unsqueeze(1).expand(num_cls, opt.batch_size)
    rot_labels = rot_labels.reshape(-1).cuda().long()
    rot_labels = F.one_hot(rot_labels.to(torch.int64), num_cls).float()

    return rot_labels


def split_input(input):

    bs, c, w, h = input.shape

    half_w, half_h = int(w / 2), int(h /2)
    input_ll = input[:, :, 0:half_w, 0:half_h]
    input_lr = input[:, :, 0:half_w, half_h:]
    input_rl = input[:, :, half_w:, 0:half_h]
    input_rr = input[:, :, half_w:, half_h:]

    rot_input_ll = rotation(input_ll)
    rot_input_lr = rotation(input_lr)
    rot_input_rl = rotation(input_rl)
    rot_input_rr = rotation(input_rr)

    cat_input_ll = torch.cat((input, input, input, input), 0)
    cat_input_lr = torch.cat((input, input, input, input), 0)
    cat_input_rl = torch.cat((input, input, input, input), 0)
    cat_input_rr = torch.cat((input, input, input, input), 0)

    cat_input_ll[:, :, 0:half_w, 0:half_h] = rot_input_ll
    cat_input_lr[:, :, 0:half_w, half_h:] = rot_input_lr
    cat_input_rl[:, :, half_w:, 0:half_h] = rot_input_rl
    cat_input_rr[:, :, half_w:, half_h:] = rot_input_rr

    return torch.cat((cat_input_ll, cat_input_rl, cat_input_rl, cat_input_rr), 0)



def print_func(info_str, fout_path):
    print(info_str)
    print(info_str, file=fout_path)



def get_center(train_loader, model, opt):

    model.eval()
    centroids = nn.Parameter(torch.randn(opt.n_cls, opt.fea_dim)).cuda()

    for idx, (input, target, _) in enumerate(train_loader):

        input = input.float()
        target = target.long()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        output = output.detach()
        for i in range(len(target)):
            label = target[i]
            centroids[label] += output[i]
    centroids /= torch.tensor(class_count(train_loader)).float().unsqueeze(1).cuda()
    return centroids




def get_prototype(logit, label, opt):

    logit = logit.detach()
    proto = torch.randn(opt.n_cls, opt.n_cls).cuda()

    for i in range(opt.n_cls):
        idx = (label == i).nonzero().squeeze(1)
        if idx.size(0) > 0:
            proto[i] = torch.mean(logit[idx,:],dim=0)
    return proto


def get_memory(logit, label, memory, opt):
    logit = logit.detach()

    for i in range(opt.n_cls):
        idx = (label == i).nonzero().squeeze(1)
        bank_i = memory[i,:,:].squeeze(0)
        if idx.size(0) > 0:
            temp_logit = logit[idx,:]
            update_bank = torch.cat([temp_logit, bank_i],dim=0)
            memory[i,:,:] = update_bank[0:opt.memory_size]

    return memory
