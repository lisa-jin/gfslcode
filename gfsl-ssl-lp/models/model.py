import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Scatter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Scatter, self).__init__()
        self.conv = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(nn.Linear(in_dim, 1024), nn.LeakyReLU(0.1))
        self.fc2 = nn.Sequential(nn.Linear(1024, out_dim), nn.ReLU())

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, rot_num=4):
        super(Classifier, self).__init__()

        self.fc = nn.Linear(in_dim, out_dim, bias=None)
        self.rot_fc = nn.Sequential(nn.Linear(out_dim, rot_num))


    def forward(self, x, rot=False):
  
        output = self.fc(x)
        if rot:
            loc_output = self.rot_fc(output)
            return output, loc_output
        else:
            return output

class Classifier_GFSL(nn.Module):
    def __init__(self, in_dim, out_dim, rot_num=4):
        super(Classifier_GFSL, self).__init__()
        weight_base = torch.FloatTensor(out_dim, in_dim).normal_(0.0, np.sqrt(2.0 / in_dim))
        self.weight_base = nn.Parameter(weight_base, requires_grad=True)
        scale_cls = 10.0  # cosine similarity temperature t
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(scale_cls), requires_grad=True)
        self.rot_fc = nn.Sequential(nn.Linear(out_dim, rot_num))


    def forward(self, x, rot=False):
        x = F.normalize(x, dim=-1)
        weight = F.normalize(self.weight_base, dim=-1)
        output = self.scale_cls * torch.mm(x, weight.transpose(0,1))
        if rot:
            loc_output = self.rot_fc(output)
            return output, loc_output
        else:
            return output



class Classifier_CAM(nn.Module):
    def __init__(self, in_dim, out_dim, scale_cls ):
        super(Classifier_CAM, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=None)

        # weight_base = torch.FloatTensor(out_dim, in_dim).normal_(0.0, np.sqrt(2.0 / in_dim))
        # self.weight_base = nn.Parameter(weight_base, requires_grad=True)
        # scale_cls = 10.0  # cosine similarity temperature t
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(scale_cls), requires_grad=True)

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        weight = F.normalize(self.fc.weight, dim=-1)
        # weight = F.normalize(self.weight_base, dim=-1)
        output = self.scale_cls * torch.mm(x, weight.transpose(0,1))
        return output


class Model_CAM(nn.Module):
    def __init__(self, backbone, classifier):
        super(Model_CAM, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self,x):
        feature = self.backbone(x)
        logit = self.classifier(feature)
        return logit






class Classifier_GFSL_2(nn.Module):
    def __init__(self, in_dim, out_dim, rot_num=4):
        super(Classifier_GFSL_2, self).__init__()
        weight_base = torch.FloatTensor(out_dim, in_dim).normal_(0.0, np.sqrt(2.0 / in_dim))
        self.weight_base = nn.Parameter(weight_base, requires_grad=True)
        scale_cls = 10.0  # cosine similarity temperature t
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(scale_cls), requires_grad=True)
        self.rot_fc = nn.Sequential(nn.Linear(out_dim, rot_num))
        self.block_fc = nn.Sequential(nn.Linear(out_dim, 4))


    def forward(self, x, rot=False, block=False):
        x = F.normalize(x, dim=-1)
        weight = F.normalize(self.weight_base, dim=-1)
        output = self.scale_cls * torch.mm(x, weight.transpose(0,1))
        if rot:
            loc_output = self.rot_fc(output)
            return output, loc_output
        elif block:
            block_output = self.block_fc(output)
            return output, block_output
        else:
            return output


class Classifier_Tensor(nn.Module):
    def __init__(self, classifier):
        super(Classifier_Tensor, self).__init__()
        self.clf = nn.Parameter(classifier, requires_grad=True)
    def forward(self, x):
        out = torch.mm(x, self.clf.transpose(0,1))
        return out


class Classifier_ineq(nn.Module):
    def __init__(self, in_dim, out_dim, no_trans=16, embd_size=64):
        super(Classifier_ineq, self).__init__()
        self.clf = nn.Linear(in_dim, out_dim)
        self.eq_head = nn.Sequential(
            nn.Linear(640, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(inplace=True),
            nn.Linear(640, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(inplace=True),
            nn.Linear(640, no_trans)
        )
        self.inv_head = nn.Sequential(
            nn.Linear(640, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(inplace=True),
            nn.Linear(640, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(inplace=True),
            nn.Linear(640, embd_size)
        )

    def forward(self, x, inductive=False):
        out = self.clf(x)
        if inductive:
            out_eq = self.eq_head(x)
            out_in = self.inv_head(x)
            return out, out_eq, out_in
        else:
            return out


class Classifier_ineq_GFSL(nn.Module):
    def __init__(self, in_dim, out_dim, no_trans=16, embd_size=64):
        super(Classifier_ineq_GFSL, self).__init__()
        weight_base = torch.FloatTensor(out_dim, in_dim).normal_(0.0, np.sqrt(2.0 / in_dim))
        self.weight_base = nn.Parameter(weight_base, requires_grad=True)
        scale_cls = 10.0  # cosine similarity temperature t
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(scale_cls), requires_grad=True)
        self.eq_head = nn.Sequential(
            nn.Linear(640, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(inplace=True),
            nn.Linear(640, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(inplace=True),
            nn.Linear(640, no_trans)
        )
        self.inv_head = nn.Sequential(
            nn.Linear(640, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(inplace=True),
            nn.Linear(640, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(inplace=True),
            nn.Linear(640, embd_size)
        )

    def forward(self, x, inductive=False):
        x = F.normalize(x, dim=-1)
        weight = F.normalize(self.weight_base, dim=-1)
        out = self.scale_cls * torch.mm(x, weight.transpose(0, 1))
        if inductive:
            out_eq = self.eq_head(x)
            out_in = self.inv_head(x)
            return out, out_eq, out_in
        else:
            return out


























