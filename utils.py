import torch
seed=4
torch.manual_seed(seed)
print(seed)
from data import *
from config_server import *
import os
import json
from torch import nn
from module import Net
import numpy as np
from torch.utils.data import DataLoader
from itertools import product
use_cuda = torch.cuda.is_available()
device = torch.device(gpu if use_cuda else "cpu")
cls_loss_crierion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weight).to(device))
ranking_loss_crierion = nn.MarginRankingLoss(margin=1)
Z = {}
M = {}
dur = []
net = Net(device)
net.to(device)

for p in net.modules():
    if isinstance(p,nn.Linear):
        nn.init.xavier_normal_(p.weight.data)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=l2_norm)
scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.9)

if dataset=="youtube":
    train_dataset = YoutubeDataset(root="train",cls=target)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    test_dataset = YoutubeDataset(root="test",cls=target)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
else:
    train_dataset=SummeFeatureDataset(root="train",target=target)
    train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True,num_workers=0)
    test_dataset=SummeFeatureDataset(root="test",target=target)
    test_loader=DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=0)

def frame_score_to_clip_score(final_score, idx, vname, reduce):
    if dataset =="youtube":
        label_path = os.path.join(os.path.join(video_dir, vname[0] + "_match_label.json"))
        labels = json.load(open(label_path, 'r'))
        dur = labels[0]
        label=labels[1]
    else:
        label_path=os.path.join(data_dir,dataset,"GT",vname[0]+".mat")
        gt=io.loadmat(label_path)
        gt_score=gt.get('gt_score')
        nFrames=gt.get('nFrames')
        dur=[(i,i+49) for i in range(0,int(nFrames)-50,50)]
        dur.append((dur[-1][1]+1,int(nFrames)))
        label=[]
        gt_score=np.squeeze(gt_score,axis=1)
        for f in dur:
            if np.mean(gt_score[f[0]:f[1]])>=0.4:
                label.append(1)
            else:
                label.append(-1)
    pos_final_score=final_score[:,1]
    pos_clip_scores = []
    for clip in dur:
        idx_tensor = torch.FloatTensor(idx)
        mask1 = torch.ge(idx_tensor, clip[0])
        mask2 = torch.le(idx_tensor, clip[1])
        mask = torch.mul(mask1, mask2)
        pos_score = pos_final_score[mask]
        if reduce == "avg":
            pos_clip_scores.append(torch.mean(pos_score))
        elif reduce == "max":
            pos_clip_scores.append(torch.mean(pos_score))
    return pos_clip_scores, label


def eval_clip(pos_input, target):
    target = torch.FloatTensor(target)
    pos_idx = torch.ge(target, 1)
    neg_idx = torch.le(target, -1)
    p=(pos_idx==1).nonzero()
    n=(neg_idx==1).nonzero()


    # for p in pos_idx
    pos_input=torch.FloatTensor(pos_input)
    pos_predict_score = pos_input[pos_idx].tolist()
    neg_predict_score = pos_input[neg_idx].tolist()
    pairs = product(pos_predict_score, neg_predict_score)
    bingo = sum([p[0] >= p[1] for p in pairs])

    all = len(p)*len(n)
    return bingo, all


def loss_function(feat, label, vname):
    final_score = net(feat)
    label = torch.from_numpy(np.asarray(label, dtype=np.int)).to(device)
    mask = torch.ne(label, 0)

    pos_idx = torch.ge(label, 1)
    neg_idx = torch.le(label, -1)

    cls_loss = cls_loss_crierion(final_score[mask], (label[mask] + 1) / 2)

    if True:
        pos_pos = final_score[pos_idx, 1]
        neg_pos = final_score[neg_idx, 1]
        pos_neg = final_score[pos_idx, 0]
        neg_neg = final_score[neg_idx, 0]
        num_pair=min(sum(pos_idx), sum(neg_idx)).float()
        o = int(num_pair*0.8)

        if o > 0:
            posp = torch.sort(pos_pos, dim=0, descending=False)
            posn = torch.sort(pos_neg, dim=0, descending=True)
            negp = torch.sort(neg_pos, dim=0, descending=True)
            negn = torch.sort(neg_neg, dim=0, descending=False)
            #
            a = torch.randperm(o)
            b = torch.randperm(o)
            c = torch.randperm(o)
            d = torch.randperm(o)
            #
            p_p = posp[0][a]
            p_n = posn[0][b]
            n_p = negp[0][c]
            n_n = negn[0][d]

            rk_loss1 = ranking_loss_crierion(p_p, n_p, torch.ones(o).to(device))
            rk_loss2 = ranking_loss_crierion(p_n, n_n, -torch.ones(o).to(device))
            rk_loss = rk_loss1+rk_loss2
        else:
            rk_loss=None
    return cls_loss, rk_loss, final_score


