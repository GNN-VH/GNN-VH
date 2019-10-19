
# dataset="Youtube"

from utils import *
import torch
import numpy as np
import time
np.set_printoptions(precision=4)
timestamp="1567682410"
# target=""

checkpoint = "./model/{}_{}_{}/".format(timestamp,dataset,target)
# model_num=len(os.listdir(checkpoint))-1

model_num=4
net.load_state_dict(torch.load(os.path.join(checkpoint,"model-{}.pt".format(model_num))))

best_acc=0
bingo = 0
all = 0
cls_mean_loss = 0
rk_mean_loss = 0
rel_mean_loss = 0
f=open(os.path.join("predict_score","{}.txt".format(target)),'w')
with torch.no_grad():
    net.eval()
    # TODO: TRANIN_LOADER TO TEST_LOADER
    start=time.time()
    for feat, label, idx, prop, vname in test_loader:
        f=open(os.path.join("predict_score","{}.txt".format(target)),'a')

        if len(feat) != len(label):
            continue
        feat = [v.to(device) for v in feat]
        label = [l.to(device) for l in label]
        final_score = net(feat)
        print("vname:{} final_Score:{}".format(vname,final_score[:,1].detach().cpu().numpy()))
        f.write("vname:{} final_Score:{}\n".format(vname,final_score[:,1].detach().cpu().numpy()))
        pos_predict_score, gt_score = frame_score_to_clip_score(final_score, idx, vname, reduce="avg")
        b, a = eval_clip(pos_predict_score, gt_score)
        bingo += b
        all += a
        f.close()
    end=time.time()
    duration=end-start
    acc = float(bingo) / float(all)
    print("acc:{} duration:{}".format(float(bingo) / float(all),duration))
