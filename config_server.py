import time
import os
dataset="SumMe"
# dataset="youtube"
min_frame_per_clip=3
rpn_per_img=20
feat_dir="./data/SumMe/feat"

timestamp=int(time.time())
learning_rate=1e-5
representation_size=512
ifeatsize=4096

lamb=0.5
l_2=0
l_3=1-lamb

max_frame=300
# D2GCN
feature_drop=0.05
att_drop=0.05
alpha=0.2
dropout=0.1
ranking_rate=0.8

l1_norm=0.001
l2_norm=5
gpu="cuda:3"

i= 15
if dataset=="SumMe":
    files=os.listdir(feat_dir)
    target=files[i].split('.')[0]
    class_weight=[0.4,0.6]
else:
    files=os.listdir(os.path.join(feat_dir,"train"))
    class_weight=[0.4,0.6]
    target=files[i]
print("{} {} {}".format(target,lamb,i))
frame_dir=""
data_dir="./data/"
checkpoint = "./model/{}_{}_{}/".format(timestamp,dataset,target)
log_file=os.path.join("./logs/{}_{}_{}.txt".format(timestamp,dataset,target))
