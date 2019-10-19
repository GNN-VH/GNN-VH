from torch.utils.data import Dataset
import os
from PIL import Image
from config_server import *
import torchvision.transforms.functional as F
import numpy as np
import json
from scipy import io

class VideoDataset(Dataset):
    def __init__(self,root='train',cls=None):
        self.root=root
        self.video_list=[]
        self.cls=cls
        if cls==None:
            cls_num=os.listdir(os.path.join(frame_dir,self.root))
            for t in cls_num:
                video_list=os.listdir(os.path.join(frame_dir,self.root,t))
                for item in video_list:
                    self.video_list.append(os.path.join(self.root,t,item))
        else:
            video_list=os.listdir(os.path.join(frame_dir,self.root,self.cls))
            for item in video_list:
                self.video_list.append(os.path.join(self.root,self.cls,item))


    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        # load image
        video_d=os.path.join(frame_dir,self.video_list[index])
        video={}
        for frames in os.listdir(video_d):
            img_path=os.path.join(video_d,frames)
            image=Image.open(img_path)
            image_tensor=F.to_tensor(image)
            # image=self.transform(image)
            idx=int(frames.split("_")[1].replace(".jpg",""))
            video[idx]=image_tensor

        frame_list=[video[key] for key in sorted(video.keys())]

        # load label
        if dataset=="youtube":
            label_path=os.path.join(video_dir,self.video_list[index]+"_match_label.json")
            labels=json.load(open(label_path,'r'))
            dur=labels[0]
            label=labels[1]
        else:
            label_path=os.path.join(data_dir,dataset,"GT",self.video_list[index]+".mat")
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

        l=[]
        j=0
        key_list=list(sorted(video.keys()))
        i=0
        while i<len(key_list):
            if j >=len(dur):
                break
            if key_list[i]>=dur[j][0] and key_list[i]<=dur[j][1]:
                l.append(label[j])
                i+=1
            elif key_list[i]<dur[j][0]:
                l.append(label[j])
                i+=1
            else:
               j+=1

        return frame_list,l,self.video_list[index],key_list

class YoutubeDataset(VideoDataset):
    def __init__(self,root='train',cls=None):
        VideoDataset.__init__(self,root,cls)

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        feat_d=os.path.join(feat_dir,self.video_list[index]+".npy")
        feat,label,prop,idx=np.load(feat_d,allow_pickle=True)
        if not isinstance(label,list):
            label=label.tolist()
        if not isinstance(feat,list):
            feat=feat.tolist()
        if not isinstance(idx,list):
            idx=idx.tolist()
        if not isinstance(prop,list):
            prop=prop.tolist()

        return feat,label,idx,prop, self.video_list[index]


class SummeFrameDataset(VideoDataset):
    def __init__(self):
        self.video_list=os.listdir(frame_dir)
    def __len__(self):
        return super().__len__()


    def __getitem__(self, index):

        return super().__getitem__(index)
class SummeFeatureDataset(YoutubeDataset):
    def __init__(self,root,target):
        self.video_list=[]
        files=os.listdir(feat_dir)
        for f in files:
            basename=f.replace(".npy","")
            if root =="train" and basename!=target:
                self.video_list.append(basename)
            if root =="test" and basename==target:
                self.video_list.append(basename)
                break
    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        return super().__getitem__(index)

