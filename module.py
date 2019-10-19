import torch
from dgl.nn.pytorch import edge_softmax
import torch.nn as nn
from dgl import DGLGraph
from config_server import *

class D2GCN(nn.Module):
    def __init__(self, in_feat_dim, out_feat_dim):
        super(D2GCN, self).__init__()
        self.fedge = nn.Sequential(
            nn.Linear(in_feat_dim*2,in_feat_dim//64),
            nn.BatchNorm1d(in_feat_dim//64),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(in_feat_dim//64,out_feat_dim),
            nn.BatchNorm1d(out_feat_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        if feature_drop:
            self.feat_drop=nn.Dropout(feature_drop)
        else:
            self.feat_drop=lambda x:x
        if att_drop:
            self.att_drop=nn.Dropout(att_drop)
        else:
            self.att_drop=lambda x:x
        self.attn_l=nn.Parameter(torch.Tensor(size=(1,out_feat_dim)))
        self.attn_r=nn.Parameter(torch.Tensor(size=(1,out_feat_dim)))

        self.relu=nn.LeakyReLU(alpha)
        self.softmax=edge_softmax
        self.fnode=nn.Sequential(
            nn.Linear(in_feat_dim+out_feat_dim,out_feat_dim//64),
            nn.BatchNorm1d(out_feat_dim//64),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(out_feat_dim//64,out_feat_dim),
            nn.BatchNorm1d(out_feat_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        nn.init.xavier_normal_(self.attn_l.data,gain=1.414)
        nn.init.xavier_normal_(self.attn_r.data,gain=1.414)

    def build_graph(self,num_nodes,device):
        self.g = DGLGraph()
        self.g.add_nodes(num_nodes)
        for i in range(0, num_nodes):
            for j in range(0, num_nodes):
                if i != j:
                    self.g.add_edge(i, j)
                    self.g.add_edge(j, i)

        self.g.to(device)
        self.g.register_message_func(self.send_source)
        self.g.register_reduce_func(self.simple_reduce)

    def send_source(self, edges):
        edge_feature = self.fedge.forward(torch.cat((edges.src["h"], edges.dst["h"]),dim=1))
        msg = self.fnode.forward(torch.cat((edges.src["h"],edge_feature),dim=1))
        m=torch.mul(msg,edges.data['a_drop'])
        return {"m": m}

    def simple_reduce(self, nodes):
        return {"h": torch.sum(nodes.mailbox['m'], dim=1) + nodes.data["h"]}

    def edge_attention(self,edges):
        a=self.relu(edges.src['a1']+edges.dst['a2'])
        return {'a':a}

    def edge_softmax(self):
        att=self.softmax(self.g,self.g.edata.pop('a'))
        self.g.edata['a_drop']=self.att_drop(att)

    def forward(self, n_feature):
        a1=(n_feature * self.attn_l).sum(dim=-1).unsqueeze(-1)
        a2=(n_feature * self.attn_r).sum(dim=-1).unsqueeze(-1)
        self.g.ndata.update({'h':n_feature,'a1':a1,'a2':a2})
        self.g.apply_edges(self.edge_attention)
        self.edge_softmax()
        self.g.send(self.g.edges())
        self.g.recv(self.g.nodes())
        return self.g.ndata.pop('h')


class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()
        self.ft=nn.Sequential(
            nn.Linear(ifeatsize,representation_size),
            nn.BatchNorm1d(representation_size),
            nn.Dropout(dropout)
        )

        self.fc=nn.Sequential(
            nn.Linear(representation_size,2),
            nn.BatchNorm1d(2)
        )

        self.device=device
        self.gcn2d = D2GCN(representation_size, representation_size)
        self.gcn2d.build_graph(rpn_per_img,device)
        self.gcn1d = D2GCN(representation_size,representation_size)
        self.name="normal"

    def forward(self, feature):
        gcn2d_feature=torch.FloatTensor().to(self.device)
        for f in feature:
            f=torch.squeeze(f,dim=0)
            f=self.ft(f)
            output = self.gcn2d(f)

            output = output.max(dim=0,keepdim=True)[0]
            gcn2d_feature=torch.cat((gcn2d_feature,output),dim=0)
        self.gcn1d.build_graph(gcn2d_feature.shape[0],self.device)
        gcn1d_feat=self.gcn1d(gcn2d_feature)
        final_score=self.fc(gcn1d_feat)

        if torch.sum(final_score!=final_score) !=0:
            print("NAN VALUE")
        return final_score
