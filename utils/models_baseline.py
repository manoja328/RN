import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import shutil
import os
def save_checkpoint(savefolder,tbs, is_best=False):
    epoch = tbs['epoch']
    
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)
    
    filename = os.path.join(savefolder,'chkpoint_{}.pth.tar'.format(epoch))
    bestfile = os.path.join(savefolder,'model_best.pth.tar')
    torch.save(tbs, filename)
    if is_best:
        shutil.copyfile(filename, bestfile)

def load_checkpoint(filename,model,optimizer):
     #resume = 'a/checkpoint-1.pth.tar'
     if os.path.isfile(filename):
         checkpoint = torch.load(filename)
         epoch = checkpoint['epoch']
         print("=> loading checkpoint '{}' at epoch: {}".format(filename,epoch))
         model.load_state_dict(checkpoint['state_dict'])
         optimizer.load_state_dict(checkpoint['optimizer'])
         print("=> loaded checkpoint '{}' (epoch {})"
               .format(filename, checkpoint['epoch']))
     else:
         print("=> no checkpoint found at '{}'".format(filename))

class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True)
        self.top = nn.Sequential(*list(vgg.children())[0])
        for p in self.top.parameters():
            p.requires_grad = False
        _bottom  = [*list(vgg.children())[1]]
        self.onlylast = nn.Sequential(*_bottom[:2])
        for p in self.onlylast.parameters():
            p.requires_grad = False

    def forward(self,x):
        x = self.top(x)
        x = x.view(-1,25088)
        x = self.onlylast(x)
        return x
    
class ConvInputModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)

        
    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x

def build_cnn():
    resnet152 = models.resnet152(pretrained=True)
    modules=list(resnet152.children())[:-1]
    new_classifier =nn.Sequential(*modules)
    for p in new_classifier.parameters():
        p.requires_grad = False
    return new_classifier


def build_mlp(input_dim, hidden_dims, output_dim,
              use_batchnorm=False, dropout=0):
  layers = []
  D = input_dim
  if dropout > 0:
      layers.append(nn.Dropout(p=dropout))
  if use_batchnorm:
      layers.append(nn.BatchNorm1d(input_dim))
  for dim in hidden_dims:
        layers.append(nn.Linear(D, dim))
        if use_batchnorm:
          layers.append(nn.BatchNorm1d(dim))
        if dropout > 0:
          layers.append(nn.Dropout(p=dropout))
        layers.append(nn.ReLU(inplace=True)) # can use leaky relu tooo
        D = dim
  layers.append(nn.Linear(D, output_dim))
  return nn.Sequential(*layers)



class MLBAtt(nn.Module):
  def __init__(self,dim_q=512,dim_h=1024):
      super().__init__()
      self.linear_v_fusion = nn.Linear(dim_q,dim_h)
  def forward(self, x_v, x_q):
      x_v = self.linear_v_fusion(x_v)
      x_att = torch.mul(x_v, x_q)
      return x_att


class Mainmodel(nn.Module):
    def __init__(self,Ncls):
        super().__init__()

        I_CNN = 2048
        Box_GRU_out = 1024
        Q_GRU_out = 512
        Q_embedding = 300
        att_out = 512

        self.BoxRNN = nn.LSTM(att_out,Box_GRU_out,num_layers=1,bidirectional=False)
        self.reghead = build_mlp( Box_GRU_out , [1024], 1)        
        self.QRNN = nn.LSTM(Q_embedding,Q_GRU_out,num_layers=1,bidirectional=False)
        self.mlb = MLBAtt(dim_q=I_CNN,dim_h=att_out)

    def forward(self,box_feats,q_feats,box_coords):
                
        enc2,_ = self.QRNN(q_feats.permute(1,0,2))
        q_rnn = enc2[-1]
        B_size = box_feats.size(0)
        N = box_feats.size(1)
        q = q_rnn.unsqueeze(1)
        #print (q_rnn.size())
        qu_repeated = q.repeat(1,N,1)
        qfeat = qu_repeated.view(-1,512)
        #print (qu_repeated.size())
        x = box_feats.view(-1, 2048)       
        y = self.mlb(x,qfeat)
        box_q_mlb= y.view(B_size,N,-1)        
        
        #img_feats = F.normalize(img_feats,dim=-1)
        enc,_ = self.BoxRNN(box_q_mlb.permute(1,0,2))
        box_rnn = enc[-1]
        reg_scores = self.reghead(box_rnn)
        return reg_scores


class QImodel(nn.Module):
    def __init__(self,Ncls):
        super().__init__()

        I_CNN = 2048
        Box_GRU_out = 1024
        Q_GRU_out = 512
        Q_embedding = 300
        att_out = 512

        self.BoxRNN = nn.LSTM(att_out,Box_GRU_out,num_layers=1,bidirectional=False)
        self.clshead = build_mlp( I_CNN + Q_GRU_out , [1024],1)      
        self.QRNN = nn.LSTM(Q_embedding,Q_GRU_out,num_layers=1,bidirectional=False)


    def forward(self,box_feats,q_feats,box_coords):
                
        enc2,_ = self.QRNN(q_feats.permute(1,0,2))
        q_rnn = enc2[-1]
                
        I = box_feats[:,0,:]        
        qi = torch.cat([I,q_rnn],-1)
               
        cls_scores = self.clshead(qi)
        return cls_scores


class Qmodel(nn.Module):
    def __init__(self,Ncls):
        super().__init__()       
        Q_GRU_out = 512
        Q_embedding = 300        
        self.clshead = build_mlp( Q_GRU_out,[1024] , 1)     
        self.QRNN = nn.LSTM(Q_embedding,Q_GRU_out,num_layers=1,bidirectional=False)

    def forward(self,box_feats,q_feats,box_coords):
        enc2,_ = self.QRNN(q_feats.permute(1,0,2))
        q_rnn = enc2[-1]
        cls_scores = self.clshead(q_rnn)
        return cls_scores

class Imodel(nn.Module):
    def __init__(self,Ncls):
        super().__init__()
        I_CNN = 2048
        self.clshead = build_mlp( I_CNN , [1024], 1)

    def forward(self,box_feats,q_feats,box_coords):
      
        I = box_feats[:,0,:]
        cls_scores = self.clshead(I)
        return cls_scores

