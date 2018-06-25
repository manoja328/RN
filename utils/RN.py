import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .models_baseline import TextProcessor
import torch.nn.init as init

class RN(nn.Module):
    def __init__(self,Ncls):
        super().__init__()

        I_CNN = 24
        Q_GRU_out = 128
        Q_embedding = 300
        LINsize = 256
        Boxcoords = 2
        
        self.Ncls = Ncls

        self.text = TextProcessor(
            embedding_tokens= 96,
            embedding_features=Q_embedding,
            lstm_features=Q_GRU_out,
            drop=0.5,
        )
        
        layers_g = [ nn.Linear( 2*I_CNN + Q_GRU_out + 2*Boxcoords, LINsize),
               nn.ReLU(inplace=True),
               nn.Linear( LINsize, LINsize),
               nn.ReLU(inplace=True),
               nn.Linear(LINsize,LINsize),
               nn.ReLU(inplace=True),
               nn.Linear(LINsize,LINsize),
               nn.ReLU(inplace=True)]

        self.g = nn.Sequential(*layers_g)

        layers_f = [ nn.Linear(LINsize ,LINsize),
                      nn.ReLU(inplace=True),
                      nn.Linear(LINsize ,LINsize),
                      nn.ReLU(inplace=True),
                      nn.Dropout(0.5),
                      nn.Linear(LINsize,self.Ncls) ]


        self.f = nn.Sequential(*layers_f)
        
        
#        def make_coords(N):           
#            c = []
#            W = 448
#            box = W/N
#            rangex = 4
#            for y in range(rangex):
#                for x in range(rangex):
#                    grid=[x*box, y*box, (x+1)*box, (y+1)*box]
#                    c.append(grid)                                
#            arr = np.array(c)/448
#            return torch.from_numpy(arr).float()
 
    

        def make_coords(total):           
            c = []
            W = int(total**0.5)
            for i in range(W):
                for j in range(W):
                    grid= [ (i - W//2)/W, (j - W//2)/W]
                    c.append(grid)                                
            arr = np.array(c)
            return torch.from_numpy(arr).float()
       
        coords = make_coords(64)

        self.pool_coords = torch.autograd.Variable(coords.type(torch.cuda.FloatTensor))
        
        
#        for m in self.g.modules():
#            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
#                init.xavier_uniform(m.weight)
#                if m.bias is not None:
#                    m.bias.data.zero_()
#           
#        for m in self.f.modules():
#            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
#                init.xavier_uniform(m.weight)
#                if m.bias is not None:
#                    m.bias.data.zero_()
    
    def forward(self,**kwargs):
        
        q_feats = kwargs['q_feats']    
        box_feats = kwargs['box_feats']    
        q_lens = kwargs['q_lens']
        
        q_rnn = self.text(q_feats,list(q_lens.data))
        
        N = box_feats.size(1)
        B = q_feats.size(0)
        
        qst = q_rnn.unsqueeze(1).unsqueeze(2).expand(-1,N,N,-1)
        

        pool_coords = self.pool_coords.unsqueeze(0).expand(B,-1,-1)
        box_feats = torch.cat([box_feats,pool_coords],-1)
        #print (N,B,box_feats.size(),pool_coords.size())
        
        #TODO: add coordinates + relative box features like size etc.
        #x_flat = torch.cat([x_flat, self.coord_tensor],2)
        
        b_i = box_feats.unsqueeze(2).expand(-1,-1,N,-1)       
        b_j = box_feats.unsqueeze(1).expand(-1,N,-1,-1)      

        b_full = torch.cat([b_i,b_j,qst],-1)
        b_full = b_full.view(B*N*N,-1)
        
        bg  = self.g(b_full)
        bg = bg.view(B,N*N,-1)        
        g_out = bg.sum(1).squeeze()
        count = self.f(g_out)                     
        return count
    
