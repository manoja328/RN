import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RN(nn.Module):
    def __init__(self,Ncls):
        super().__init__()

        I_CNN = 24
        Q_GRU_out = 128
        Q_embedding = 300
        LINsize = 256
        Boxcoords = 4
        
        self.Ncls = Ncls

        self.QRNN = nn.LSTM(Q_embedding,Q_GRU_out,num_layers=1,bidirectional=False)


        layers_g = [ nn.Linear( 2*I_CNN + Q_GRU_out, LINsize),
               nn.ReLU(inplace=True),
               nn.Linear( LINsize, LINsize),
               nn.ReLU(inplace=True),
               nn.Linear(LINsize,LINsize),
               nn.ReLU(inplace=True)]

        self.g = nn.Sequential(*layers_g)



        layers_f = [ nn.Linear(LINsize ,LINsize),
                      nn.ReLU(inplace=True),
                      nn.Dropout(0.5),
                      nn.Linear(LINsize,self.Ncls) ]


        self.f = nn.Sequential(*layers_f)
        
        
        def make_coords(N):           
            c = []
            W = 448
            box = W/N
            rangex = 4
            for y in range(rangex):
                for x in range(rangex):
                    grid=[x*box, y*box, (x+1)*box, (y+1)*box]
                    c.append(grid)                                
            arr = np.array(c)/448
            return torch.from_numpy(arr).float()
        
        coords = make_coords(4)

        self.pool_coords = torch.autograd.Variable(coords.type(torch.cuda.FloatTensor))
   
    
    
    def forward(self,**kwargs):
        
        q_feats = kwargs['q_feats']    
        box_feats = kwargs['box_feats']    
        
        enc2,_ = self.QRNN(q_feats.permute(1,0,2))
        q_rnn = enc2[-1]
        
        N = box_feats.size(1)
        B = q_feats.size(0)
        
        qst = q_rnn.unsqueeze(1).unsqueeze(2).expand(-1,N,N,-1)

        
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
    
