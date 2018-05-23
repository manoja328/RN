from torch.utils.data import Dataset
import numpy as np
import torch
from .language import getglove
import os
import json
import pickle
from torchvision import transforms
import h5py
from copy import deepcopy
from collections import Counter
import numpy as np


class CountDataset(Dataset):

    def __init__(self, file,train=False):

        normalize = transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                            )

        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             normalize])
        with open(file,'rb') as f:
            self.data = pickle.load(f) #change this
#        if train:            
#            dataaug = []
#            a = 0
#            for ent in self.data:
#                if ent.get('answer',None) is None:
#                    cnt = Counter([ans['answer'] for ans in ent['answers']])
#                    common = [data for data in cnt.most_common(3) if data[1]>2]
#                    if len(common) <=1:
#                        #common = [('mc',ent['multiple_choice_answer'])]
#                        dataaug.append(ent)
#                    elif len(common) >1:
#                        #print (common)
#                        a+=1
#                        for c in common:
#                            entnew = deepcopy(ent)
#                            entnew['multiple_choice_answer'] = c[0]
#                            dataaug.append(entnew)
#                else:
#                    dataaug.append(ent)  
#                
#            self.data = dataaug   

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ent = self.data[idx]
        #print (ent)
        img_name = ent['image']
        nouns =  ent['noun']
        ans = ent.get('multiple_choice_answer',None)
        if ans is None:
            ans = ent['answer']
        #print (ent)
        que = ent['question']
        
       
        lasttwo = '/'.join(img_name.split("/")[-2:])
        lasttwo +=".pkl"
                
        if not os.path.exists(os.path.join("feats",lasttwo)):
            print ("file not found", lasttwo)
        
        pk = pickle.load(open(os.path.join("feats",lasttwo),"rb"))
        L =  len(pk) - 1 # lenght of entries in pickle file

        if L<=0:
            L = 1
            #print ("wow this has not entries",lasttwo)
        N =  20  # maximum entries to use
        L = 20   # uncomment if you want variable L 
        if L>N:
            L=N
        allfeat = np.zeros((N,2048),dtype=np.float32)    
        box_coords = np.zeros((N,4))
        for i,ent in enumerate(pk[:-1]):
            if i == N:
                break

            allfeat[i,:] = ent['feat'].flatten()
            box_coords[i,:] = np.array(ent['coords'])

        lastent = pk[-1]
        W = lastent['w']
        H = lastent['h']
        wholefeat = lastent['image']
        
        wholefeat = torch.from_numpy(wholefeat)
        
        qfeat = getglove(que)
        qfeat = torch.from_numpy(qfeat)
       
        imgarr = np.float32(np.array(allfeat))
        imgarr = torch.from_numpy(imgarr)
        box_coords = torch.from_numpy(np.array(box_coords,dtype=np.float32))        
        scale = torch.from_numpy(np.array([W,H,W,H],dtype=np.float32))
        box_coords = box_coords / scale   
        return wholefeat,imgarr,np.float32(ans),qfeat,box_coords,L



