from torch.utils.data import Dataset
import numpy as np
import torch
from .language import getglove
import json


ANS = ['0',
 '1',
 '2',
 '3',
 '4',
 '5',
 '6',
 '7',
 '8',
 '9',
 '10',
 'blue',
 'brown',
 'purple',
 'cyan',
 'gray',
 'green',
 'yellow',
 'red',
 'cube',
 'cylinder',
 'sphere',
 'large',
 'small',
 'metal',
 'rubber',
 'yes',
 'no',
  ]


ANS_to_idx = {ans:idx for ans,idx in zip(ANS,range(len(ANS)))}
idx_to_ANS = {idx:ans for ans,idx in zip(ANS,range(len(ANS)))}

class CLEVR(Dataset):

    def __init__(self, file,train=False):


        with open(file,'rb') as f:
            self.data = json.load(f)['questions']
        
        self.train = train
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ent = self.data[idx]
        img_name = ent['image_filename']
        img_index = ent['image_index']
        if self.train:
            real_answer = ent['answer']        
            ans = ANS_to_idx[real_answer]
        que = ent['question']
        que_family = ent['question_family_index']
              
        qfeat = getglove(que)
        qfeat = torch.from_numpy(qfeat)
        if self.train:    
            return img_index,np.float32(ans),qfeat
        else:
            return img_index, 0 , qfeat

