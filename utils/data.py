from torch.utils.data import Dataset
import numpy as np
import torch
from .language import getglove
from .lang import Lang
import json

class CLEVR(Dataset):

    def __init__(self, file,istrain=False):


        with open(file,'rb') as f:
            self.data = json.load(f)['questions']
            
        
        self.lang = Lang()      
        self.istrain = istrain
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ent = self.data[idx]
        img_name = ent['image_filename']
        img_index = ent['image_index']
        question = ent['question']
        que_family = ent['question_family_index']             
        #qfeat = getglove(question)
        
        #print (question)
               
        qfeat,qlen = self.lang.encode_question(question)
        
        #qfeat = torch.from_numpy(qfeat)
        if self.istrain:   
            real_answer = ent['answer']        
            ans = self.lang.encode_answer(real_answer)
            return idx,img_index,np.float32(ans),qfeat,qlen
        else:
            return idx,img_index, 0 , qfeat,qlen

