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
def getjson(fname,jsondir):
    file = fname.strip("\n")
    if (file == ''):
        return None
    
    
    fname = file.split("/")[-1][:-4]
    
    #print (fname)
    fullpath = os.path.join(jsondir,fname+'.json')
    if os.path.exists(fullpath):
        #print ('yes')
        with open(fullpath) as f:
            js_arr = json.load(f)            
    else:
        print (file,fullpath, " Not found !!!")
    return js_arr


class CountDataset(Dataset):

    def __init__(self, file,train=False):

        normalize = transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                            )

        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             normalize])
        #self.vocab = vocab
        with open(file,'rb') as f:
            self.data = pickle.load(f) #change this
    
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
        N = 15
        imgfeat = pk[-1]['image']       
        allfeat = np.zeros((N+1,2048),dtype=np.float32)    
        #print (que,nouns)
        for i,ent in enumerate(pk[:-1]):
            if i == N:
                break
            #print (ent['noun'])
            if ent['noun'] in nouns:
                allfeat[i+1,:] = ent['feat'].flatten()
            

        allfeat[0,:] = imgfeat # append whole scene CNN for initial step of RNN
        


        qfeat = getglove(que)
        qfeat = torch.from_numpy(qfeat)

#            class_score = js.get('confidence')
#            xmin= js.get('topleft').get('x')
#            ymin= js.get('topleft').get('y')
#            xmax= js.get('bottomright').get('x')
#            ymax= js.get('bottomright').get('y')
#            x =[1,xmin/w,ymin/h,xmax/w,ymax/h,(xmin/w)**2,
#                (ymin/h)**2,(xmax/w)**2,(ymax/h)**2,
#                xmin/w*ymin/h,xmax/w * ymax/h,class_score]
        
        imgarr = np.float32(np.array(allfeat))
        imgarr = torch.from_numpy(imgarr)
        x = [0,0,0,0]
        box_coords = torch.from_numpy(np.array(x,dtype=np.float32))
        return imgarr,np.float32(ans),qfeat,box_coords
