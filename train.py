import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle
import time
import argparse
import os
from utils.meter import AverageMeter
from torch.utils.data import  DataLoader
from config import dataset as ds
from utils.log import log
from utils.metrics import getaccuracy
from utils.data import CLEVR
from utils.models_baseline import Imodel,Qmodel,QImodel
from utils.RN import RN
from utils.models_baseline import save_checkpoint
from utils.callbacks import EarlyStopping
from collections import defaultdict
from utils.models_baseline import ConvInputModel
from torchvision import transforms
from PIL import Image
#%%

IMG = '/media/manoj/hdd/CLEVR_v1.0/images/{}/CLEVR_{}_{:06d}.png'

normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                    )

transform = transforms.Compose(
    [transforms.ToTensor(),
     normalize])

use_gpu = torch.cuda.is_available()
dtype = torch.FloatTensor
if use_gpu == True:
    dtype = torch.cuda.FloatTensor

cnn = ConvInputModel()
cnn.type(dtype)
print (cnn)


#%%

def run(net, split,loader, optimizer,tracker, epoch=0):
    global logger

    start_time = time.time()
    true = []
    pred = []
    loss_meter = tracker
    loss_meter.reset()
    Nprint = 100
    
    if split == 'train':
        train= True
    elif split == 'val' or split =='test':
        train = False

    if train:
        net.train()
        cnn.train()
    else:
        net.eval()
        cnn.eval()

    clslossfn = nn.CrossEntropyLoss()

    for i, data in enumerate(loader):

        #idx,img_index,np.float32(ans),qfeat,qlen
        qidxes,img_indices,labels,ques,qlen = data
        
        wholefeat  = None
        
        B = ques.size(0)       
        boxtensors = []
        for j in range(len(ques)):
            path = IMG.format(split,split,img_indices[j])
            img = Image.open(path).convert("RGB")
            img = img.resize((128,128))
            boxtensor = transform(img)
            boxtensors.append(boxtensor.unsqueeze(0))
        boxtensors = torch.cat(boxtensors,0)    
        boxvar = Variable(boxtensors.type(dtype))
        out = cnn(boxvar)    
        out = out.squeeze(-1).squeeze(-1)     
        Featsize = out.size(1)
        wholefeat = out
        wholefeat = wholefeat.permute(0,2,3,1)
        wholefeat = wholefeat.contiguous().view(B,-1,Featsize)
        


        true.extend(labels.long().numpy().tolist())
        cls_labels = Variable(labels.type(dtype))
        q_feats = Variable(ques.type(dtype).long())
        q_lens  = Variable(qlen.type(dtype).long())
        optimizer.zero_grad()
        out = net(box_feats = wholefeat,q_feats = q_feats,q_lens = q_lens)
        #sometimes in a batch only 1 example at the end
        if out.dim() == 1: # add one more dimension
            out = out.unsqueeze(0)

        loss = clslossfn(out, cls_labels.long())
        _,clspred = torch.max(out,-1)
        pred.extend(clspred.data.cpu().numpy().ravel())

        loss_meter.update(float(loss.data))

        if train:
            loss.backward()
            optimizer.step()

        if i == 0 and epoch == 0 and train:
            print ("Starting loss: {:.4f}".format(float(loss.data)))


        if i % Nprint == Nprint-1:
            infostr = "Epoch [{}]:Iter [{}]/[{}] Loss: {:.4f} Time: {:2.2f} s"
            printinfo = infostr.format(epoch , i, len(loader),
                                       loss_meter.avg,time.time() - start_time)

            print (printinfo)


    print("Completed in: {:2.2f} s".format(time.time() - start_time))
    ent = {}
    ent['true'] = true
    ent['pred'] = pred
    ent['loss'] = loss_meter.avg
    return ent

#%%
if __name__ == '__main__':
    global logger
    parser = argparse.ArgumentParser()
    parser.add_argument('--Nepochs', type=int,help='Number of epochs',default=200)
    parser.add_argument('--model', help='Model Q | I| QI | Main | RN')
    parser.add_argument('--lr', type=float,default=0.001,help='Learning rate')
    parser.add_argument('--save', help='save folder name',default='')
    parser.add_argument('--savefreq', help='save model frequency',type=int,default=1)

    args = parser.parse_args()
    #args =  argparse.Namespace(Nepochs=50, model='RN', save='',lr=2.5e-4 ,savefreq=1)
    args =  argparse.Namespace(Nepochs=50, model='RN', save='',lr=2.5e-4 ,savefreq=1)
    print (args)
    
    Ncls = ds['Ncls']

    savefolder = '_'.join([args.model,args.save])

    logger = log(savefolder,savefolder+".log")

    if not os.path.exists(savefolder):
            os.mkdir(savefolder)

    #resultfile = open(os.path.join(savefolder,savefolder+".txt"),"a")

    import torch.utils.data

    def collate_fn(batch):
        # put question lengths in descending order so that we can use packed sequences later
        batch.sort(key=lambda x: x[-1], reverse=True)
        return torch.utils.data.dataloader.default_collate(batch)



    trainset = CLEVR(file = ds['train'],istrain=True)
    testset = CLEVR(file = ds['val'],istrain=True)
    

    trainloader = DataLoader(trainset, batch_size=32,
                         shuffle=True, num_workers=4,collate_fn=collate_fn)
    testloader = DataLoader(testset, batch_size=32,
                             shuffle=False, num_workers=4,collate_fn=collate_fn)
    
#    use_gpu = torch.cuda.is_available()
#    dtype = torch.FloatTensor
#    if use_gpu == True:
#        dtype = torch.cuda.FloatTensor
    if args.model =='Q':
        model = Qmodel(Ncls)
    elif args.model =='I':
        model = Imodel(Ncls)
    elif args.model =='QI':
        model = QImodel(Ncls)
    elif args.model =='RN':
        model = RN(Ncls)
    model.type(dtype)
    print (model)

    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)

    Nepochs = args.Nepochs
    Modelsavefreq = args.savefreq
    tracker = AverageMeter()


    early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=5)
    saveloss = defaultdict(list)
    for epoch in range(0,Nepochs):

        train_kwargs = { 'net' : model,
                         'split' : 'train',
                         'loader': trainloader,
                         'optimizer' : optimizer,
                         'tracker' : tracker,
                         'epoch':epoch
                         }

        test_kwargs = { 'net' : model,
                       'split' : 'val',
                         'loader': testloader,
                         'optimizer' : optimizer,
                         'tracker' : tracker,
                         'epoch':epoch
                         }

        train = run(**train_kwargs)
        test = run(**test_kwargs)

        saveloss['trainloss'].append(train['loss'])
        saveloss['testloss'].append(test['loss'])

        pickle.dump(saveloss,open(savefolder+".pkl","wb"))

        acc = getaccuracy(train['true'],train['pred'])
        print('Train Loss: {:.4f} Acc: {:.2f} '.format(train['loss'],acc))        
        logger.warning('Train epoch:{} Loss: {:.3f} Acc: {:.2f}'.format(epoch,train['loss'],acc))

        acc = getaccuracy(test['true'],test['pred'])
        print('Test Loss: {:.4f} Acc: {:.2f} '.format(test['loss'],acc))
        logger.warning('Test epoch:{} Loss: {:.3f} Acc: {:.2f}'.format(epoch,test['loss'],acc))


        is_best = False
        if epoch % Modelsavefreq == 0:
            print ('Saving model ....')
            tbs = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'true':test['true'],
                'pred':test['pred'],
                'optimizer' : optimizer.state_dict(),
            }

            save_checkpoint(savefolder,tbs,is_best)

        early_stop.on_epoch_end(epoch,logs=test)
        if early_stop.stop_training:
            #decrease learning rate every 10 epocsh
            optimizer.param_groups[0]['lr'] *= 0.1
            lr =  optimizer.param_groups[0]['lr']
            print ("New Learning rate: ",lr)
            #early_stop.reset()
            break
    print('Finished Training')
    
    
    
    
#%%
    
def testd():
    fiter = iter(trainloader)    
    
    data = next(fiter)
    qidxes,img_indices,labels,ques,qlen = data
    split = 'train'
    
    wholefeat  = None
    
    B = ques.size(0)       
    boxtensors = []
    for j in range(len(ques)):
        path = IMG.format(split,split,img_indices[j])
        img = Image.open(path).convert("RGB")
        img = img.resize((128,128))
        boxtensor = transform(img)
        boxtensors.append(boxtensor.unsqueeze(0))
    boxtensors = torch.cat(boxtensors,0)    
    boxvar = Variable(boxtensors.type(dtype))
    out = cnn(boxvar)    
    out = out.squeeze(-1).squeeze(-1)     
    Featsize = out.size(1)
    wholefeat = out
    wholefeat = wholefeat.permute(0,2,3,1)
    wholefeat.contiguous().view(B,-1,Featsize)

    
