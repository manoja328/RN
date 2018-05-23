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
dtype2 = torch.FloatTensor
if use_gpu == True:
    dtype2 = torch.cuda.FloatTensor

cnn = ConvInputModel()
cnn.type(dtype2)
print (cnn)


#%%

def run(net, split,loader, optimizer,tracker, epoch=0):
    global logger, dtype

    start_time = time.time()
    true = []
    pred = []
    loss_meter = tracker
    loss_meter.reset()
    Nprint = 100
    
    if split == 'train':
        train= True
    elif split == 'val':
        train = False

    if train: net.train()
    else: net.eval()


    clslossfn = nn.CrossEntropyLoss()

    for i, data in enumerate(loader):


        img_indices,labels,ques = data
        
        B = ques.size(0)       
        boxtensors = []
        for j in range(len(ques)):
            path = IMG.format(split,split,img_indices[j])
            img = Image.open(path).convert("RGB")
            img = img.resize((224,224))
            boxtensor = transform(img)
            boxtensors.append(boxtensor.unsqueeze(0))
        boxtensors = torch.cat(boxtensors,0)    
        boxvar = Variable(boxtensors.type(dtype))
        out = cnn(boxvar)    
        out = out.squeeze(-1).squeeze(-1)     
        
        wholefeat = out
        wholefeat = wholefeat.permute(0,2,3,1)
        wholefeat = wholefeat.contiguous().view(B,196,-1)

        true.extend(labels.long().numpy().tolist())
        cls_labels = Variable(labels.type(dtype))
        q_feats = Variable(ques.type(dtype))
        optimizer.zero_grad()
        out = net(wholefeat,q_feats)
        #sometimes in a batch only 1 example at the end
        if out.dim() == 1: # add one more dimension
            out = out.unsqueeze(0)

        loss = clslossfn(out, cls_labels.long())
        _,clspred = torch.max(out,-1)
        pred.extend(clspred.data.cpu().numpy().ravel())

        loss_meter.update(loss.data[0])

        if train:
            loss.backward()
            optimizer.step()

        if i == 0 and epoch == 0 and train:
            print ("Starting loss: {:.4f}".format(loss.data[0]))


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
    global logger,dtype
    parser = argparse.ArgumentParser()
    parser.add_argument('--Nepochs', type=int,help='Number of epochs',default=200)
    parser.add_argument('--model', help='Model Q | I| QI | Main | RN')
    parser.add_argument('--lr', type=float,default=0.001,help='Learning rate')
    parser.add_argument('--save', help='save folder name',default='')
    parser.add_argument('--savefreq', help='save model frequency',type=int,default=1)

    args = parser.parse_args()
    args =  argparse.Namespace(Nepochs=50, model='RN', save='',lr=7e-4 ,savefreq=1)
    print (args)
    
    Ncls = ds['Ncls']

    savefolder = '_'.join([args.model,args.save])

    logger = log(savefolder,savefolder+".log")

    if not os.path.exists(savefolder):
            os.mkdir(savefolder)

    #resultfile = open(os.path.join(savefolder,savefolder+".txt"),"a")



    trainset = CLEVR(file = ds['train'],train=True)
    testset = CLEVR(file = ds['val'],train=True)
    
    testloader = DataLoader(testset, batch_size=32,
                             shuffle=False, num_workers=4)
    trainloader = DataLoader(trainset, batch_size=32,
                         shuffle=True, num_workers=4)

    use_gpu = torch.cuda.is_available()
    dtype = torch.FloatTensor
    if use_gpu == True:
        dtype = torch.cuda.FloatTensor
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
        logger.warning('Train epoch:{} Loss: {:.3f} Acc {:.2f}'.format(epoch,train['loss'],acc))

        acc = getaccuracy(test['true'],test['pred'])
        print('Test Loss: {:.4f}'.format(test['loss'],acc))
        logger.warning('Test epoch:{} Loss: {:.3f} Acc {:.2f}'.format(epoch,test['loss'],acc))


        is_best = False
        if epoch % Modelsavefreq == 0:
            print ('Saving model ....')
            tbs = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'true':test['true'],
                'pred_reg':test['pred_reg'],
                'pred_cls':test['pred_cls'],
                'optimizer' : optimizer.state_dict(),
            }

            save_checkpoint(savefolder,tbs,is_best)

        early_stop.on_epoch_end(epoch,logs=test)
        if early_stop.stop_training:
            #decrease learning rate every 10 epocsh
            optimizer.param_groups[0]['lr'] *= 0.1
            lr =  optimizer.param_groups[0]['lr']
            print ("New Learning rate: ",lr)
            early_stop.reset()
            #break
    print('Finished Training')
    
    
    
    
#%%
    
def testd():
    fiter = iter(trainloader)    
    
    data = next(fiter)
    img_indices,labels,ques = data
    split = 'train'
    
    B = ques.size(0)       
    boxtensors = []
    for j in range(len(ques)):
        path = IMG.format(split,split,img_indices[j])
        img = Image.open(path).convert("RGB")
        img = img.resize((224,224))
        boxtensor = transform(img)
        boxtensors.append(boxtensor.unsqueeze(0))
    boxtensors = torch.cat(boxtensors,0)    
    boxvar = Variable(boxtensors.type(dtype))
    out = cnn(boxvar)    
    out = out.squeeze(-1).squeeze(-1)     
    
    wholefeat = out
    wholefeat = wholefeat.permute(0,2,3,1)
    wholefeat = wholefeat.contiguous().view(B,196,-1)
        
