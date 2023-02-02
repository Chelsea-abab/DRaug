import argparse
from ast import arg
from cv2 import log
import time
import tensorboard
import torch, torchvision
import numpy as np
from torch.utils.tensorboard import SummaryWriter   
import sys, os, random, logging
from torch.utils import data
from validate import *
from newdataloader import *

from new_aug import randaug
datasets=['APTOS','DEEPDR','FGADR','IDRID','MESSIDOR','RLDR']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str,  default="0", help="gpu id")
    
    parser.add_argument("--test_env", type=int, default=0)
    parser.add_argument("--logs", type=str, default="temp")
    parser.add_argument("--root",type=str,default='./ML/data/FundusDG')
    
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--max_epoch",type=int,default=100)
    parser.add_argument("--lr",type=float,default=3e-4)
    parser.add_argument("--weight_decay",type=float,default=5e-4)
    parser.add_argument("--momentum",type=float,default=0.9)
    parser.add_argument("--logging_iter",type=int,default=1)
    parser.add_argument("--val_ep",type=int,default=1)
    parser.add_argument("--num_workers",type=int,default=4)
    parser.add_argument("--optim",type=str,default='adam')
    
    parser.add_argument("--aug_prob",type=float,default=0.5)
    parser.add_argument("--aug_type",type=str,default='001')
    parser.add_argument("--aug_level",type=int,default=5)
    
    return parser.parse_args()


if __name__ == "__main__":
    print("program starts...")
    args = get_args()
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    lr = args.lr
    weight_decay = args.weight_decay
    momentum = args.momentum
    logging_iter = args.logging_iter
    val_ep = args.val_ep
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    temp=torch.Tensor([0.])
    temp.cuda()
    
    
    model=torchvision.models.resnet50(pretrained=True)
    model_dict = model.state_dict()
    pretrain_path='/home/chengyuhan/ML/learn_code/pretrain/resnet50-19c8e357.pth'
    pretrained_dict = torch.load(pretrain_path)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    fc_features = model.fc.in_features
    model.fc = torch.nn.Linear(fc_features ,5)
    
    train_sets=[]
    test_sets=[]
    for i in range(6):
        if i!=args.test_env:
            train_sets.append(datasets[i])
        else:
            test_sets.append(datasets[i])
    print(train_sets)
    print(test_sets)
    
    
    train_dataset=dataSet('train',args.root,train_sets)
    print('train:',train_dataset.__len__())
    train_dataloader=data.DataLoader(train_dataset,batch_size = batch_size,shuffle=True,num_workers=args.num_workers)
    
    test_dataset=dataSet('test',args.root,test_sets)
    print('test:',test_dataset.__len__())
    test_dataloader=data.DataLoader(test_dataset,batch_size = batch_size,shuffle=True,num_workers=args.num_workers)

    

    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    else:
        optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay,nesterov=True)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(max_epoch * 0.5), int(max_epoch * 0.8)], gamma=0.1)


    criterion = torch.nn.CrossEntropyLoss()
    
    tensorboard_logname=os.path.join('./ML/image_augmentation/results/tensorboards',args.logs)
    writer = SummaryWriter(tensorboard_logname)
    #writer.add_text('config', str(cfg) + str(args))
    logging.basicConfig(filename=tensorboard_logname + '/log.txt', level=logging.INFO,format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("{} iterations per epoch".format(len(train_dataloader)))
    logging.info("{} iterations in test loader".format(len(test_dataloader)))
    logging.info("We have {} images in train set, have no validation set, and have {} images in test set.".format(len(train_dataset),  len(test_dataset)))

    best_performance = 0.0
    best_performance_test = 0.0
    
    #log_path=os.path.join("./ML/image_augmentation/results/logs",args.logs)
    model_path="./ML/image_augmentation/results/models"

    #for i in iterator:
    model.cuda()
    
    print('before train:')
    test_acc,test_auc,test_f1, test_loss = model_validate(model, test_dataloader, writer, criterion, -1, 'test')
    print(f"epoch:{-1}   acc: {test_acc} ; auc: {test_auc} ; f1: {test_f1} ; loss: {test_loss}")
    
    model.train()
    print('batch size:',args.batch_size,'  numworkers:', args.num_workers)
    print('new params range randaug')
    for epoch in range(max_epoch):
        loss_temp = 0
        epoch_start_time=time.time()
        for i,(image,mask,label) in enumerate(train_dataloader):
            itr_start_time=time.time()
            image = image.cuda().float()
            label = label.cuda().long()
            mask = mask.cuda()
            #print('itr cuda time:',time.time()-itr_start_time,'  out of ',image.shape[0])
            #itr_time=time.time()
            augimg=randaug(image,mask)

            #print('itr aug time:',time.time()-itr_time,'  out of ',image.shape[0])
            #time_stop1=time.time()
            #print('-------itr combine time:',time.time()-time_stop1)
            
            
            
            image_show = image[0]
            output = model(image)
            
            loss = criterion(output, label)
            #print('ori loss:',ori_loss,'loss:',loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_temp += loss.item()
            #print('itr total time::',time.time()-itr_start_time)

        aug_time=time.time()-epoch_start_time
        loss_temp = loss_temp / len(train_dataloader)
        '''loss_real = loss_real / len(train_dataloader)
        loss_fake = loss_fake / len(train_dataloader)'''

        logging.info('epoch: {}, loss: {}, aug time: {}.'.format(epoch, loss_temp, aug_time))
        writer.add_scalar('info/loss', loss_temp, epoch)
        writer.add_scalar('info/lr', scheduler.get_last_lr()[0], epoch)
        writer.add_image('Image/Original', image_show, epoch)
        #writer.add_image('Image/Preturb', tensor_normalize(image_new_show), epoch)      
        scheduler.step()
        
        
        if epoch % val_ep == 0:
            test_acc,test_auc,test_f1, test_loss = model_validate(model, test_dataloader, writer, criterion, epoch, 'test')
            
            print(f"epoch:{epoch}   acc: {test_acc} ; auc: {test_auc} ; f1: {test_f1} ; loss: {test_loss}")
            if test_auc > best_performance_test:
                best_performance_test = test_auc
                #logging.info("Saving best model...")
                #torch.save(model.state_dict(), os.path.join(model_path, f'{args.logs}_best_model.pth'))
        epoch_time=time.time()-epoch_start_time
        print('epoch time:',epoch_time)
    
    #model.load_state_dict(torch.load(os.path.join(log_path, 'best_model.pth')))
    test_acc,test_auc,test_f1, test_loss = model_validate(model, test_dataloader, writer, criterion, epoch, 'test')

    logging.info(f'Best performance on test: {best_performance_test}')
    logging.info(f'last performance on test: {test_acc},{test_auc},{test_f1}')
    logging.info("Saving final model...")
    #torch.save(model.state_dict(), os.path.join(model_path, f'{args.logs}_final_model.pth'))
    writer.close()
