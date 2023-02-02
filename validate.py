import sys, os, random, logging, shutil
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import numpy as np

def model_validate(model, data_loader, writer, criterion, epoch, val_type):
    
    model.eval()
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss = 0
        label_list = []
        output_list = []
        pred_list = []

        for image,mask, label in data_loader:
            image = image.cuda().float()
            label = label.cuda().long()

            output = model(image)
            loss += criterion(output, label).item()

            _, pred = torch.max(output, 1)
            output_sf = softmax(output)

            label_list.append(label.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())
        
        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]

        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred, average='macro')

        # auc_ovr = roc_auc_score(label, output, average='macro', multi_class='ovr')        
        auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')

        loss = loss / len(data_loader)

        if epoch>=0:
            writer.add_scalar('info/{}_accuracy'.format(val_type), acc, epoch)
            writer.add_scalar('info/{}_loss'.format(val_type), loss, epoch)
            # writer.add_scalar('info/{}_auc_ovr'.format(val_type), auc_ovr, epoch)
            writer.add_scalar('info/{}_auc_ovo'.format(val_type), auc_ovo, epoch)
            writer.add_scalar('info/{}_f1'.format(val_type), f1, epoch)
            logging.info('INFO:  {} - epoch: {}, loss: {}, acc: {}, auc: {}, F1: {}.'.format(val_type, epoch, loss, acc, auc_ovo, f1))

    model.train()
    return acc,auc_ovo,f1, loss
