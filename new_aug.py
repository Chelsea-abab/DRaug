from torch.utils import data
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
import os
import torch
#from multiprocessing.pool import Pool
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
#from degradation_code.utils_de import *
from PIL import Image
from new_augmentation import *

size=224
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])   

types=['brightness','contrast','saturation','hue','sharpness','gamma','HALO','HOLE','SPOT']
AUG_PROB=np.random.beta(100,100)
levels=[[0,2],
        [0,2],
        [0,2],
        [-0.05,0.05],
        [0,2],
        [0.5,2],
        [0,2],
        [1],
        [1]]
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]


def randaug(image,mask):
  aug_prob=0.5
  op = np.random.choice(types)
  level=levels[types.index(op)]
  aug_level=0
  if len(level)==1:
    aug_level=1
  else:
    aug_level=np.random.uniform(level[0],level[1])
      
  trans=[]
  #tra.append(ToTensor())
  #trans.append(Resize(size))
  #op='HOLE'
  if op=='brightness': trans.append(Brightness(aug_level,prob=aug_prob))
  if op=='contrast':   trans.append(Contrast(aug_level,prob=aug_prob))
  if op=='saturation': trans.append(Saturation(aug_level,prob=aug_prob))
  if op=='hue':        trans.append(Hue(aug_level,prob=aug_prob))
  if op=='sharpness':  trans.append(Sharpness(aug_level,prob=aug_prob))
  if op=='gamma':      trans.append(Gamma(aug_level,prob=aug_prob))
  if op=='HALO':       trans.append(Halo(size,aug_level,prob=aug_prob))
  if op=='HOLE':       trans.append(Hole(size,prob=aug_prob))
  if op=='SPOT':       trans.append(Spot(size,prob=aug_prob))
  #if op=='BLUR':       trans.append(Blur(aug_level,prob=aug_prob))
  trans.append(Masked())
  trans.append(RandomHorizontalFlip(prob=1.0))
  trans.append(Normalize(MEAN,STD))
  traaug=Compose(trans)
  #print(type(traaug),op,image.device)
  auged,_=traaug(image,mask)
  #print('aug finished')
  return auged
