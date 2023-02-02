from torch.utils import data
from torchvision.transforms import transforms
import os
#import torch
#from multiprocessing.pool import Pool
#import cv2
import numpy as np
#from degradation_code.utils_de import *
from PIL import Image,ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
size=224
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])                                 
tra = transforms.Compose([
                transforms.Resize([size,size]),
                transforms.ToTensor(),
                #normalize
                ])

mask_tra=transforms.Compose([
                transforms.Resize([size,size]),
                transforms.ToTensor(),
                ])
class_files=['nodr','mild_npdr','moderate_npdr','severe_npdr','pdr']
label_dis=[[0.49290005,0.10103768,0.27280175,0.05270344,0.08055707],
           [0.45858586,0.10707071,0.2010101, 0.17676768,0.05656566],
           [0.0548317, 0.11509229,0.32301846,0.35124864,0.1558089],
           [0.34008097,0.05060729,0.31781377,0.17408907,0.11740891],
           [0.5831422, 0.15481651,0.19896789,0.04300459,0.02006881],
           [0.1042059, 0.21155053,0.5831764, 0.06214689,0.03892028]]
#root='./ML/data/FundusDG/APTOS'
class dataSet(data.Dataset):
  def __init__(self, mode, root, sets):
    #self.files=[]
    self.mode=mode
    self.data=[]
    self.masks=[]
    self.labels=[]
    roots_img=os.path.join(root,'images')
    roots_mask=os.path.join(root,'masks')
    totalsum=0
    for setname in sets:
      count=0
      label=-1
      root=os.path.join(roots_img,setname)
      print('root:',root)
      for file_name in class_files:
        count_label=0
        label+=1
        file_path=os.path.join(root,file_name)
        img_names=os.listdir(file_path)
        #masks=os.listdir(os.path.join(roots_mask,setname,file_name))
        #print(file_path,os.path.join(roots_mask,setname,file_name))
        for imgname in img_names:
          img=Image.open(os.path.join(file_path,imgname))
          img=img.convert('RGB')
          mask=Image.open(os.path.join(roots_mask,setname,file_name,imgname))
          '''img=tra(img)
          mask=mask_tra(mask)'''

          self.data.append(img)
          self.masks.append(mask)
          self.labels.append(label)
          #print(file_name,imgname,label)
          count+=1
          count_label+=1
          totalsum+=1
        print(file_name,count_label)
      print(count)
    print('total:',totalsum)
    state=np.random.get_state()
    np.random.shuffle(self.data)
    np.random.set_state(state)
    np.random.shuffle(self.masks)
    np.random.set_state(state)
    np.random.shuffle(self.labels)
    
    

  def __getitem__(self, index):
    
    img=self.data[index]
    mask=self.masks[index]
    label=self.labels[index]
    img=tra(img)
    mask=mask_tra(mask)
    return img,mask,label

  def __len__(self):
        return len(self.data)
  