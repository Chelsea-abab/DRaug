import math
import numpy as np
import random
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
types=['brightness','contrast','saturation','hue','sharpness','gamma','HALO','HOLE','SPOT','BLUR']
levels=[[0.4,2],
        [0.4,2],
        [0.3,2],
        [-0.05,0.05],
        [0,2],
        [0.4,2],
        [0,2],
        [1],
        [1],
        [0,8],
        [1]]
class Resize(object):
    def __init__(self, size):
        self.size = size
 
    def __call__(self, image, target=None):
        #print(image.shape)
        image = F.resize(image, self.size)
        if target is not None:
            target = F.resize(target, self.size)
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob
 
    def __call__(self, image, target=None):
        if random.random() < self.prob:
            image = F.hflip(image)
            if target is not None:
                mask = []
                for target_item in target:
                    mask.append(F.hflip(target_item))
                target = mask
        return image, target
        
class RandomRotation(object):
    def __init__(self, degree, prob = 0.5):
        self.degree = degree
        self.prob = prob

    def __call__(self, image, target=None):
        angle = float(torch.empty(1).uniform_(float(self.degree[0]), float(self.degree [1])).item())
        
        if random.random() < self.prob:
            image = F.rotate(image, angle)
            if target is not None:
                mask = []
                for target_item in target:
                    mask.append(F.rotate(target_item, angle))
        return image, target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
 
    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
 
class ToTensor(object):
    def __call__(self, image, target=None):
        image = F.to_tensor(image)
        if target is not None:
            mask = []
            for target_item in target:
                mask.append(torch.as_tensor(np.array(target_item), dtype=torch.int64))
                target = mask
        return image, target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
 
    def __call__(self, image, mask=None):
        for t in self.transforms:
            #print('in compose???')
            image, mask = t(image, mask)
            
        return image, mask
    
class Masked(object):
    def __call__(self, image, mask=None):          
        return image*mask, mask
    
class Brightness(object):
    def __init__(self, level, prob = 0.5):
        self.level = level
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            #print('Brightness')
            image = F.adjust_brightness(image, self.level)
        return image, target
    
class Contrast(object):
    def __init__(self, level, prob = 0.5):
        self.level = level
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            #print('Contrast')
            image = F.adjust_contrast(image, self.level)
        return image, target
    
class Saturation(object):
    def __init__(self, level, prob = 0.5):
        self.level = level
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            #print('Saturation')
            image = F.adjust_saturation(image, self.level)
        return image, target
    
class Hue(object):
    def __init__(self, level, prob = 0.5):
        self.level = level
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            #print('Hue')
            image = F.adjust_hue(image, self.level)
        return image, target

class Sharpness(object):
    def __init__(self, level, prob = 0.5):
        self.level = level
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            #print('Sharpness')
            image = F.adjust_sharpness(image, self.level)
        return image, target

class Gamma(object):
    def __init__(self, level, prob = 0.5):
        self.level = level
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            #print('Gamma')
            image = F.adjust_gamma(image, self.level)
        return image, target
    
class Blur(object):
    def __init__(self, sigma, prob = 0.5):
        self.sigma = sigma
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            rad_w = random.randint(int(self.sigma/3), int(self.sigma/2))
            if (rad_w % 2) == 0: rad_w = rad_w + 1
            rad_h = rad_w
            image = F.gaussian_blur(image, (rad_w,rad_h), self.sigma)
        return image, target
    
class Halo(object):
    def __init__(self, size, brightness_factor, prob = 0.5):
        self.size = size
        self.brightness_factor = brightness_factor
        self.prob = prob

    def __call__(self, image, target=None):
        #print(image.device)
        if random.random() < self.prob:
            weight_r = [251/255,141/255,177/255]
            weight_g = [249/255,238/255,195/255]
            weight_b = [246/255,238/255,147/255]
            
            if self.brightness_factor >= 0.2:
                num=random.randint(1,2)
            else:
                num=random.randint(0,2)
            w0_a = random.randint(self.size/2-int(self.size/8),self.size/2+int(self.size/8))
            center_a = [w0_a,w0_a]
            wei_dia_a =0.75 + (1.0-0.75) * random.random()
            dia_a = self.size*wei_dia_a
            Y_a, X_a = np.ogrid[:self.size, :self.size]
            dist_from_center_a = np.sqrt((X_a - center_a[0]) ** 2 + (Y_a - center_a[1]) ** 2)
            circle_a = dist_from_center_a <= (int(dia_a / 2))

            mask_a = torch.zeros((self.size, self.size)).cuda()
            mask_a[circle_a] = torch.mean(image) #np.multiply(A[0], (1 - t))

            center_b =center_a
            Y_b, X_b = np.ogrid[:self.size, :self.size]
            dist_from_center_b = np.sqrt((X_b - center_b[0]) ** 2 + (Y_b - center_b[1]) ** 2)

            dia_b_max =2* int(np.sqrt(max(center_a[0],self.size-center_a[0])*max(center_a[0],self.size-center_a[0])+max(center_a[1],self.size-center_a[1])*max(center_a[1],self.size-center_a[1])))/self.size
            wei_dia_b = 1.0+(dia_b_max-1.0) * random.random()

            if num ==0:
                # if halo tend to be a white one, set the circle with a larger radius.
                dia_b = self.size * wei_dia_b + abs(max(center_b[0] - self.size / 2, center_b[1] - self.size / 2) + self.size*2 / 3)
            else:
                dia_b =self.size* wei_dia_b +abs(max(center_b[0]-self.size/2,center_b[1]-self.size/2)+self.size/2)

            circle_b = dist_from_center_b <= (int(dia_b / 2))

            mask_b = torch.zeros((self.size, self.size)).cuda()
            mask_b[circle_b] = torch.mean(image)

            weight_hal0 = [0, 1, 1.5, 2, 2.5]
            delta_circle = torch.abs(mask_a - mask_b) * weight_hal0[1]
            
            dia = max(center_a[0],self.size-center_a[0],center_a[1],self.size-center_a[1])*2
            gauss_rad = int(np.abs(dia-dia_a))
            sigma = 2/3*gauss_rad+0.01

            if(gauss_rad % 2) == 0:
                gauss_rad= gauss_rad+1
            delta_circle = F.gaussian_blur(torch.reshape(delta_circle,(1,224,224)), (gauss_rad, gauss_rad), sigma)

            if num==0 or num==1 or num==2:
                delta_circle = torch.tensor([(weight_r[num]*delta_circle).tolist(),(weight_g[num]*delta_circle).tolist(),(weight_b[num]*delta_circle).tolist()]).cuda()
            else:
                num=1
                delta_circle = torch.tensor([(weight_r[num]*delta_circle).tolist(),(weight_g[num]*delta_circle).tolist(),(weight_b[num]*delta_circle).tolist()]).cuda()
            
            image = image + torch.reshape(delta_circle,(3,224,224))

            image = torch.maximum(image, torch.tensor(0))
            image = torch.minimum(image, torch.tensor(1))
            
            
        return image, target
    

class Hole(object):
    def __init__(self,size,prob = 0.5):
        self.size = size
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            diameter_circle = random.randint(int(0.4 * self.size), int(0.7 * self.size))
            center =[random.randint(self.size/4,self.size*3/4),random.randint(self.size*3/8,self.size*5/8)]
            Y, X = np.ogrid[:self.size, :self.size]
            dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
            circle = dist_from_center <= (int(diameter_circle/2))

            mask = torch.zeros((self.size, self.size)).cuda()
            mask[circle] = 1

            num_valid = torch.sum(target)
            aver_color = torch.sum(image) / (3*num_valid)
            if aver_color>0.25:
                brightness = random.uniform(-0.26,-0.262)
                brightness_factor = random.uniform((brightness-0.06*aver_color), brightness-0.05*aver_color)
            else:
                brightness =0
                brightness_factor =0
            # print( (aver_color,brightness,brightness_factor))
            mask = mask * brightness_factor

            rad_w = random.randint(int(diameter_circle*0.55), int(diameter_circle*0.75))
            rad_h = random.randint(int(diameter_circle*0.55), int(diameter_circle*0.75))
            sigma = 2/3 * max(rad_h, rad_w)*1.2

            if (rad_w % 2) == 0: rad_w = rad_w + 1
            if(rad_h % 2) ==0 : rad_h =rad_h + 1

            mask = F.gaussian_blur(torch.reshape(mask,(1,224,224)), (rad_w, rad_h), sigma)
            mask = torch.tensor([mask.tolist(), mask.tolist(), mask.tolist()]).cuda()
            image = image + torch.reshape(mask,(3,224,224))
            image = torch.maximum(image, torch.tensor(0))
            image = torch.minimum(image, torch.tensor(1))
        return image, target
    
class Spot(object):
    def __init__(self,size,center=None, radius=None,prob = 0.5):
        self.size = size
        self.center=center
        self.radius=radius
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            s_num =random.randint(5,10)
            mask0 =  torch.zeros((self.size, self.size)).cuda()
            center=self.center
            radius=self.radius
            for i in range(s_num):
                # if radius is None: # use the smallest distance between the center and image walls
                    # radius = min(center[0], center[1], w-center[0], h-center[1])
                radius = random.randint(math.ceil(0.01*self.size),int(0.05*self.size))

                # if center is None: # in the middle of the image
                center  = [random.randint(radius+1,self.size-radius-1),random.randint(radius+1,self.size-radius-1)]
                Y, X = np.ogrid[:self.size, :self.size]
                dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
                circle = dist_from_center <= (int(radius/2))

                k =(14/25) +(1.0-radius/25)
                beta = 0.5 + (1.5 - 0.5) * (radius/25)
                A = k *torch.ones((3,1)).cuda()
                d =0.3 *(radius/25)
                t = math.exp(-beta * d)

                mask = torch.zeros((self.size, self.size)).cuda()
                mask[circle] = torch.multiply(A[0],torch.tensor(1-t).cuda())
                mask0 = mask0 + mask
                mask0[mask0 != 0] = 1

                sigma = (5 + (20 - 0) * (radius/25))*2
                rad_w = random.randint(int(sigma / 5), int(sigma / 4))
                rad_h = random.randint(int(sigma / 5), int(sigma / 4))
                if (rad_w % 2) == 0: rad_w = rad_w + 1
                if (rad_h % 2) == 0: rad_h = rad_h + 1

                mask = F.gaussian_blur(torch.reshape(mask,(1,224,224)), (rad_w, rad_h), sigma)
                mask = torch.tensor([mask.tolist(), mask.tolist(), mask.tolist()]).cuda()
                
                image = image + torch.reshape(mask,(3,224,224))
                
                image = torch.maximum(image,torch.tensor(0))
                image = torch.minimum(image,torch.tensor(1))
        return image, target