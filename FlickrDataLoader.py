
# coding: utf-8
# written by Zhuoran Gu 3/12/2018
# modified by Yang Zhang 3/13/2018

# In[1]:


import os
import pickle
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import PIL


# In[2]:


class Flicker8k(Dataset):
    '''
    Args:
    
        img_dir(string): Location of images
        capt_dir(string): Caption file path. e.g. ./Flicker8k.tocken.txt
        txt_dir(string): Path of Flickr_8k.trainImages.txt
        transform(torch.transform): transform to perform on input image
    '''
    def __init__(self, img_dir, capt_dir, txt_dir, transform=None, train=True):
        self.img_dir = img_dir
        self.capt_dir = capt_dir
        self.transform = transform
        self.train = train
        self.files = {}
        self.caption = {}
        
        with open(txt_dir, "r") as f:
            self.list = f.readlines()
        f.close()

        self.list = [a.replace("\n", "") for a in self.list]
        with open(self.capt_dir, "r") as f:
            while 1:
                line = f.readline()
                if not line:
                    break
                img = line.split("#")[0]
                txt = line.split("#")[1].split("\t")[1][:-1]
                if img not in self.caption:
                    self.caption[img] = [txt]
                else:
                    self.caption[img].append(txt)
        
        for img in self.list:
            self.files[os.path.join(img_dir, img)] = self.caption[img]


    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_name, caption = list(self.files.items())[idx]
        image = PIL.Image.open(img_name).convert("RGB")
        if self.transform: 
            image = self.transform(image)
        sample = (image, caption)
        return sample


# In[3]:


img_dir = r'./data/img/Flicker8k_Dataset/'
cap_path = r'./data/label/Flickr8k.lemma.token.txt'
train_txt = r'./data/label/Flickr_8k.trainImages.txt'
val_txt = r'./data/label/Flickr_8k.devImages.txt'
test_txt = r'./data/label/Flickr_8k.testImages.txt'


# In[4]:


mytransform = transforms.Compose(
            [
                transforms.Scale((224,224)),
                transforms.RandomHorizontalFlip(),
                # (H x W x C) in the range [0, 255] to (C x H x W) in the range [0.0, 1.0].
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ]
        )


# In[5]:


flicker8k_trn = Flicker8k(img_dir, cap_path, train_txt, transform=mytransform, train=True)


# In[6]:


flicker8k_val = Flicker8k(img_dir, cap_path, val_txt, transform=mytransform, train=False)


# In[7]:


flicker8k_tst = Flicker8k(img_dir, cap_path, test_txt, transform=mytransform, train=False)

