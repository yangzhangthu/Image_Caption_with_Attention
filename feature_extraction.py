
# coding: utf-8

# In[ ]:


import numpy as np
from PIL import Image
import torch
import os
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
import vgg
import time
import pickle
import FlickrDataLoader
import torchvision.transforms as transforms
import argparse


class VggConv(nn.Module):
    def __init__(self, model):
        super(VggConv, self).__init__()
        self.features = nn.Sequential(
            *list(model.features.children())[:-3]
        )
    def forward(self, x):
        x = self.features(x)
        return x

def get_transform():
    mytransform = transforms.Compose(
        [
            transforms.Scale((224,224)),
            transforms.RandomHorizontalFlip(),
            # (H x W x C) in the range [0, 255] to (C x H x W) in the range [0.0, 1.0].
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    return mytransform

def feature_extract(data_loader, is_dict=True, file_name='out'):
    feat = None
    w2id = {}
    count = 0
    for i, data in enumerate(data_loader, 0):

        inputs, capt = data
        inputs = Variable(inputs, volatile=True).cuda()

        out_ = model_conv(inputs)

        if feat is None:
            feat = out_.view(bsz,512,-1).transpose(1,2)
        else:
            feat = torch.cat((feat, out_.view(out_.data.shape[0], 512, -1).transpose(1,2)), 0)
        
        if i%100 == 99:
            print('{} batches done!'.format(i+1))
        
        if is_dict:
            for j in range(5):
                for k in range(out_.data.shape[0]):
                    tmp = capt[j][k].lower().split(' ')
                    for l in tmp:
                        if l not in w2id:
                            w2id[l] = count
                            count += 1
    
    print(feat.data.shape)
    with open('feat_'+file_name+'.npy', 'w') as f:
        np.save(f, feat.data.cpu().numpy())
    if is_dict:
        voc = len(w2id)
        w2id['<start>'] = voc
        w2id['<end>'] = voc+1
        w2id['<null>'] = voc+2
        id2w = {}
        for (w,idx) in w2id.iteritems():
            id2w[idx] = w
        with open('dict_'+file_name+'.pkl','w') as f:
            pickle.dump((w2id, id2w), f)
            print(len(w2id))
    

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgdir', default=r'./data/img/Flicker8k_Dataset/', type=str, help='path to the images')
    parser.add_argument('--tokentxt', default=r'./data/label/Flickr8k.lemma.token.txt', 
                        type=str, help='path to Flickr8k.lemma.token.txt')
    parser.add_argument('--imgtxt', default=r'./data/label/Flickr_8k.trainImages.txt', 
                        type=str, help='path to Flickr_8k.xxxImages.txt')
    parser.add_argument('--bsz', default=64, type=int, help='batch size')
    parser.add_argument('--dict', default=True, type=bool, help='True if it\' used for training')
    parser.add_argument('--fname', default='out', type=str, help='name of output')
    args = parser.parse_args()
    
    bsz = args.bsz
    imgdir = args.imgdir
    tokentxt = args.tokentxt
    imgtxt = args.imgtxt
    is_dict = args.dict
    fname = args.fname
    
    flicker8k = FlickrDataLoader.Flicker8k(imgdir, tokentxt, imgtxt, transform=get_transform(), train=True)
    model = vgg.vgg16_bn(True)
    model.eval()
    model_conv = VggConv(model).cuda()

    trainloader = torch.utils.data.DataLoader(flicker8k, batch_size=bsz,
                                              shuffle=False, num_workers=2)
    
    feature_extract(trainloader, is_dict, fname)
