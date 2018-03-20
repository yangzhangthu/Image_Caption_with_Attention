
# coding: utf-8

# In[ ]:


import numpy as np
import time
import torch
import FlickrDataLoader
import pickle
import argparse
import torchvision.transforms as transforms

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

def capt_coder(data_loader, w2id, fname):
    caption = []
    count = 0
    for i, data in enumerate(data_loader, 0):
        inputs, capt = data
        for k in range(inputs.shape[0]):
            cap = []
            for j in range(5):
                tmp = capt[j][k].lower().split(' ')
                a = []
                for l in tmp:
                    if l in w2id:
                        a.append(w2id[l])
                    else:
                        a.append(w2id['<null>'])
                cap.append(a)
            caption.append(cap)
            count+=1

    print(count)
    with open(fname+'.pkl','w') as f:
        pickle.dump(caption, f)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgdir', default=r'./data/img/Flicker8k_Dataset/', type=str, help='path to the images')
    parser.add_argument('--tokentxt', default=r'./data/label/Flickr8k.lemma.token.txt', 
                        type=str, help='path to Flickr8k.lemma.token.txt')
    parser.add_argument('--imgtxt', default=r'./data/label/Flickr_8k.trainImages.txt', 
                        type=str, help='path to Flickr_8k.xxxImages.txt')
    parser.add_argument('--dictdir', default='dict_out.pkl', type=str, help='path to the dictionary')
    parser.add_argument('--fname', default='out', type=str, help='name of output')
    args = parser.parse_args()
    
    imgdir = args.imgdir
    tokentxt = args.tokentxt
    imgtxt = args.imgtxt
    dictdir = args.dictdir
    fname = args.fname
    
    flicker8k = FlickrDataLoader.Flicker8k(imgdir, tokentxt, imgtxt, transform=get_transform(), train=True)

    dataloader = torch.utils.data.DataLoader(flicker8k, batch_size=64,
                                              shuffle=False, num_workers=2)
    
    with open(dictdir,'r') as f:
        (w2id, id2w) = pickle.load(f)
    
    capt_coder(dataloader, w2id, fname)

