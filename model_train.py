
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import FlickrDataLoader
import torchvision.transforms as transforms
import PIL
from model import CaptionModel

def train(epoches):
    for epoch in range(epoches):
        loss_ = 0
        model.train()
        for i in range(6000):
            model.zero_grad()

            c = Variable(torch.FloatTensor(feat_tr[i:i+1])).cuda()
            cap_id = np.random.randint(5,size=1)[0]
            cap = caption_trn[i][cap_id]
            inp = [5832] + cap
            tchr = cap + [5833]
            d = Variable(torch.LongTensor(np.array(inp))).cuda().unsqueeze(0)
            tt = Variable(torch.LongTensor(np.array(tchr))).cuda().unsqueeze(0)


            logt,alphas,betas = model(c, d, False,False)
            loss = criterion(logt.view(-1, model.V), tt.view(-1))
            loss.backward()
            optimizer.step()
            loss_ += loss.data.cpu().numpy()[0]
            if i % 500 == 499:
                print( '[{} {}] loss: {}'.format(epoch, i+1, loss_*1.0/500))
                loss_ = 0
        loss_ = 0
        for i in range(1000):
            model.eval()

            c = Variable(torch.FloatTensor(feat_val[i:i+1]), volatile=True).cuda()
            cap_id = np.random.randint(5,size=1)[0]
            cap = caption_val[i][cap_id]
            inp = [5832] + cap
            tchr = cap + [5833]
            d = Variable(torch.LongTensor(np.array(inp))).cuda().unsqueeze(0)
            tt = Variable(torch.LongTensor(np.array(tchr))).cuda().unsqueeze(0)
            logt,alphas,betas = model(c, d, False,False)
            loss = criterion(logt.view(-1, model.V), tt.view(-1))
            loss_ += loss.data.cpu().numpy()[0]
        print( 'Epoch {}: validation loss: {}'.format(epoch, loss_*1.0/1000))

    return 0

if __name__ == '__main__':
    with open('dict6k.pkl','r') as f:
        (w2id, id2w) = pickle.load(f)
    img_dir = r'./data/img/Flicker8k_Dataset/'
    cap_path = r'./data/label/Flickr8k.lemma.token.txt'
    train_txt = r'./data/label/Flickr_8k.trainImages.txt'
    val_txt = r'./data/label/Flickr_8k.devImages.txt'
    test_txt = r'./data/label/Flickr_8k.testImages.txt'

    mytransform = transforms.Compose(
                [
                    transforms.Scale((224,224)),
                    transforms.ToTensor(),
                ]
            )
    flicker8k_trn = FlickrDataLoader.Flicker8k(img_dir, cap_path, train_txt, transform=mytransform, train=True)
    flicker8k_val = FlickrDataLoader.Flicker8k(img_dir, cap_path, val_txt, transform=mytransform, train=True)
    with open('feat6k.npy','r') as f:
        feat_tr = np.load(f)

    with open('capt6k.pkl','r') as f:
        caption_trn = pickle.load(f)
        
    with open('feat.pkl','r') as f:
        feat_val = pickle.load(f)

    with open('capt1k.pkl','r') as f:
        caption_val = pickle.load(f)
    
    model = CaptionModel(bsz=1, feat_dim=(196, 512), n_voc=5834, n_embed=512, n_hidden=1024).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train(epoches =1)
    with open('model_t.pth','r') as f:
        model.load_state_dict(torch.load(f))
