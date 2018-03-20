
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
get_ipython().magic(u'matplotlib inline')

class CaptionModel(nn.Module):
    def __init__(self, bsz, feat_dim, n_voc, n_embed, n_hidden):
        super(CaptionModel, self).__init__()
        self.N = bsz
        self.L = feat_dim[0]
        self.C = feat_dim[1]
        self.V = n_voc
        self.M = n_embed
        self.H = n_hidden
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.bn = nn.BatchNorm1d(self.L)
        
        self.init_h = nn.Linear(self.C, self.H)
        self.init_c = nn.Linear(self.C, self.H)
        
        self.proj = nn.Linear(self.C, self.C, bias=False)
        
        self.att0 = nn.Linear(self.H, self.C, bias=False)
        self.att1 = nn.Linear(self.C, 1)
        
        self.slct = nn.Linear(self.H, 1)
        
        self.encoder = nn.Embedding(self.V, self.M)
        self.rnn = nn.LSTMCell(self.M + self.C, self.H)
        
        self.fc1 = nn.Linear(self.H, self.M)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(self.M, self.V)
        
        self.ctx2out = nn.Linear(self.C, self.M, bias=False)
        
        
    def init_hidden(self, feat):
        feat_mean = torch.mean(feat,dim=1).squeeze(1)
        h0 = self.tanh(self.init_h(feat_mean))
        c0 = self.tanh(self.init_c(feat_mean))
        return (h0, c0)
    
    def lyr_att(self, h, feat, pctx_):
        pctx = self.tanh(self.att0(h).unsqueeze(1) + pctx_)
        out_att = self.att1(pctx.view(-1, self.C)).view(-1, self.L)
        alpha = self.softmax(out_att)
        ctx = torch.sum((alpha.unsqueeze(2) * feat),dim=1).squeeze(1)
        return alpha, ctx, pctx_
    
    def lyr_slct(self, h, ctx):
        beta = self.sigmoid(self.slct(h))
        ctx_ = beta*ctx
        return beta, ctx_
    
    def decoder(self, h, ctx_, x, debug):
        out1 = self.fc1(h) + self.ctx2out(ctx_)
        if debug:
            print out1
            print x
        out2 = self.tanh(out1 + x)
        out2 = self.dropout(out2)
        logit = self.fc2(out2)
        if debug:
            print logit
        return logit
    
    def sample_distr(self, distr, temperature):
        distr = distr*1.0/temperature
        distr = np.exp(distr)
        distr = distr*1.0/np.sum(distr)
        idx = np.random.choice(self.V, 1, p=distr)
        return idx[0]
    
    def forward(self, feat, capt, debug=False, debug2=False):
        
        feat = self.bn(feat)
        h, c = self.init_hidden(feat)
        pctx_ = self.tanh(self.proj(feat))

        seq_len = capt.data.shape[1]
        logit = Variable(torch.zeros((capt.data.shape[0],capt.data.shape[1],self.V))).cuda()
        
        if debug:
            print('output size is {}'.format(logit.data.shape))
        
        alphas = []
        betas = []
        
        for t in range(seq_len):

            x = self.encoder(capt[:,t])
            if debug:
                print('capt size is {}'.format(capt[:,t].data.shape))
                print('x size is {}'.format(x.data.shape))
        
            alpha, ctx, pctx_ = self.lyr_att(h, feat, pctx_)
            if debug:
                print('alpha size is {}'.format(alpha.data.shape))
                print('context size is {}'.format(ctx.data.shape))
            alphas.append(alpha.view(-1,14,14).data.cpu().numpy())
            ctx_ = ctx
#             beta, ctx_ = self.lyr_slct(h, ctx)
#             if debug:
#                 print('beta size is {}'.format(beta.data.shape))
#                 print('context_ size is {}'.format(ctx_.data.shape))
#             betas.append(beta.data.cpu().numpy()[0][0])
            
            input_ = torch.cat((x,ctx_), 1)
            if debug:
                print('input_ size is {}'.format(input_.data.shape))

            h, c = self.rnn(input_, (h,c))

            out = self.decoder(h, ctx_, x, debug2).unsqueeze(1)
            if debug:
                print('output of decode size is {}'.format(out.data.shape))
            
            logit[:,t,:] = out
            
        return logit, alphas,betas
    
    def predict(self, feat, max_len=20, debug=False):
        
        feat = self.bn(feat)
        h, c = self.init_hidden(feat)
        pctx_ = self.tanh(self.proj(feat))

        alphas = []
        betas = []
        
        capt = Variable(torch.LongTensor(np.array([5832]))).cuda()
        if debug:
            print('capt size is {}'.format(capt.data.shape))
        
        count = 0
        preds = []
        
        while True:

            x = self.encoder(capt)
            if debug:
                print('x size is {}'.format(x.data.shape))
        
            alpha, ctx, pctx_ = self.lyr_att(h, feat, pctx_)
            if debug:
                print('alpha size is {}'.format(alpha.data.shape))
                print('context size is {}'.format(ctx.data.shape))
            alphas.append(alpha.view(-1,14,14).data.cpu().numpy())
            ctx_ = ctx
#             beta, ctx_ = self.lyr_slct(h, ctx)
#             if debug:
#                 print('beta size is {}'.format(beta.data.shape))
#                 print('context_ size is {}'.format(ctx_.data.shape))
#             betas.append(beta.data.cpu().numpy()[0][0])
            
            input_ = torch.cat((x,ctx_), 1)
            if debug:
                print('input_ size is {}'.format(input_.data.shape))

            h, c = self.rnn(input_, (h,c))

            out = self.decoder(h, ctx_, x, False).unsqueeze(1)
            if debug:
                print('output of decode size is {}'.format(out.data.shape))
            
            a = self.sample_distr(out.data.cpu()[0][0].numpy(), 0.5)

            capt = Variable(torch.LongTensor(np.array([a]))).cuda()
            
            preds.append(a)
            
            count += 1
            if count==max_len:
                break
    
        return alphas,betas,preds

