#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time

import torch

import sys
sys.path.append('../')
from ptsemseg.models import get_model


# In[2]:


n_classes = 19
unet_baseline = get_model({'arch': 'unet_baseline'}, n_classes)
unet_alpha0125 = get_model({'arch': 'unet_octconv', 'alpha': 0.125}, n_classes)
unet_alpha0250 = get_model({'arch': 'unet_octconv', 'alpha': 0.250}, n_classes)
unet_alpha0500 = get_model({'arch': 'unet_octconv', 'alpha': 0.500}, n_classes)
unet_alpha0750 = get_model({'arch': 'unet_octconv', 'alpha': 0.750}, n_classes)
unet_alpha0875 = get_model({'arch': 'unet_octconv', 'alpha': 0.875},n_classes)

input_shape = [3, 224, 224]
dummy_input = torch.zeros(input_shape)
dummy_input = dummy_input.unsqueeze(0)


# In[3]:


def measure(model, dummy_input, trial_num=50, warmup=5, device='cpu'):
    model.eval()
    model = model.to(device)
    dummy_input = dummy_input.to(device)
    
    for _ in range(warmup):
        with torch.no_grad():
            model(dummy_input)
    
    start = time.time()
    for _ in range(trial_num):
        with torch.no_grad():
            model(dummy_input)
    end = time.time()
    
    average = (end - start) / trial_num
    return average


# ## CPU

# In[4]:


kargs_cpu = {'trial_num': 50, 'warmup': 5, 'device': 'cpu'}


# In[5]:


print('cpu, alpha=0.00 [sec/image]: ', measure(unet_baseline, dummy_input, **kargs_cpu))


# In[6]:


print('cpu, alpha=0.125 [sec/image]: ', measure(unet_alpha0125, dummy_input, **kargs_cpu))


# In[7]:


print('cpu, alpha=0.25 [sec/image]: ', measure(unet_alpha0250, dummy_input, **kargs_cpu))


# In[8]:


print('cpu, alpha=0.50 [sec/image]: ', measure(unet_alpha0500, dummy_input, **kargs_cpu))


# In[9]:


print('cpu, alpha=0.75 [sec/image]: ', measure(unet_alpha0750, dummy_input, **kargs_cpu))


# In[10]:


print('cpu, alpha=0.875 [sec/image]: ', measure(unet_alpha0875, dummy_input, **kargs_cpu))


# ## GPU

# In[ ]:


kargs_gpu = {'trial_num': 100, 'warmup': 10, 'device': 'cuda:0'}


# In[ ]:


print('gpu, alpha=0.00 [sec/image]: ', measure(unet_baseline, dummy_input, **kargs_gpu))


# In[ ]:


print('gpu, alpha=0.125 [sec/image]: ', measure(unet_alpha0125, dummy_input, **kargs_gpu))


# In[ ]:


print('gpu, alpha=0.25 [sec/image]: ', measure(unet_alpha0250, dummy_input, **kargs_gpu))


# In[ ]:


print('gpu, alpha=0.50 [sec/image]: ', measure(unet_alpha0500, dummy_input, **kargs_gpu))


# In[ ]:


print('gpu, alpha=0.75 [sec/image]: ', measure(unet_alpha0750, dummy_input, **kargs_gpu))


# In[ ]:


print('gpu, alpha=0.875 [sec/image]: ', measure(unet_alpha0875, dummy_input, **kargs_gpu))


# In[ ]:




