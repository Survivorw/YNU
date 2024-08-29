from pathlib import Path
import copy

from functools import partial, wraps

from vector_quantize_pytorch import VectorQuantize as VQ

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.autograd import grad as torch_grad

import torchvision

from einops import rearrange, repeat

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)
if __name__=='__main__':
    a=group_dict_by_key
    print(a)
