#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from model.dcn  import  DCNv2, DCN
# from dcn_v2 import dcn_v2_pooling, DCNv2Pooling, DCNPooling

deformable_groups = 1
N, inC, inH, inW = 2, 2, 4, 4
outC = 2
kH, kW = 3, 3

def example_dconv():
    input = torch.randn(2, 64, 128, 128).cuda()
    # wrap all things (offset and mask) in DCN
    dcn = DCN(64, 64, kernel_size=(3, 3), stride=1,
              padding=1).cuda()
    # print(dcn.weight.shape, input.shape)
    output = dcn(input)
    print("output.shape", output.shape)
    targert = output.new(*output.size())
    targert.data.uniform_(-0.01, 0.01)
    error = (targert - output).mean()
    error.backward()



if __name__ == '__main__':

    example_dconv()