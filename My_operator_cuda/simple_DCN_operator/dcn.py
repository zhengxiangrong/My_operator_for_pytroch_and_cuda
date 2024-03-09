#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import math

import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

import _ext as _backend


class _DCNv2(Function):
    @staticmethod
    def forward(
            ctx, input, offset, weight, bias, stride, padding, dilation
    ):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(weight.shape[2:4])
        output, offset_output = _backend.dcn_forward(
            input,
            weight,
            bias,
            offset,
            ctx.kernel_size[0],
            ctx.kernel_size[1],
            ctx.stride[0],
            ctx.stride[1],
            ctx.padding[0],
            ctx.padding[1],
            ctx.dilation[0],
            ctx.dilation[1],
        )
        print("offset_output.shape", offset_output.shape)
        ctx.save_for_backward(input, offset, weight, bias, offset_output)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight, bias, offset_output = ctx.saved_tensors

        input_grad, offset_grad, weight_grad, bias_grad = _backend.dcn_backward(
            grad_output,
            offset_output,
            input,
            weight,
            bias,
            offset,
            ctx.kernel_size[0],
            ctx.kernel_size[1],
            ctx.stride[0],
            ctx.stride[1],
            ctx.padding[0],
            ctx.padding[1],
            ctx.dilation[0],
            ctx.dilation[1],
        )

        return input_grad, offset_grad, weight_grad, bias_grad, None, None, None

    @staticmethod
    def symbolic(
            g, input, offset, weight, bias, stride, padding, dilation
    ):
        from torch.nn.modules.utils import _pair

        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        # as of trt 7, the dcn operation will be translated again by modifying the onnx file
        # so the exporting code is kept to resemble the forward()
        return g.op(
            "DCN_Op",
            input,
            offset,
            weight,
            bias,
            stride_i=stride,
            padding_i=padding,
            dilation_i=dilation,
        )


dcn_conv = _DCNv2.apply


class DCNv2(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=1,
    ):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        assert (
                2 * self.kernel_size[0] * self.kernel_size[1]
                == offset.shape[1]
        )
        return dcn_conv(
            input,
            self.weight,
            self.bias,
            offset,
            self.stride,
            self.padding,
            self.dilation
        )


class DCN(DCNv2):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=1,
    ):
        super(DCN, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation
        )

        channels_ = 2 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            channels_,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        offset = self.conv_offset_mask(input)
        print("offset.shape",offset.shape)
        print("111input.shape", input.shape)
        print("111self.weight.shape", self.weight.shape)
        print("111self.bias.shape", self.bias.shape)
        return dcn_conv(
            input,
            offset,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
        )
