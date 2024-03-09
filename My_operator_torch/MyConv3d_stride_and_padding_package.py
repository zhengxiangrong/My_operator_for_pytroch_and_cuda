import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn

class MyConv3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X_input, weight, bias, stride, padding):
        OutChannel, inChannel, h_w, w_w, d_w = weight.shape
        h_stride, w_stride, d_stride =stride
        padding_left,padding_right,padding_up,padding_down,padding_forward,padding_backward = padding
        weight_clone = weight.clone()
        weight_clone = weight_clone[None]
        X=torch.nn.functional.pad(X_input, pad=(padding_forward, padding_backward, padding_up, padding_down, padding_left, padding_right), mode='constant', value=0)
        batch, channel, h, w, d = X.shape
        X = X[:, None]
        batch_Y, channel_Y, h_Y, w_Y, d_Y = batch, OutChannel, (h - h_w)//h_stride + 1, (w - w_w)//w_stride + 1, (d - d_w)//d_stride + 1
        Y = torch.zeros((batch_Y, channel_Y, h_Y, w_Y, d_Y))
        sum_st = torch.zeros((batch_Y, channel_Y, inChannel, h_Y, w_Y, d_Y))
        for i in range(w_w):
            for j in range(h_w):
                for k in range(d_w):
                    # X[:,:,:,j:j+h_stride*h_Y:h_stride,i:i+w_stride*w_Y:w_stride]*weight_clone[:,:,:,j,i,None,None]
                    sum_st += X[:, :, :, j:h_stride*h_Y+j:h_stride, i:w_stride*w_Y + i:w_stride, k:k + d_stride*d_Y:d_stride] * weight_clone[:, :, :, j, i, k, None, None,None]
        bias = bias[None]
        bias = bias[:, :, None, None, None]
        Y[:, :, :, :, :] = torch.sum(sum_st, dim=2) + bias
        backTensor = torch.tensor([h_stride, w_stride, d_stride, padding_left, padding_right, padding_up, padding_down,padding_forward,padding_backward])
        ctx.save_for_backward(X, weight_clone,backTensor)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        X, weight,backTensor = ctx.saved_tensors
        _, OutChannel, inChannel, h_w, w_w, d_w = weight.shape
        batch, out_channel, channel, h, w, d = X.shape
        _, grad_output_channel, grad_output_h, grad_output_w, grad_output_d = grad_output.shape
        weight_rotate = torch.flip(weight, dims=(3, 4, 5))
        weight_change = weight_rotate
        grad_padding = torch.zeros((batch, grad_output_channel, backTensor[0]*(grad_output_h-1)+1+2*(h_w-1), backTensor[1]*(grad_output_w-1)+1+2*(w_w-1), backTensor[2]*(grad_output_d-1)+1+2*(d_w-1)))
        # grad_padding = torch.zeros((batch, grad_output_channel, grad_output_h + 2 * (h_w - 1),grad_output_w + 2 * (w_w - 1), grad_output_d + 2 * (d_w - 1)))
        grad_output_ = grad_output.clone()
        grad_padding[:,:,(h_w-1):backTensor[0]*(grad_output_h-1)+h_w:backTensor[0],(w_w-1):backTensor[1]*(grad_output_w-1)+w_w:backTensor[1],(d_w-1):backTensor[2]*(grad_output_d-1)+d_w:backTensor[2]] = grad_output_[:,:,:,:,:]

        # grad_padding[:, :, (h_w - 1):grad_output_h + (h_w - 1), (w_w - 1):grad_output_w + (w_w - 1),(d_w - 1):grad_output_d + (d_w - 1)] = grad_output_[:, :, :, :, :]
        grad_new = grad_padding[:, :, None]
        _, _, _, h_grad_new, w_grad_new, d_grad_new = grad_new.shape
        batch_X, channel_X, h_X, w_X, d_X = batch, OutChannel, h_grad_new - h_w + 1, w_grad_new - w_w + 1, d_grad_new - d_w + 1
        sum_for_input = torch.zeros((batch, OutChannel, inChannel, h_X, w_X, d_X))
        for i in range(w_w):
            for j in range(h_w):
                for k in range(d_w):
                    sum_for_input += grad_new[:, :, :, j:h_X + j, i:w_X + i, k:d_X + k] * weight_change[:, :, :, j, i,k, None, None, None]
        dx = torch.sum(sum_for_input, dim=1)
        _, _, dx_h, dx_w,dx_d = dx.shape
        dinput = dx[:, :, backTensor[5]:dx_h - backTensor[6], backTensor[3]:dx_w - backTensor[4], backTensor[7]:dx_d - backTensor[8]]

        # batch, out_channel, grad_h, grad_w, grad_d = grad_output.shape
        grad_output_new = grad_output.clone()
        # grad_output_new = grad_output_new[:, :, None]
        # h_weight, w_weight, d_weight = h - grad_h + 1, w - grad_w + 1, d - grad_d + 1
        grad_for_weight = torch.zeros((batch, grad_output_channel, backTensor[0]*(grad_output_h-1)+1, backTensor[1]*(grad_output_w-1)+1, backTensor[2]*(grad_output_d-1)+1))
        grad_for_weight[:, :, 0:backTensor[0] * (grad_output_h - 1) + 1:backTensor[0],0:backTensor[1] * (grad_output_w - 1) + 1:backTensor[1],0:backTensor[2] * (grad_output_d - 1) + 1:backTensor[2]] = grad_output_new[:, :, :, :,:]

        h_new, w_new, d_new = ((h - h_w) // backTensor[0]) * backTensor[0] + h_w, ((w - w_w) // backTensor[1]) * backTensor[1] + w_w, ((d - d_w) // backTensor[2]) * backTensor[2] + d_w

        batch, out_channel, grad_h, grad_w, grad_d = grad_for_weight.shape
        # print("d , d_w ,grad_d", d, d_w, grad_d)
        grad_for_weight = grad_for_weight[:, :, None]
        h_weight, w_weight, d_weight = (h_new - grad_h) + 1, (w_new - grad_w) + 1, (d_new - grad_d) + 1

        sum_for_weight = torch.zeros((batch, OutChannel, inChannel, h_weight, w_weight, d_weight))
        for i in range(grad_w):
            for j in range(grad_h):
                for k in range(grad_d):
                    sum_for_weight += X[:, :, :, j:h_weight + j, i:w_weight + i, k:d_weight + k] * grad_for_weight[:, :,:, j, i, k, None, None,None]
        dweight = torch.sum(sum_for_weight, dim=(0))

        dbias = torch.sum(grad_output, dim=(0, 2, 3, 4))
        return dinput, dweight, dbias, None, None

class MyConv3d_package(nn.Module):
    """
    kernal = (k_h, k_w, k_d)
    stride = (stride_h, stride_w, stride_d)
    padding =(padding_left, padding_right, padding_up, padding_down, padding_forward, padding_backward)
    """
    def __init__(self, in_channel, out_channel, kernal, stride=(0,0,0) ,padding=(0,0,0,0,0,0)):
        super(MyConv3d_package, self).__init__()
        self.weight = nn.Parameter(torch.normal(0, 0.5, (out_channel, in_channel) + kernal, dtype=torch.float32))
        self.bias = nn.Parameter(torch.normal(0, 0.5, (out_channel,), dtype=torch.float32))
        self.MyConv3d = MyConv3d.apply
        self.stride = stride
        self.padding = padding

    def forward(self, input):
        output = self.MyConv3d(input, self.weight, self.bias, self.stride, self.padding)
        return output