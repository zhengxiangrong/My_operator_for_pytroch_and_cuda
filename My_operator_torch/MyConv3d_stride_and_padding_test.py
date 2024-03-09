import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn

class MyConv3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X_input, weight, bias, stride, padding):
        OutChannel, inChannel, d_w, h_w, w_w = weight.shape
        d_stride, h_stride, w_stride =stride
        padding_forward,padding_backward,padding_up,padding_down,padding_left,padding_right= padding
        weight_clone = weight.clone()
        weight_clone = weight_clone[None]
        X=torch.nn.functional.pad(X_input, pad=(padding_left, padding_right,padding_up, padding_down,padding_forward, padding_backward), mode='constant', value=0)
        batch, channel, d, h, w = X.shape
        X = X[:, None]
        batch_Y, channel_Y, d_Y, h_Y, w_Y = batch, OutChannel, (d - d_w)//d_stride + 1, (h - h_w)//h_stride + 1, (w - w_w)//w_stride + 1
        Y = torch.zeros((batch_Y, channel_Y, d_Y, h_Y, w_Y))
        # sum_st = torch.zeros((batch_Y, channel_Y, inChannel, d_Y, h_Y, w_Y))
        x_append = []
        w_append = []
        for i in range(h_w):
            for j in range(d_w):
                for k in range(w_w):
                    # print("X[:,:,:,j:j+d_stride*h_Y:d_stride,i:i+h_stride*h_Y:h_stride].shape",X[:,:,:,j:j+d_stride*h_Y:d_stride,i:i+h_stride*h_Y:h_stride].shape)
                    # X[:,:,:,j:j+d_stride*h_Y:d_stride,i:i+h_stride*h_Y:h_stride]*weight_clone[:,:,:,j,i,None,None]
                    x_append.append(X[:, :, :, j:d_stride*d_Y+j:d_stride, i:h_stride*h_Y + i:h_stride, k:k + w_stride*w_Y:w_stride,None])
                    w_append.append(weight_clone[:, :, :, j, i, k, None, None,None,None])
                    # sum_st += X[:, :, :, j:d_stride*d_Y+j:d_stride, i:h_stride*h_Y + i:h_stride, k:k + w_stride*w_Y:w_stride] * weight_clone[:, :, :, j, i, k, None, None,None]
        x_tensor = torch.concatenate(x_append,dim=6)
        w_tensor = torch.concatenate(w_append,dim=6)
        # torch.squeeze(torch.sum(x_tensor*w_tensor,dim=6),dim=6)
        bias = bias[None]
        bias = bias[:, :, None, None, None]
        # print(x_tensor.shape)
        # print(torch.sum(x_tensor*w_tensor,dim=6).shape)
        # print(bias.shape)
        Y[:, :, :, :, :] = torch.sum(x_tensor*w_tensor,dim=[2,6]) + bias
        backTensor = torch.tensor([d_stride, h_stride, w_stride, padding_left, padding_right, padding_up, padding_down,padding_forward,padding_backward])
        ctx.save_for_backward(X, weight_clone,backTensor)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        X, weight,backTensor = ctx.saved_tensors
        _, OutChannel, inChannel, d_w, h_w, w_w = weight.shape
        batch, out_channel, channel, d, h, w = X.shape
        _, grad_output_channel, grad_output_d, grad_output_h, grad_output_w = grad_output.shape
        weight_rotate = torch.flip(weight, dims=(3, 4, 5))
        weight_change = weight_rotate
        grad_padding = torch.zeros((batch, grad_output_channel, backTensor[0]*(grad_output_d-1)+1+2*(d_w-1), backTensor[1]*(grad_output_h-1)+1+2*(h_w-1), backTensor[2]*(grad_output_w-1)+1+2*(w_w-1)))
        # grad_padding = torch.zeros((batch, grad_output_channel, grad_output_d + 2 * (d_w - 1),grad_output_h + 2 * (h_w - 1), grad_output_w + 2 * (w_w - 1)))
        grad_output_ = grad_output.clone()
        grad_padding[:,:,(d_w-1):backTensor[0]*(grad_output_d-1)+d_w:backTensor[0],(h_w-1):backTensor[1]*(grad_output_h-1)+h_w:backTensor[1],(w_w-1):backTensor[2]*(grad_output_w-1)+w_w:backTensor[2]] = grad_output_[:,:,:,:,:]

        # grad_padding[:, :, (d_w - 1):grad_output_d + (d_w - 1), (h_w - 1):grad_output_h + (h_w - 1),(w_w - 1):grad_output_w + (w_w - 1)] = grad_output_[:, :, :, :, :]
        grad_new = grad_padding[:, :, None]
        _, _, _, d_grad_new, h_grad_new, w_grad_new_w = grad_new.shape
        batch_X, channel_X, d_X, h_X, w_X = batch, OutChannel, d_grad_new - d_w + 1, h_grad_new - h_w + 1, w_grad_new_w - w_w + 1
        sum_for_input = torch.zeros((batch, OutChannel, inChannel, d_X, h_X, w_X))
        grad_new_append = []
        weight_change_append = []

        for i in range(h_w):
            for j in range(d_w):
                for k in range(w_w):
                    grad_new_append.append(grad_new[:, :, :, j:d_X + j, i:h_X + i, k:w_X + k])
                    grad_new_append.append(weight_change[:, :, :, j, i,k, None, None, None])
                    # sum_for_input += grad_new[:, :, :, j:d_X + j, i:h_X + i, k:w_X + k] * weight_change[:, :, :, j, i,k, None, None, None]
        dx = torch.sum(sum_for_input, dim=1)
        _, _, dx_d, dx_h,dx_w = dx.shape
        # backTensor = torch.tensor([d_stride, h_stride, w_stride, padding_left, padding_right, padding_up, padding_down,padding_forward,padding_backward])
        dinput = dx[:, :, backTensor[7]:dx_d - backTensor[8], backTensor[5]:dx_h - backTensor[6], backTensor[3]:dx_w - backTensor[4]]

        # batch, out_channel, grad_d, grad_h, grad_w = grad_output.shape
        grad_output_new = grad_output.clone()
        # grad_output_new = grad_output_new[:, :, None]
        # d_weight, h_weight, w_weight = h - grad_d + 1, w - grad_h + 1, d - grad_w + 1
        grad_for_weight = torch.zeros((batch, grad_output_channel, backTensor[0]*(grad_output_d-1)+1, backTensor[1]*(grad_output_h-1)+1, backTensor[2]*(grad_output_w-1)+1))
        grad_for_weight[:, :, 0:backTensor[0] * (grad_output_d - 1) + 1:backTensor[0],0:backTensor[1] * (grad_output_h - 1) + 1:backTensor[1],0:backTensor[2] * (grad_output_w - 1) + 1:backTensor[2]] = grad_output_new[:, :, :, :,:]

        d_new, h_new, w_new = ((d - d_w) // backTensor[0]) * backTensor[0] + d_w, ((h - h_w) // backTensor[1]) * backTensor[1] + h_w, ((w - w_w) // backTensor[2]) * backTensor[2] + w_w

        batch, out_channel, grad_d, grad_h, grad_w = grad_for_weight.shape
        # print("d , w_w ,grad_w", d, w_w, grad_w)
        grad_for_weight = grad_for_weight[:, :, None]
        d_weight, h_weight, w_weight = (d_new - grad_d) + 1, (h_new - grad_h) + 1, (w_new - grad_w) + 1

        sum_for_weight = torch.zeros((batch, OutChannel, inChannel, d_weight, h_weight, w_weight))
        for i in range(grad_h):
            for j in range(grad_d):
                for k in range(grad_w):
                    sum_for_weight += X[:, :, :, j:d_weight + j, i:h_weight + i, k:w_weight + k] * grad_for_weight[:, :,:, j, i, k, None, None,None]
        dweight = torch.sum(sum_for_weight, dim=(0))

        dbias = torch.sum(grad_output, dim=(0, 2, 3, 4))
        return dinput, dweight, dbias, None, None


class MyConv3d_package(nn.Module):
    def __init__(self, in_channel, out_channel, kernal,weight,bias,stride,padding):
        super(MyConv3d_package, self).__init__()
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.stride = stride
        self.padding = padding
        self.MyConv3d = MyConv3d.apply

    def forward(self, input):
        output = self.MyConv3d(input, self.weight, self.bias,self.stride, self.padding)
        return output

#  (padD, padH, padW)
# 3D卷积, 一般是在处理的视频的时候才会使用，目的是为了提取时序信息(temporal feature)，输入的size是(N,Cin,D,H,W)，输出size是(N,Cout,Dout,Hout,Wout)
conv3d=torch.nn.Conv3d(4,4,(3,1,1),stride=(1,1,1),padding=(1,0,0))



class Net(nn.Module):
    def __init__(self,in_channel, out_channel,kernel, weight, bias,stride,padding):
        super(Net,self).__init__()
        self.MyConv3d_package = MyConv3d_package(in_channel, out_channel,kernel, weight, bias, stride ,padding)

    def forward(self,input):
        out = self.MyConv3d_package(input)
        # print("out_new", np.shape(out_new), out_new)
        return out


model=Net(4,4,(3,1,1),conv3d.weight.clone(),conv3d.bias.clone(),(1,1,1),padding=(1,1,0,0,0,0))


lossfunc3d = torch.nn.MSELoss()
lossfunc2d = torch.nn.MSELoss()

update3d=optim.Adam(conv3d.parameters(), lr=0.01)
update2d=optim.Adam(model.parameters(), lr=0.01)

for i in range(20):
    input = torch.rand([4, 4, 5, 8, 8])
    y_out=torch.rand([4,4,5,8,8])
    old_weight = conv3d.weight
    # print("old weight",old_weight)
    ouput3d = conv3d(input)
    loss3d=lossfunc3d(ouput3d,y_out)
    loss3d.backward()
    update3d.step()
    update3d.zero_grad()

    out_new = model(input)
    loss2d=lossfunc2d(out_new,y_out)
    loss2d.backward()
    update2d.step()
    update2d.zero_grad()

    # print("ouput3d",ouput3d)
    # print("out_new",out_new)
    # new_conv2d_weight = model.conv2d.weight[:,:,None,:,:]
    # new_conv2d_weight_ = model.conv2d_.weight[:,:,None,:,:]
    # new_conv2d_w=torch.cat((new_conv2d_weight,new_conv2d_weight_),dim=2)
    print("ouput3d \n",ouput3d)
    print("out_new \n",out_new)
    # print(out_new==ouput3d)
