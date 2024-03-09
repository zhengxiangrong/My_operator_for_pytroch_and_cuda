import torch
import torch.nn as nn

class MyConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X_input, weight, bias, stride, padding):
        padding_left, padding_right, padding_up, padding_down = padding
        OutChannel,inChannel,h_w,w_w = weight.shape
        h_stride,w_stride=stride
        weight_clone=weight.clone()
        weight_clone = weight_clone[None]
        pad=nn.ZeroPad2d(padding)
        X = pad(X_input)
        batch, channel, h, w = X.shape
        X = X[:,None]
        batch_Y,channel_Y,h_Y,w_Y = batch,OutChannel,(h-h_w)//h_stride+1,(w-w_w)//w_stride+1
        Y = torch.zeros((batch_Y,channel_Y,h_Y,w_Y))
        sum_st = torch.zeros((batch_Y,channel_Y,inChannel,h_Y,w_Y))
        for i in range(w_w):
            for j in range(h_w):
                sum_st += X[:,:,:,j:j+h_stride*h_Y:h_stride,i:i+w_stride*w_Y:w_stride]*weight_clone[:,:,:,j,i,None,None]
        bias = bias[None]
        bias = bias[:,:,None,None]
        backTensor = torch.tensor([h_stride,w_stride,padding_left, padding_right, padding_up, padding_down])
        Y[:,:,:,:] = torch.sum(sum_st,dim=2)+bias
        ctx.save_for_backward(X,weight_clone,backTensor)
        return Y

    @staticmethod
    def backward(ctx,grad_output):
        X,weight,backTensor = ctx.saved_tensors
        _,OutChannel, inChannel, h_w, w_w = weight.shape
        batch, out_channel,channel, h, w = X.shape
        _, grad_output_channel, grad_output_h, grad_output_w = grad_output.shape
        weight_rotate = torch.flip(weight,dims=(3,4))
        weight_change = weight_rotate
        # grad 在相邻元素间插入stride个0,在周围补上weight-1个0
        grad_padding = torch.zeros((batch, grad_output_channel, backTensor[0]*(grad_output_h-1)+1+2*(h_w-1), backTensor[1]*(grad_output_w-1)+1+2*(w_w-1)))
        grad_output_ = grad_output.clone()
        grad_padding[:,:,(h_w-1):backTensor[0]*(grad_output_h-1)+h_w:backTensor[0],(w_w-1):backTensor[1]*(grad_output_w-1)+w_w:backTensor[1]] = grad_output_[:,:,:,:]
        grad_new = grad_padding[:,:,None]
        _,_,_,h_grad_new,w_grad_new = grad_new.shape
        batch_X, channel_X, h_X, w_X = batch, OutChannel, h_grad_new - h_w + 1, w_grad_new - w_w + 1
        sum_for_input = torch.zeros((batch,OutChannel,inChannel,h_X,w_X))

        for i in range(w_w):
            for j in range(h_w):
                # 卷积的stride,在input反向传播时变成导数之间间隔变为stride,反向传播的卷积stride变为1
                sum_for_input += grad_new[:,:,:,j:h_X+j,i:w_X+i]*weight_change[:,:,:,j,i,None,None]
        dx = torch.sum(sum_for_input, dim=1)
        _,_,dx_h,dx_w =dx.shape
        dinput = dx[:,:,backTensor[4]:dx_h-backTensor[5],backTensor[2]:dx_w-backTensor[3]]
        grad_output_new = grad_output.clone()
        # grad 在相邻元素间插入stride个0
        grad_for_weight = torch.zeros((batch, grad_output_channel, backTensor[0]*(grad_output_h-1)+1, backTensor[1]*(grad_output_w-1)+1))
        grad_for_weight[:,:,0:backTensor[0]*(grad_output_h-1)+1:backTensor[0],0:backTensor[1]*(grad_output_w-1)+1:backTensor[1]] = grad_output_new[:,:,:,:]
        ###########

        batch, out_channel, grad_h, grad_w = grad_for_weight.shape
        grad_for_weight = grad_for_weight[:, :, None]
        # 如果用 h_weight, w_weight = (h - grad_h) + 1, (w - grad_w)+ 1，这边就会有可能出现报错，例如输入高和宽分别为2，7,卷积核为2*2，stride=(1,4)这时就会出现权重dweight输出维度为宽高为2*3,
        # 不等于2*2,出现报错
        # print("h", h, "grad_h", grad_h)
        h_new, w_new = ((h - h_w) // backTensor[0]) * backTensor[0] + h_w, ((w - w_w) // backTensor[1]) * backTensor[1] + w_w
        h_weight, w_weight = (h_new - grad_h) + 1, (w_new - grad_w) + 1
        sum_for_weight = torch.zeros((batch,OutChannel,inChannel,h_weight,w_weight))
        for  i in range(grad_w):
            for j in range(grad_h):
                sum_for_weight += X[:,:, :, j:h_weight + j, i:w_weight + i] * grad_for_weight[:, :, :, j, i,None,None]
        dweight = torch.sum(sum_for_weight, dim=(0))

        ###################################
        dbias =torch.sum(grad_output,dim=(0,2,3))
        return dinput, dweight, dbias, None,None

class MyConv2d_package(nn.Module):
    """
    kernal = (k_h, k_w)
    stride = (stride_h, stride_w)
    padding =(padding_left, padding_right, padding_up, padding_down)
    """
    def __init__(self,in_channel,out_channel,kernal, weight,bias, stride=(0,0),padding=(0,0,0,0)):
        super(MyConv2d_package, self).__init__()
        self.weight = weight
        self.bias =bias
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.rand((out_channel,in_channel)+kernal,dtype=torch.float32))
        self.bias = nn.Parameter(torch.rand((out_channel,),dtype=torch.float32))
        self.MyConv2d = MyConv2d.apply

    def forward(self, input):
        # print("MyConv2d_package input.shape",input.shape)
        output = self.MyConv2d(input,self.weight,self.bias,self.stride,self.padding)
        return output

