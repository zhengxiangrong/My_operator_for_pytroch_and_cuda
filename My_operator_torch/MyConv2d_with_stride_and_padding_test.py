import torch
import torch.nn as nn
import torch.optim as optim
import inspect



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
        # print("Y.shape",Y.shape)
        # print("sum_st.shape", sum_st.shape)
        # print("bias.shape", bias.shape)
        backTensor = torch.tensor([h_stride,w_stride,padding_left, padding_right, padding_up, padding_down])
        Y[:,:,:,:] = torch.sum(sum_st,dim=2)+bias
        ctx.save_for_backward(X,weight_clone,backTensor)
        print("Y.shape",Y.shape)
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
        ###################

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
    def __init__(self,in_channel,out_channel,kernal, weight,bias, stride,padding):
        super(MyConv2d_package, self).__init__()
        self.weight = weight
        self.bias =bias
        self.stride = stride
        self.padding = padding
        # self.weight = nn.Parameter(torch.rand((out_channel,in_channel)+kernal,dtype=torch.float32))
        # self.bias = nn.Parameter(torch.rand((out_channel,),dtype=torch.float32))
        self.MyConv2d = MyConv2d.apply

    def forward(self, input):
        # print("MyConv2d_package input.shape",input.shape)
        output = self.MyConv2d(input,self.weight,self.bias,self.stride,self.padding)
        return output


class MyBatchNorm2d(torch.autograd.Function):
    # def __init__(self, num_dims, num_features):
    #     super(MyBatchNorm2d, self).__init__()
    #     if num_dims == 2:
    #         shape = (1, num_features)
    #     else:
    #         shape = (1, num_features, 1, 1)
    #     # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成0和1
    #     # self.gamma = nn.Parameter(torch.ones(shape))
    #     # self.beta = nn.Parameter(torch.zeros(shape))
    #     self.gamma = None
    #     self.beta = None
    #     # 不参与求梯度和迭代的变量，全在内存上初始化成0
    #     self.moving_mean = torch.zeros(shape)
    #     self.moving_var = torch.zeros(shape)

    @staticmethod
    def forward(ctx, X, gamma,beta):
        mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        eps = 1e-5
        X_hat = (X - mean) / torch.sqrt(var + eps)
        Y = gamma * X_hat + beta  # self.gamma相当与MyBatchNorm2dal的weight,self.beta相当于MyBatchNorm2dal的bias
        ctx.save_for_backward(X_hat, gamma, X-mean,var+eps,torch.sqrt(var + eps))
        # print("Y.detach().numpy()", Y.detach().numpy())
        return Y

    @staticmethod
    def backward(ctx,grad_output):
        # np.hape(ivar)=(1,1,1,1) ,np.shape(gamma)=(1,), np.shape(grad_output)=(1,1,2,3)
        xhat, gamma, xmu, ivar, sqrtvar = ctx.saved_tensors
        N,C,H,W = grad_output.shape
        # print("N,C,H,W",N,C,H,W)
        # calculate gradients
        # dgamma = torch.ones((1,1,1,1),dtype=torch.float32)*200
        # dbeta= torch.ones((1,1,1,1),dtype=torch.float32)*200
        dgamma = torch.sum(xhat*grad_output , dim=[0,2,3], keepdim=True)
        dbeta = torch.sum(grad_output, dim=[0,2,3], keepdim=True)
        # (1,1,1,1)
        # print("np.shape(dgamma.detach().numpy())",np.shape(dgamma.detach().numpy()))
        # print("np.shape(grad_output.detach().numpy())", np.shape(grad_output.detach().numpy()))
        # np.shape(dx_)=(N,C,H,W)
        # dx_ = gamma * grad_output
        # dx = N * dx_ - torch.sum(dx_, dim=0, keepdim=True) - xhat * torch.sum(dx_ * xhat, dim=0, keepdim=True)
        # dx1 = dx * (torch.tensor([1,],dtype=torch.float32) /(N*sqrtvar))
        dy = grad_output * gamma
        dx3=(dy * xhat).sum(dim=[0,2,3],keepdim=True)
        dx2 = dy.sum(dim=[0,2,3],keepdim=True)
        dx = (dy - dx2/(N*H*W) - xhat * dx3/(N*H*W))/(sqrtvar)

        # print("np.shape(ivar.detach().numpy())", np.shape(ivar.detach().numpy()))
        # print("np.shape(dx.detach().numpy())", np.shape(dx.detach().numpy()))
        # print("np.shape(dbeta.detach().numpy())", np.shape(dbeta.detach().numpy()))
        # print("np.shape(xhat * torch.sum(dx_ * xhat, dim=0, keepdim=True).detach().numpy())", np.shape(xhat * torch.sum(dx_ * xhat, dim=0, keepdim=True).detach().numpy()))
        # print("dx1",dx1)
        return dx, dgamma, dbeta

    @staticmethod
    def symbolic(g: torch.Graph, input: torch.Value, gamma: torch.Value,beta: torch.Value) -> torch.Value:
        return g.op("MyBatchNormal2d", input, g.op("input",
                         value_t=torch.tensor([], dtype=torch.float32)),gamma,g.op("gamma",
                         value_t=torch.tensor([], dtype=torch.float32)),beta,g.op("beta",
                         value_t=torch.tensor([], dtype=torch.float32)))



class model1(nn.Module):
    def __init__(self):
        super(model1, self).__init__()
        self.conv2d = nn.Conv2d(5,6,(2,2),stride=(2,4),padding=(0,1))
        # self.MyConv2d =MyConv2d
        # self.conv2d.weight = nn.Parameter(weight)
        self.normal = nn.BatchNorm2d(6,momentum=1)
        # self.normal.running_var=torch.zeros((1,))
        # self.normal.running_mean=torch.zeros((1,))
        # print('running_mean:', self.normal.running_mean)  # 初始值
        # print('running_var:', self.normal.running_var)

    def forward(self,input):
        # print("model1 input.detach().numpy()", input.detach().numpy())
        normal_out=self.conv2d(input)
        # print("normal_out.shape",normal_out.shape)
        # print("model1 self.conv2d.weight.detach().numpy()", self.conv2d.weight.detach().numpy())
        # print("model1 normal_out.detach().numpy()", np.shape(normal_out.detach().numpy()))
        # print("self.normal.weight.detach().numpy()",self.normal.weight.detach().numpy())
        # print("self.normal.bias.detach().numpy()", self.normal.bias.detach().numpy())
        normal_out=self.normal(normal_out)
        return normal_out


class model2(nn.Module):
    def __init__(self,weight,channel,bias,stride,padding):
        super(model2, self).__init__()
        # self.conv2d = nn.Conv2d(3, channel, (2, 2),bias=False)
        # self.conv2d.weight=nn.Parameter(weight)
        # self.MyConv2d = MyConv2d.apply

        self.weight =  nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.MyConv2d = MyConv2d_package(1, 1, 1, self.weight, self.bias,stride,padding)
        # self.normal  = MyBatchNormal3(3,1)
        self.gamma= nn.Parameter(torch.ones((1,channel,1,1),dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros((1,channel,1,1),dtype=torch.float32))
        self.normal = MyBatchNorm2d.apply

    def forward(self, input):
        # print("model2 input.detach().numpy()", input.detach().numpy())
        # normal_out = self.MyConv2d(input,self.weight,self.bias)
        normal_out = self.MyConv2d(input)
        # print("model2 self.weight.detach().numpy()", self.weight.detach().numpy())
        # print("model2 out.detach().numpy()", np.shape(normal_out.detach().numpy()))
        # print("model2 self.gamma.detach().numpy()", np.shape(self.gamma.detach().numpy()))
        # print("model2 self.gamma.detach().numpy()", self.gamma.detach().numpy())
        # print("model2 self.beta.detach().numpy()", self.beta.detach().numpy())
        normal_out = self.normal(normal_out,self.gamma,self.beta)
        return normal_out

torch.manual_seed(100)
# random_weight = torch.rand([1,1,2,2],dtype=torch.float32)
# random_weight_clone = random_weight.clone()

# input = torch.rand([1, 1, 3, 4])
# weight=batch.conv2d.weight.clone()

batch=model1()
weight=batch.conv2d.weight.clone()
bias=batch.conv2d.bias.clone()
myBatch=model2(weight,6,bias,(2,4),(1,1,0,0))
lossfunc3d = torch.nn.MSELoss()
lossfunc2d = torch.nn.MSELoss()
update3d=optim.Adam(batch.parameters(), lr=0.01)
update2d=optim.Adam(myBatch.parameters(), lr=0.01)

import numpy as np
for i in range(100):
    input = torch.rand([4, 5, 5, 8])
    y_out=torch.rand([4, 6, 2, 3],dtype=torch.float32)
    # print(y_out.detach().numpy())
    # print("old weight",old_weight)
    batch_out = batch(input)
    # print("torch.Size(batch_out)",np.shape(batch_out.detach().numpy()))
    # print("torch.Size(myBatchOut)",np.shape(myBatchOut.detach().numpy()))
    # for var_name in batch.conv2d.state_dict():
    #     if "weight" ==var_name:
    #         print("batch",var_name, '\t', batch.conv2d.state_dict()[var_name])
    loss3d=lossfunc3d(batch_out,y_out)
    update3d.zero_grad()
    loss3d.backward()
    update3d.step()

    weight_3=batch.conv2d.state_dict()['weight']
    myBatchOut = myBatch(input)
    loss2d=lossfunc2d(myBatchOut,y_out)
    update2d.zero_grad()
    loss2d.backward()
    update2d.step()
    # weight_2 =myBatch.conv2d.state_dict()["weight"]
    # print("weight_2", weight_2)
    # print("weight_3", weight_3)
    # print("weight_2/(weight_3+1e-5)",weight_2/(weight_3+1e-5))
    # for var_name in myBatch.normal.state_dict():
    #     print("myBatch",var_name, '\t', myBatch.normal.state_dict()[var_name]) #self.gamma相当与MyBatchNorm2dal的weight,self.beta相当于MyBatchNorm2dal的bias
    # print("myBatch.normal.gamma.detach().numpy()", myBatch.normal.gamma.detach().numpy())
    # print("batch.conv2d.weight",batch.conv2d.weight)
    print("batch_out",batch_out)
    print("myBatchOut", myBatchOut)