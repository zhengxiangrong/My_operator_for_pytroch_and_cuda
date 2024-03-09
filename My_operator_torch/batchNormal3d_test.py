import torch
import torch.nn as nn
import torch.optim as optim
import inspect



class MyBatchNorm3d(torch.autograd.Function):
    # def __init__(self, num_dims, num_features):
    #     super(MyBatchNorm3d, self).__init__()
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
        mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        eps = 1e-5
        X_hat = (X - mean) / torch.sqrt(var + eps)
        Y = gamma * X_hat + beta  # self.gamma相当与MyBatchNorm3dal的weight,self.beta相当于MyBatchNorm3dal的bias
        ctx.save_for_backward(X_hat, gamma, X-mean,var+eps,torch.sqrt(var + eps))
        # print("Y.detach().numpy()", Y.detach().numpy())
        return Y

    @staticmethod
    def backward(ctx,grad_output):
        # np.hape(ivar)=(1,1,1,1) ,np.shape(gamma)=(1,), np.shape(grad_output)=(1,1,2,3)
        xhat, gamma, xmu, ivar, sqrtvar = ctx.saved_tensors
        N,C,H,W,D = grad_output.shape
        print("N,C,H,W",N,C,H,W)
        # calculate gradients
        # dgamma = torch.ones((1,1,1,1),dtype=torch.float32)*200
        # dbeta= torch.ones((1,1,1,1),dtype=torch.float32)*200
        dgamma = torch.sum(xhat*grad_output , dim=[0,2,3,4], keepdim=True)
        dbeta = torch.sum(grad_output, dim=[0,2,3,4], keepdim=True)
        # (1,1,1,1)
        # print("np.shape(dgamma.detach().numpy())",np.shape(dgamma.detach().numpy()))
        # print("np.shape(grad_output.detach().numpy())", np.shape(grad_output.detach().numpy()))
        # np.shape(dx_)=(N,C,H,W)
        # dx_ = gamma * grad_output
        # dx = N * dx_ - torch.sum(dx_, dim=0, keepdim=True) - xhat * torch.sum(dx_ * xhat, dim=0, keepdim=True)
        # dx1 = dx * (torch.tensor([1,],dtype=torch.float32) /(N*sqrtvar))
        dy = grad_output * gamma
        dx3=(dy * xhat).sum(dim=[0,2,3,4],keepdim=True)
        dx2 = dy.sum(dim=[0,2,3,4],keepdim=True)
        dx = (dy - dx2/(N*H*W*D) - xhat * dx3/(N*H*W*D))/(sqrtvar)

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
        self.conv3d = nn.Conv3d(3,4,(2,2,2),bias=False)
        self.normal = nn.BatchNorm3d(4,momentum=1,track_running_stats=False,affine=True)
        self.normal.running_var=torch.zeros((1,))
        self.normal.running_mean=torch.zeros((1,))
        # print('running_mean:', self.normal.running_mean)  # 初始值
        # print('running_var:', self.normal.running_var)

    def forward(self,input):
        # print("model1 input.detach().numpy()", input.detach().numpy())
        normal_out=self.conv3d(input)
        # print("model1 self.conv3d.weight.detach().numpy()", self.conv3d.weight.detach().numpy())
        # print("model1 out.detach().numpy()", out.detach().numpy())
        # print("self.normal.weight.detach().numpy()",self.normal.weight.detach().numpy())
        # print("self.normal.bias.detach().numpy()", self.normal.bias.detach().numpy())
        normal_out=self.normal(normal_out)
        return normal_out


class model2(nn.Module):
    def __init__(self,weight,channel):
        super(model2, self).__init__()
        self.conv3d = nn.Conv3d(3, channel, (2,2,2),bias=False)
        self.conv3d.weight=nn.Parameter(weight)
        # self.normal  = MyBatchNormal3(3,1)
        self.gamma= nn.Parameter(torch.ones((1,channel,1,1,1),dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros((1,channel,1,1,1),dtype=torch.float32))
        self.normal = MyBatchNorm3d.apply

    def forward(self, input):
        # print("model2 input.detach().numpy()", input.detach().numpy())
        normal_out = self.conv3d(input)
        # print("model2 self.conv3d.weight.detach().numpy()", self.conv3d.weight.detach().numpy())
        # print("model2 out.detach().numpy()", out.detach().numpy())
        # print("model2 self.gamma.detach().numpy()", self.gamma.detach().numpy())
        # print("model2 self.beta.detach().numpy()", self.beta.detach().numpy())
        normal_out = self.normal(normal_out,self.gamma,self.beta)
        return normal_out

torch.manual_seed(100)
# random_weight = torch.rand([1,1,2,2,2],dtype=torch.float32)
# random_weight_clone = random_weight.clone()
batch=model1()
weight_clone = batch.conv3d.weight.clone()
# input = torch.rand([1, 1, 3, 4])
# weight=batch.conv2d.weight.clone()
myBatch=model2(weight_clone,4)

lossfunc3d = torch.nn.MSELoss()
lossfunc2d = torch.nn.MSELoss()
update3d=optim.Adam(batch.parameters(), lr=0.01)
update2d=optim.Adam(myBatch.parameters(), lr=0.01)

import numpy as np
for i in range(100):
    input = torch.rand([3, 3, 3, 4,5])
    y_out=torch.rand([3,4,2,3,4],dtype=torch.float32)
    # print(y_out.detach().numpy())
    # print("old weight",old_weight)
    batch_out = batch(input)
    # print("torch.Size(batch_out)",np.shape(batch_out.detach().numpy()))
    # print("torch.Size(myBatchOut)",np.shape(myBatchOut.detach().numpy()))
    loss3d=lossfunc3d(batch_out,y_out)
    update3d.zero_grad()
    loss3d.backward()
    update3d.step()
    # for var_name in batch.normal.state_dict():
    #     if "weight" ==var_name or "bias" ==var_name:
    #         print("batch",var_name, '\t', batch.normal.state_dict()[var_name])
    weight_3=batch.conv3d.state_dict()['weight']
    myBatchOut = myBatch(input)
    loss2d=lossfunc2d(myBatchOut,y_out)
    update2d.zero_grad()
    loss2d.backward()
    update2d.step()
    weight_2 =myBatch.conv3d.state_dict()["weight"]
    # print("weight_2", weight_2)
    # print("weight_3", weight_3)
    # print("weight_2/(weight_3+1e-5)",weight_2/(weight_3+1e-5))
    # for var_name in myBatch.normal.state_dict():
    #     print("myBatch",var_name, '\t', myBatch.normal.state_dict()[var_name]) #self.gamma相当与MyBatchNorm3dal的weight,self.beta相当于MyBatchNorm3dal的bias
    # print("myBatch.normal.gamma.detach().numpy()", myBatch.normal.gamma.detach().numpy())
    # print("batch.conv2d.weight",batch.conv2d.weight)
    print("batch_out",batch_out)
    print("myBatchOut", myBatchOut)