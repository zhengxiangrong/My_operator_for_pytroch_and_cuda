import torch
import torch.nn as nn
import torch.optim as optim
import inspect

class MyBatchNorm2d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, gamma,beta):
        mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        eps = 1e-5
        X_hat = (X - mean) / torch.sqrt(var + eps)
        Y = gamma * X_hat + beta  # self.gamma相当与MyBatchNorm2dal的weight,self.beta相当于MyBatchNorm2dal的bias
        ctx.save_for_backward(X_hat, gamma, X-mean,var+eps,torch.sqrt(var + eps))
        return Y

    @staticmethod
    def backward(ctx,grad_output):
        # np.hape(ivar)=(1,1,1,1) ,np.shape(gamma)=(1,), np.shape(grad_output)=(1,1,2,3)
        xhat, gamma, xmu, ivar, sqrtvar = ctx.saved_tensors
        N,C,H,W = grad_output.shape
        dgamma = torch.sum(xhat*grad_output , dim=[0,2,3], keepdim=True)
        dbeta = torch.sum(grad_output, dim=[0,2,3], keepdim=True)
        dy = grad_output * gamma
        dx3=(dy * xhat).sum(dim=[0,2,3],keepdim=True)
        dx2 = dy.sum(dim=[0,2,3],keepdim=True)
        dx = (dy - dx2/(N*H*W) - xhat * dx3/(N*H*W))/(sqrtvar)
        return dx, dgamma, dbeta

    @staticmethod
    def symbolic(g: torch.Graph, input: torch.Value, gamma: torch.Value,beta: torch.Value) -> torch.Value:
        return g.op("MyBatchNormal2d", input, g.op("input",
                         value_t=torch.tensor([], dtype=torch.float32)),gamma,g.op("gamma",
                         value_t=torch.tensor([], dtype=torch.float32)),beta,g.op("beta",
                         value_t=torch.tensor([], dtype=torch.float32)))

class MyBatchNorm2d_package(nn.Module):
    def __init__(self,in_channel):
        super(MyBatchNorm2d_package, self).__init__()
        self.MyBatchNorm2d=MyBatchNorm2d.apply
        self.gamma = nn.Parameter(torch.ones((1, in_channel, 1, 1), dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros((1, in_channel, 1, 1), dtype=torch.float32))


    def forward(self,input):
        normal_out = self.MyBatchNorm2d(input, self.gamma, self.beta)
        return normal_out