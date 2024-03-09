import torch
import torch.nn as nn
import torch.optim as optim
import inspect

class MyMaxPooling3d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, kernel, stride):
        h_w, w_w,d_w = kernel
        h_stride, w_stride,d_stride = stride
        batch, channel, h, w, d = X.shape
        batch_Y, channel_Y, h_Y, w_Y,d_Y = batch, channel, (h - h_w)//h_stride + 1, (w - w_w)//w_stride + 1, (d - d_w)//d_stride + 1
        list_ =[]
        for j in range(h_w):
            for i in range(w_w):
                for k in range(d_w):
                    list_.append(X[:, :, j:j+h_stride*h_Y:h_stride, i:i+w_stride*w_Y:w_stride, k:k+d_stride*d_Y:d_stride, None])
        x_cat = torch.cat(list_, dim=5)
        Y,index_ = torch.max(x_cat,dim=5)
        backTensor = torch.tensor([h_w, w_w, d_w, h_Y, w_Y,d_Y, batch, channel,h_stride, w_stride,d_stride])
        ctx.save_for_backward(torch.zeros_like(x_cat),torch.zeros_like(X),index_,backTensor)
        return Y

    @staticmethod
    def backward(ctx,grad_output):
        #h_w, w_w,h_Y,w_Y,batch, channel,h_stride, w_stride
        x_cat, X_zero, max_index, backTensor= ctx.saved_tensors
        # print("x_cat[max_index].shape",x_cat[max_index].shape)
        x_cat_=x_cat.view(backTensor[6]*backTensor[7]*backTensor[3]*backTensor[4]*backTensor[5],-1)
        max_index_=max_index.view(-1,)
        first_index = torch.tensor(range(backTensor[6]*backTensor[7]*backTensor[3]*backTensor[4]*backTensor[5]))
        x_cat_[first_index,max_index_] = grad_output.view(-1,) # 将导数赋值给最大值位置切片，其余为0
        x_cat_new=x_cat_.view(backTensor[6],backTensor[7], backTensor[3],backTensor[4],backTensor[5],-1)

        for j in range(backTensor[0]):
            for i in range(backTensor[1]):
                for k in range(backTensor[2]):
                    # 将各个切片加回到原来位置
                    X_zero[:,:,j:j+backTensor[8]*backTensor[3]:backTensor[8], i:i+backTensor[9]*backTensor[4]:backTensor[9], k:k+backTensor[10]*backTensor[5]:backTensor[10]] += x_cat_new[:,:,:,:,:,k+i*(backTensor[2])+(backTensor[2]*backTensor[1])*j]
        return X_zero, None,None

    @staticmethod
    def symbolic(g: torch.Graph, input: torch.Value, gamma: torch.Value,beta: torch.Value) -> torch.Value:
        return g.op("MyBatchNormal3d", input, g.op("input",
                         value_t=torch.tensor([], dtype=torch.float32)),gamma,g.op("gamma",
                         value_t=torch.tensor([], dtype=torch.float32)),beta,g.op("beta",
                         value_t=torch.tensor([], dtype=torch.float32)))

class MyMaxPooling3d_package(nn.Module):
    def __init__(self,kernel,stride):
        super(MyMaxPooling3d_package, self).__init__()
        self.MyMaxPooling3d=MyMaxPooling3d.apply
        self.kernel = kernel
        self.stride = stride

    def forward(self,input):
        normal_out = self.MyMaxPooling3d(input, self.kernel,self.stride)
        return normal_out

class model1(nn.Module):
    def __init__(self):
        super(model1, self).__init__()
        self.conv3d = nn.Conv3d(4, 5, (2, 2, 2))
        self.MyMaxPooling3d = MyMaxPooling3d_package((2, 2, 2),(2, 2, 2))


    def forward(self,input):
        normal_out=self.conv3d(input)
        normal_out=self.MyMaxPooling3d(normal_out)
        return normal_out

class model2(nn.Module):
    def __init__(self,weight,bias):
        super(model2, self).__init__()
        self.conv3d = nn.Conv3d(4, 5, (2, 2, 2))
        self.conv3d.weight = nn.Parameter(weight)
        self.conv3d.bias = nn.Parameter(bias)
        self.MaxPool3d = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))


    def forward(self,input):
        normal_out=self.conv3d(input)
        normal_out=self.MaxPool3d(normal_out)
        return normal_out

torch.manual_seed(100)
module_ =model1()
# out=module_(input)

lossfunc3d = torch.nn.MSELoss()
update3d=optim.Adam(module_.parameters(), lr=0.01)

module_maxpooling =model2(module_.conv3d.weight.clone(),module_.conv3d.bias.clone())
# out=module_(input)

lossfunc3d_max = torch.nn.MSELoss()
update3d_max=optim.Adam(module_maxpooling.parameters(), lr=0.01)

for i in range(100):
    input = torch.rand([3, 4, 7, 8, 6])
    y_out=torch.rand([3, 5, 3, 3, 2],dtype=torch.float32)
    batch_out = module_(input)
    loss3d=lossfunc3d(batch_out,y_out)
    update3d.zero_grad()
    loss3d.backward()
    update3d.step()

    max_out = module_maxpooling(input)
    print(max_out.shape)
    # print("torch.Size(batch_out)",np.shape(batch_out.detach().numpy()))
    # print("torch.Size(myBatchOut)",np.shape(myBatchOut.detach().numpy()))
    loss3d_max = lossfunc3d_max(max_out, y_out)
    update3d_max.zero_grad()
    loss3d_max.backward()
    update3d_max.step()
    print("batch_out", batch_out)
    print("max_out", max_out)