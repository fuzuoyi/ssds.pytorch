import torch
import torch.nn as nn

from collections import namedtuple
import functools



BlockS = namedtuple('BlockS', ['depth','repeat'])
BlockD = namedtuple('BlockD', ['in_channels','out_channels'])

GP_1 = [
    # stage2
    BlockD(in_channels=24, out_channels=64),
    BlockS(depth=64,  repeat=2),
   
    # stage3
    BlockD(in_channels = 64, out_channels=96),
    BlockS(depth=96,  repeat=2),
    
    # stage4
    BlockD(in_channels = 96, out_channels=160),
    BlockS(depth=160,  repeat=3),

    # stage5
    BlockD(in_channels = 160, out_channels=320),
    BlockS(depth=320,  repeat=4),

    # stage6
    BlockD(in_channels = 320, out_channels=480),
    BlockS(depth=480,  repeat=3),
]

class Channel_shuffle(nn.Module):

    def __init__(self, groups):
        super(Channel_shuffle,self).__init__()
        self.groups = groups

    def forward(self,x):
        N, C, H, W = x.data.size()
        channels_per_group = C // self.groups
        x = x.view(N, self.groups, channels_per_group, H, W)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(N, C, H, W)
        # x = x.view(N, -1, H, W)
        return x

class _conv_1x1(nn.Module):
    def __init__(self, inp, oup):
        super(_conv_1x1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, 1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )
        self.depth = oup

    def forward(self, x):
        return self.conv(x)

class _conv_dw(nn.Module):
    def __init__(self, inp, oup, stride):
        super(_conv_dw, self).__init__()
        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            # pw
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )
        self.depth = oup

    def forward(self, x):
        return self.conv(x)


class XUnit_S(nn.Module):

    def __init__(self, depth):
        super(XUnit_S,self).__init__()

        mid = depth//2

        self.conv = nn.Sequential(
            _conv_1x1(depth,mid),
            _conv_dw(mid,mid,1),
            _conv_1x1(mid,depth),
            )
        # self.pool = nn.AvgPool2d(3,1,1)
    def forward(self, x):
        # return self.pool(x) + self.conv(x)
        return x + self.conv(x)


class XUnit_D(nn.Module):

    def __init__(self, inp, out):
        super(XUnit_D,self).__init__()

        depth = inp
        self.conv = nn.Sequential(
            _conv_1x1(depth, depth),
            _conv_dw(depth, out,stride=2),
            )
        self.conv_pool = nn.Sequential(
            _conv_dw(depth,out,1),
            nn.AvgPool2d(3,2,1)
            )
    def forward(self, x):
        return self.conv_pool(x) + self.conv(x)


def xxbase(conv_defs):
    
    layers = []
    in_channels = 3

    # stage 1
    layers += [nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1),nn.BatchNorm2d(24),nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

    for conv_def in conv_defs:
        if isinstance(conv_def, BlockD):
            layers += [XUnit_D(conv_def.in_channels, conv_def.out_channels)]
        elif isinstance(conv_def, BlockS):
            for i in range(conv_def.repeat):
                layers += [XUnit_S(conv_def.depth)]
          
    return layers

def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


xx_net = wrapped_partial(xxbase, conv_defs=GP_1)


if __name__ == '__main__':
    import os 
    import sys
    import time

    # print(os.getcwd())
    # sys.path.insert(0,'.')
    from lib.utils.summary import summary
    from lib.utils.compute_flops.utils import profile

    class MM(nn.Module):
        def __init__(self,base):
            super(MM,self).__init__()
            self.base = nn.ModuleList(base)
        def forward(self,x):
            for k in range(len(self.base)):
                x = self.base[k](x)
            
            return x



    model = MM(xx_net())
    # print(model)
    # summary(model,(3,300,300))
    total_ops, total_params = profile(model,(1,3,300,300))
    print('Gflops:',total_ops / (10**9))
    print('Total params(MB):',total_params / (1024**2 / 4))



    # data = torch.autograd.Variable(torch.rand(32,3,300,300), requires_grad = False)
    # # model = model.cuda()
    # # data = data.cuda()
    # model.eval()

    # statrt = time.time()

    # model(data)

    # dura = time.time() - statrt
    # print(dura)