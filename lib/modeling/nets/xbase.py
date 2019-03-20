import torch
import torch.nn as nn

base = {
    # 'x_pool': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            # 512, 512, 512],
    'x_pool': [64, 'M', 128, 'M', 256, 'C', 512, 'M', 256, 'C', 128, 'M',128, 'C', 64,'M']
}



def base_net(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers

def xbase():
    return base_net(base['x_pool'],3,True)
xbase.name = 'xbase'
# def vgg16():
#     return vgg(base['vgg16'], 3)
# vgg16.name='vgg16'




if __name__ == '__main__':

    # from lib.utils.summary import summary
    class BaseNet(nn.Module):

        def __init__(self,base):
            super(BaseNet, self).__init__()
            self.base = nn.ModuleList(base)
        
        def forward(self,x):
            for k in range(len(self.base)):
                x = self.base[k](x)
            
            return x
    basenet = base_net(base['x_pool'],3,True)
    model = BaseNet(basenet)
    print(model)
    # summary(model,(3,300,300))
