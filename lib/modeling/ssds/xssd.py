import torch
import torch.nn as nn
# from lib.layers import PriorBox


class XSSD(nn.Module):

    def __init__(self, base, head, feature_layer, num_classes):
        super(XSSD, self).__init__()
        self.num_classes = num_classes
        # SSD network
        self.base = nn.ModuleList(base)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.softmax = nn.Softmax(dim=-1)

        self.feature_layer = feature_layer[0]

    def forward(self,x,phase='eval'):
        sources, loc, conf = [list() for _ in range(3)]
        # apply bases layers and cache source layer outputs
        for k in range(len(self.base)):
            # print(k)
            x = self.base[k](x)
            if k in self.feature_layer:
                sources.append(x)

        if phase == 'feature':
            return sources

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if phase == 'eval':
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output


def add_extras(base, feature_layer, mbox, num_classes):
    extra_layers = []
    loc_layers = []
    conf_layers = []
    in_channels = None
    for layer, depth, box in zip(feature_layer[0], feature_layer[1], mbox):
        loc_layers += [nn.Conv2d(depth, box * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(depth, box * num_classes, kernel_size=3, padding=1)]
    return base, (loc_layers, conf_layers)

def build_xssd(base, feature_layer, mbox, num_classes):
    base_, head_ = add_extras(base(), feature_layer, mbox, num_classes)
    return XSSD(base_, head_, feature_layer, num_classes)
