import torch
from torch import nn
import torchvision.models as models

class Backbone(nn.Module):
    def __init__(self, checkpoint):
        super(Backbone, self).__init__()
        # modify state_dict for model compatibility
        state_dict = torch.load(checkpoint)['state_dict']
        for k in list(state_dict.keys()):
            if not k.startswith('module.projection_head') and \
                not k.startswith('module.prototypes'):
                state_dict[k[len('module.'):]] = state_dict[k]
            del state_dict[k]

        # create pre-trained encoder
        network = models.resnet50()
        network.load_state_dict(state_dict, strict=False)
        self.encoder = torch.nn.Sequential(*list(network.children())[:-1])

    def forward(self, x):
        x = self.encoder(x)
        h = x.flatten(1)
        return h

class RegLog(nn.Module):
    def __init__(self, num_labels):
        super(RegLog, self).__init__()
        self.linear = nn.Linear(2048, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.fill_(0.1)

    def forward(self, x):
        return self.linear(x)
