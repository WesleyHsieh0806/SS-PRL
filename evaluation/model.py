import torch
from torch import nn
import torchvision.models as models

class Backbone(nn.Module):
    def __init__(self, name, checkpoint):
        super(Backbone, self).__init__()
        state_dict = self._get_state_dict(name, checkpoint)
        self.encoder = self._get_pretrained_encoder(name, state_dict)

    def _get_state_dict(self, name, checkpoint):
        assert name in ['byol', 'swav', 'densecl', 'scrl', 'moco', 'simclr', 'detco']

        if name in ['byol', 'scrl']:
            state_dict = checkpoint['online_network_state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('encoder'):
                    state_dict[k[len('encoder.'):]] = state_dict[k]
                del state_dict[k]
        elif name in ['densecl', 'simclr', 'detco']:
            state_dict = checkpoint['state_dict']
        elif name == 'swav':
            state_dict = checkpoint
            for k in list(state_dict.keys()):
                if not k.startswith('module.projection_head') and \
                   not k.startswith('module.prototypes'):
                    state_dict[k[len('module.'):]] = state_dict[k]
                del state_dict[k]
        elif name == 'moco':
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    state_dict[k[len('module.encoder_q.'):]] = state_dict[k]
                del state_dict[k]
        
        return state_dict

    def _get_pretrained_encoder(self, name, state_dict):
        network = models.resnet50()

        if name in ['byol', 'scrl']:
            encoder = torch.nn.Sequential(*list(network.children())[:-1])
            encoder.load_state_dict(state_dict, strict=False)
        elif name in ['densecl', 'swav', 'moco', 'simclr', 'detco']:
            network.load_state_dict(state_dict, strict=False)
            encoder = torch.nn.Sequential(*list(network.children())[:-1])

        return encoder

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
