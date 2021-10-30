# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import ntpath
import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block,
            layers,
            zero_init_residual=False,
            groups=1,
            widen=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
            normalize=False,
            output_dim=0,
            hidden_mlp=0,
            nmb_ptypes=0,
            nmb_local_ptypes=[0],
            npatch=[0],
            eval_mode=False,
            grid_per_side=[],
    ):
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.eval_mode = eval_mode
        self.padding = nn.ConstantPad2d(1, 0.0)

        self.inplanes = width_per_group * widen
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(
                    replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # change padding 3 -> 2 compared to original torchvision code because added a padding layer
        num_out_filters = width_per_group * widen
        self.conv1 = nn.Conv2d(
            3, num_out_filters, kernel_size=7, stride=2, padding=2, bias=False
        )
        self.bn1 = norm_layer(num_out_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, num_out_filters, layers[0])
        num_out_filters *= 2
        self.layer2 = self._make_layer(
            block, num_out_filters, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        num_out_filters *= 2
        self.layer3 = self._make_layer(
            block, num_out_filters, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        num_out_filters *= 2
        self.layer4 = self._make_layer(
            block, num_out_filters, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # normalize output features
        self.l2norm = normalize

        #############
        # Projection head
        ############
        # Global projection head
        if output_dim == 0:
            self.projection_head = None
        elif hidden_mlp == 0:
            self.projection_head = nn.Linear(
                num_out_filters * block.expansion, output_dim)
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(num_out_filters * block.expansion, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, output_dim),
            )
        # Local projection head
        l_pro_heads = []
        for i in range(len(nmb_local_ptypes)):
            if output_dim == 0:
                self.add_module("l_pro_head" + str(i), None)
            elif hidden_mlp == 0:
                self.add_module("l_pro_head" + str(i),
                                nn.Linear(num_out_filters * block.expansion, output_dim))
            else:
                self.add_module("l_pro_head" + str(i),
                                nn.Sequential(
                    nn.Linear(num_out_filters *
                              block.expansion, hidden_mlp),
                    nn.BatchNorm1d(hidden_mlp),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_mlp, output_dim),
                )
                )

        ###########
        # Construct prototype layers
        ##########
        # prototype layer
        self.ptypes = None
        if isinstance(nmb_ptypes, list):
            self.ptypes = MultiPrototypes(output_dim, nmb_ptypes)
        elif nmb_ptypes > 0:
            self.ptypes = nn.Linear(output_dim, nmb_ptypes, bias=False)

        # Local prototype layers
        for i in range(len(nmb_local_ptypes)):
            if isinstance(nmb_local_ptypes[i], list):
                self.add_module("local_ptypes" + str(i),
                                MultiPrototypes(output_dim, nmb_local_ptypes[i]))
            elif nmb_local_ptypes[i] > 0:
                self.add_module("local_ptypes" + str(i),
                                nn.Linear(output_dim, nmb_local_ptypes[i], bias=False))
            else:
                self.add_module("local_ptypes" + str(i), None)

        # Local2 Global prototype layer
        for i in range(len(nmb_local_ptypes)):
            if (getattr(self, "local_ptypes" + str(i)) is not None) and (self.ptypes is not None):
                self.add_module("l2g_ptypes" + str(i),
                                nn.Linear(nmb_local_ptypes[i], nmb_ptypes, bias=False))
        self.nmb_local_levels = len(nmb_local_ptypes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # Record the number of grids for local patches: Ex:[3,5]
        self.grid_per_side = grid_per_side

    def clean_grad(self):
        for name, p in self.named_parameters():
            if ("ptypes" in name):
                p.grad = None

    def ptypes_normalize(self):
        with torch.no_grad():
            if self.ptypes is not None:
                w = self.ptypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.ptypes.weight.copy_(w)

            for i in range(self.nmb_local_levels):
                local_ptypes = getattr(self, "local_ptypes" + str(i))
                if local_ptypes is not None:
                    loc_w = local_ptypes.weight.data.clone()
                    loc_w = nn.functional.normalize(loc_w, dim=1, p=2)
                    local_ptypes.weight.copy_(loc_w)

                l2g_ptypes = getattr(self, "l2g_ptypes" + str(i))
                if l2g_ptypes is not None:
                    lg2_w = l2g_ptypes.weight.data.clone()
                    lg2_w = nn.functional.normalize(lg2_w, dim=1, p=2)
                    l2g_ptypes.weight.copy_(lg2_w)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward_backbone(self, x):
        x = self.padding(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.eval_mode:
            return x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        if self.ptypes is not None:
            return x, self.ptypes(x)
        return x

    def local_forward_head(self, x_list):
        ret_z, ret_logit = [], []
        for i in range(self.nmb_local_levels):
            x = x_list[i]
            l_pro_head = getattr(self, "l_pro_head" + str(i))
            local_ptypes = getattr(self, "local_ptypes" + str(i))
            if l_pro_head is not None:
                x = l_pro_head(x)

            if self.l2norm:
                x = nn.functional.normalize(x, dim=1, p=2)

            ret_z.append(x)
            if local_ptypes is not None:
                ret_logit.append(local_ptypes(x))
        return ret_z, ret_logit

    def forward_l2g(self, loc_cat_logits, idx):
        l2g_ptypes = getattr(self, "l2g_ptypes" + str(idx))
        if self.l2norm:
            loc_cat_logits = nn.functional.normalize(
                loc_cat_logits, dim=1, p=2)
        return l2g_ptypes(loc_cat_logits)

    def forward(self, inputs):
        # Dataset returns list(Sequence), so the inputs produced by Dataloader are also lists
        # Ex:[Tensor of shape(Batch,3,224,224), (Batch,3,224,224), (Batch,3,96,96), .....]
        if not isinstance(inputs, list):
            inputs = [inputs]

        # [224,224,96,96,96,96,96,96,65,....,65] ->[2,6,18](nof of consecutive unique values) ->[2,8,26](cumulative sum)
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        bs = inputs[0].shape[0]  # batch size

        ############
        # Obtain f by Encoder
        ############
        for end_idx in idx_crops:
            # torch.cat( [(Batch,3,224,224), (Batch,3,224,224)] ) -> Tensor(Batch*2, 3, 224, 224)
            _out = self.forward_backbone(
                torch.cat(inputs[start_idx: end_idx]).cuda(non_blocking=True))
            # _out corresponds to f in the paper
            if start_idx == 0:
                f = _out
            else:
                f = torch.cat((f, _out))
            start_idx = end_idx
        # Partition apart the global and local f
        local_idx_offset = -(self.nmb_local_levels + 1)
        global_f = f[:bs * idx_crops[local_idx_offset]]
        # local_f is a list [local_f0, local_f1, ...]
        local_f = [f[bs * idx_crops[local_idx_offset + i]:bs * idx_crops[local_idx_offset + i + 1]] 
                                                        for i in range(self.nmb_local_levels)]

        # 3. Return (global_z,global_p), ([local_z0, local_z1, ...], [local_p0, local_p1, ...]) respectively
        return self.forward_head(global_f), self.local_forward_head(local_f)


class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_ptypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_ptypes)
        for i, k in enumerate(nmb_ptypes):
            self.add_module("ptypes" + str(i),
                            nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "ptypes" + str(i))(x))
        return out


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet50w2(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=2, **kwargs)


def resnet50w4(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=4, **kwargs)


def resnet50w5(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=5, **kwargs)
