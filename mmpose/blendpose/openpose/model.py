# Copyright (c) wilson.xu. All rights reserved.
from collections import OrderedDict

import torch
import torch.nn as nn


def make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(
                in_channels=v[0],
                out_channels=v[1],
                kernel_size=v[2],
                stride=v[3],
                padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))

    return nn.Sequential(OrderedDict(layers))


class BodyPoseModel(nn.Module):

    def __init__(self):
        super(BodyPoseModel, self).__init__()

        # these layers have no relu layer
        no_relu_layers = ['conv5_5_CPM_L1', 'conv5_5_CPM_L2', 'Mconv7_stage2_L1',\
                          'Mconv7_stage2_L2', 'Mconv7_stage3_L1', 'Mconv7_stage3_L2',\
                          'Mconv7_stage4_L1', 'Mconv7_stage4_L2', 'Mconv7_stage5_L1',\
                          'Mconv7_stage5_L2', 'Mconv7_stage6_L1', 'Mconv7_stage6_L1']
        blocks = {}
        block0 = OrderedDict([('conv1_1', [3, 64, 3, 1, 1]),
                              ('conv1_2', [64, 64, 3, 1, 1]),
                              ('pool1_stage1', [2, 2, 0]),
                              ('conv2_1', [64, 128, 3, 1, 1]),
                              ('conv2_2', [128, 128, 3, 1, 1]),
                              ('pool2_stage1', [2, 2, 0]),
                              ('conv3_1', [128, 256, 3, 1, 1]),
                              ('conv3_2', [256, 256, 3, 1, 1]),
                              ('conv3_3', [256, 256, 3, 1, 1]),
                              ('conv3_4', [256, 256, 3, 1, 1]),
                              ('pool3_stage1', [2, 2, 0]),
                              ('conv4_1', [256, 512, 3, 1, 1]),
                              ('conv4_2', [512, 512, 3, 1, 1]),
                              ('conv4_3_CPM', [512, 256, 3, 1, 1]),
                              ('conv4_4_CPM', [256, 128, 3, 1, 1])])

        # Stage 1
        block1_1 = OrderedDict([('conv5_1_CPM_L1', [128, 128, 3, 1, 1]),
                                ('conv5_2_CPM_L1', [128, 128, 3, 1, 1]),
                                ('conv5_3_CPM_L1', [128, 128, 3, 1, 1]),
                                ('conv5_4_CPM_L1', [128, 512, 1, 1, 0]),
                                ('conv5_5_CPM_L1', [512, 38, 1, 1, 0])])

        block1_2 = OrderedDict([('conv5_1_CPM_L2', [128, 128, 3, 1, 1]),
                                ('conv5_2_CPM_L2', [128, 128, 3, 1, 1]),
                                ('conv5_3_CPM_L2', [128, 128, 3, 1, 1]),
                                ('conv5_4_CPM_L2', [128, 512, 1, 1, 0]),
                                ('conv5_5_CPM_L2', [512, 19, 1, 1, 0])])
        blocks['block1_1'] = block1_1
        blocks['block1_2'] = block1_2

        self.model0 = make_layers(block0, no_relu_layers)

        # Stages 2 - 6
        for i in range(2, 7):
            blocks['block%d_1' % i] = OrderedDict([
                ('Mconv1_stage%d_L1' % i, [185, 128, 7, 1, 3]),
                ('Mconv2_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                ('Mconv3_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                ('Mconv4_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                ('Mconv5_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                ('Mconv6_stage%d_L1' % i, [128, 128, 1, 1, 0]),
                ('Mconv7_stage%d_L1' % i, [128, 38, 1, 1, 0])
            ])

            blocks['block%d_2' % i] = OrderedDict([
                ('Mconv1_stage%d_L2' % i, [185, 128, 7, 1, 3]),
                ('Mconv2_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                ('Mconv3_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                ('Mconv4_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                ('Mconv5_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                ('Mconv6_stage%d_L2' % i, [128, 128, 1, 1, 0]),
                ('Mconv7_stage%d_L2' % i, [128, 19, 1, 1, 0])
            ])

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_1 = blocks['block1_1']
        self.model2_1 = blocks['block2_1']
        self.model3_1 = blocks['block3_1']
        self.model4_1 = blocks['block4_1']
        self.model5_1 = blocks['block5_1']
        self.model6_1 = blocks['block6_1']

        self.model1_2 = blocks['block1_2']
        self.model2_2 = blocks['block2_2']
        self.model3_2 = blocks['block3_2']
        self.model4_2 = blocks['block4_2']
        self.model5_2 = blocks['block5_2']
        self.model6_2 = blocks['block6_2']

    def forward(self, x):

        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)

        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        return out6_1, out6_2


class HandPoseModel(nn.Module):

    def __init__(self):
        super(HandPoseModel, self).__init__()

        # these layers have no relu layer
        no_relu_layers = ['conv6_2_CPM', 'Mconv7_stage2', 'Mconv7_stage3',\
                          'Mconv7_stage4', 'Mconv7_stage5', 'Mconv7_stage6']
        # stage 1
        block1_0 = OrderedDict([('conv1_1', [3, 64, 3, 1, 1]),
                                ('conv1_2', [64, 64, 3, 1, 1]),
                                ('pool1_stage1', [2, 2, 0]),
                                ('conv2_1', [64, 128, 3, 1, 1]),
                                ('conv2_2', [128, 128, 3, 1, 1]),
                                ('pool2_stage1', [2, 2, 0]),
                                ('conv3_1', [128, 256, 3, 1, 1]),
                                ('conv3_2', [256, 256, 3, 1, 1]),
                                ('conv3_3', [256, 256, 3, 1, 1]),
                                ('conv3_4', [256, 256, 3, 1, 1]),
                                ('pool3_stage1', [2, 2, 0]),
                                ('conv4_1', [256, 512, 3, 1, 1]),
                                ('conv4_2', [512, 512, 3, 1, 1]),
                                ('conv4_3', [512, 512, 3, 1, 1]),
                                ('conv4_4', [512, 512, 3, 1, 1]),
                                ('conv5_1', [512, 512, 3, 1, 1]),
                                ('conv5_2', [512, 512, 3, 1, 1]),
                                ('conv5_3_CPM', [512, 128, 3, 1, 1])])

        block1_1 = OrderedDict([('conv6_1_CPM', [128, 512, 1, 1, 0]),
                                ('conv6_2_CPM', [512, 22, 1, 1, 0])])

        blocks = {}
        blocks['block1_0'] = block1_0
        blocks['block1_1'] = block1_1

        # stage 2-6
        for i in range(2, 7):
            blocks['block%d' % i] = OrderedDict([
                ('Mconv1_stage%d' % i, [150, 128, 7, 1, 3]),
                ('Mconv2_stage%d' % i, [128, 128, 7, 1, 3]),
                ('Mconv3_stage%d' % i, [128, 128, 7, 1, 3]),
                ('Mconv4_stage%d' % i, [128, 128, 7, 1, 3]),
                ('Mconv5_stage%d' % i, [128, 128, 7, 1, 3]),
                ('Mconv6_stage%d' % i, [128, 128, 1, 1, 0]),
                ('Mconv7_stage%d' % i, [128, 22, 1, 1, 0])
            ])

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_0 = blocks['block1_0']
        self.model1_1 = blocks['block1_1']
        self.model2 = blocks['block2']
        self.model3 = blocks['block3']
        self.model4 = blocks['block4']
        self.model5 = blocks['block5']
        self.model6 = blocks['block6']

    def forward(self, x):
        out1_0 = self.model1_0(x)
        out1_1 = self.model1_1(out1_0)
        concat_stage2 = torch.cat([out1_1, out1_0], 1)
        out_stage2 = self.model2(concat_stage2)
        concat_stage3 = torch.cat([out_stage2, out1_0], 1)
        out_stage3 = self.model3(concat_stage3)
        concat_stage4 = torch.cat([out_stage3, out1_0], 1)
        out_stage4 = self.model4(concat_stage4)
        concat_stage5 = torch.cat([out_stage4, out1_0], 1)
        out_stage5 = self.model5(concat_stage5)
        concat_stage6 = torch.cat([out_stage5, out1_0], 1)
        out_stage6 = self.model6(concat_stage6)
        return out_stage6


class FaceNet(nn.Module):
    """Model the cascading heatmaps."""

    def __init__(self):
        super(FaceNet, self).__init__()
        # cnn to make feature map
        self.relu = nn.ReLU()
        self.max_pooling_2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1_1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv2_1 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv2_2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv3_1 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv3_2 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv3_3 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv3_4 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv4_1 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv4_2 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv4_3 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv4_4 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv5_1 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv5_2 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv5_3_CPM = nn.Conv2d(
            in_channels=512,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1)

        # stage1
        self.conv6_1_CPM = nn.Conv2d(
            in_channels=128,
            out_channels=512,
            kernel_size=1,
            stride=1,
            padding=0)
        self.conv6_2_CPM = nn.Conv2d(
            in_channels=512,
            out_channels=71,
            kernel_size=1,
            stride=1,
            padding=0)

        # stage2
        self.Mconv1_stage2 = nn.Conv2d(
            in_channels=199,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3)
        self.Mconv2_stage2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3)
        self.Mconv3_stage2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3)
        self.Mconv4_stage2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3)
        self.Mconv5_stage2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3)
        self.Mconv6_stage2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0)
        self.Mconv7_stage2 = nn.Conv2d(
            in_channels=128,
            out_channels=71,
            kernel_size=1,
            stride=1,
            padding=0)

        # stage3
        self.Mconv1_stage3 = nn.Conv2d(
            in_channels=199,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3)
        self.Mconv2_stage3 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3)
        self.Mconv3_stage3 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3)
        self.Mconv4_stage3 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3)
        self.Mconv5_stage3 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3)
        self.Mconv6_stage3 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0)
        self.Mconv7_stage3 = nn.Conv2d(
            in_channels=128,
            out_channels=71,
            kernel_size=1,
            stride=1,
            padding=0)

        # stage4
        self.Mconv1_stage4 = nn.Conv2d(
            in_channels=199,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3)
        self.Mconv2_stage4 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3)
        self.Mconv3_stage4 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3)
        self.Mconv4_stage4 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3)
        self.Mconv5_stage4 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3)
        self.Mconv6_stage4 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0)
        self.Mconv7_stage4 = nn.Conv2d(
            in_channels=128,
            out_channels=71,
            kernel_size=1,
            stride=1,
            padding=0)

        # stage5
        self.Mconv1_stage5 = nn.Conv2d(
            in_channels=199,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3)
        self.Mconv2_stage5 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3)
        self.Mconv3_stage5 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3)
        self.Mconv4_stage5 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3)
        self.Mconv5_stage5 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3)
        self.Mconv6_stage5 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0)
        self.Mconv7_stage5 = nn.Conv2d(
            in_channels=128,
            out_channels=71,
            kernel_size=1,
            stride=1,
            padding=0)

        # stage6
        self.Mconv1_stage6 = nn.Conv2d(
            in_channels=199,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3)
        self.Mconv2_stage6 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3)
        self.Mconv3_stage6 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3)
        self.Mconv4_stage6 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3)
        self.Mconv5_stage6 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3)
        self.Mconv6_stage6 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0)
        self.Mconv7_stage6 = nn.Conv2d(
            in_channels=128,
            out_channels=71,
            kernel_size=1,
            stride=1,
            padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Return a list of heatmaps."""
        heatmaps = []

        h = self.relu(self.conv1_1(x))
        h = self.relu(self.conv1_2(h))
        h = self.max_pooling_2d(h)
        h = self.relu(self.conv2_1(h))
        h = self.relu(self.conv2_2(h))
        h = self.max_pooling_2d(h)
        h = self.relu(self.conv3_1(h))
        h = self.relu(self.conv3_2(h))
        h = self.relu(self.conv3_3(h))
        h = self.relu(self.conv3_4(h))
        h = self.max_pooling_2d(h)
        h = self.relu(self.conv4_1(h))
        h = self.relu(self.conv4_2(h))
        h = self.relu(self.conv4_3(h))
        h = self.relu(self.conv4_4(h))
        h = self.relu(self.conv5_1(h))
        h = self.relu(self.conv5_2(h))
        h = self.relu(self.conv5_3_CPM(h))
        feature_map = h

        # stage1
        h = self.relu(self.conv6_1_CPM(h))
        h = self.conv6_2_CPM(h)
        heatmaps.append(h)

        # stage2
        h = torch.cat([h, feature_map], dim=1)  # channel concat
        h = self.relu(self.Mconv1_stage2(h))
        h = self.relu(self.Mconv2_stage2(h))
        h = self.relu(self.Mconv3_stage2(h))
        h = self.relu(self.Mconv4_stage2(h))
        h = self.relu(self.Mconv5_stage2(h))
        h = self.relu(self.Mconv6_stage2(h))
        h = self.Mconv7_stage2(h)
        heatmaps.append(h)

        # stage3
        h = torch.cat([h, feature_map], dim=1)  # channel concat
        h = self.relu(self.Mconv1_stage3(h))
        h = self.relu(self.Mconv2_stage3(h))
        h = self.relu(self.Mconv3_stage3(h))
        h = self.relu(self.Mconv4_stage3(h))
        h = self.relu(self.Mconv5_stage3(h))
        h = self.relu(self.Mconv6_stage3(h))
        h = self.Mconv7_stage3(h)
        heatmaps.append(h)

        # stage4
        h = torch.cat([h, feature_map], dim=1)  # channel concat
        h = self.relu(self.Mconv1_stage4(h))
        h = self.relu(self.Mconv2_stage4(h))
        h = self.relu(self.Mconv3_stage4(h))
        h = self.relu(self.Mconv4_stage4(h))
        h = self.relu(self.Mconv5_stage4(h))
        h = self.relu(self.Mconv6_stage4(h))
        h = self.Mconv7_stage4(h)
        heatmaps.append(h)

        # stage5
        h = torch.cat([h, feature_map], dim=1)  # channel concat
        h = self.relu(self.Mconv1_stage5(h))
        h = self.relu(self.Mconv2_stage5(h))
        h = self.relu(self.Mconv3_stage5(h))
        h = self.relu(self.Mconv4_stage5(h))
        h = self.relu(self.Mconv5_stage5(h))
        h = self.relu(self.Mconv6_stage5(h))
        h = self.Mconv7_stage5(h)
        heatmaps.append(h)

        # stage6
        h = torch.cat([h, feature_map], dim=1)  # channel concat
        h = self.relu(self.Mconv1_stage6(h))
        h = self.relu(self.Mconv2_stage6(h))
        h = self.relu(self.Mconv3_stage6(h))
        h = self.relu(self.Mconv4_stage6(h))
        h = self.relu(self.Mconv5_stage6(h))
        h = self.relu(self.Mconv6_stage6(h))
        h = self.Mconv7_stage6(h)
        heatmaps.append(h)

        return heatmaps
