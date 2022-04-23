# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from base.layers import Conv2dWithConstraint, LinearWithConstraint

class classifier(nn.Module):
    def __init__(self, num_classes=4):
        super(classifier, self).__init__()
        self.fc = nn.Sequential(nn.Flatten(), 
                                nn.Linear(1080, 1080),
                                nn.ReLU(),
                                # nn.Dropout(0.3),
                                nn.Linear(1080, num_classes))

    
    def forward(self, x):
        return self.fc(x)


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder

        # build a 3-layer projector
        # prev_dim = self.encoder.fc.weight.shape[1]
        self.fc = nn.Sequential(nn.Linear(27, 27),
                                        nn.BatchNorm1d(40),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(27, 27),
                                        nn.BatchNorm1d(40),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(27, 27),
                                        nn.BatchNorm1d(40)) # output layer

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim),
                                        nn.BatchNorm1d(40),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1, x2, mode='train'):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view

        # print(x1.shape, x2.shape)
        e1 = self.encoder(x1)
        e2 = self.encoder(x2)

        z1 = self.fc(e1) # NxC
        z2 = self.fc(e2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC
        if mode == 'downstream':
            return z1
        else:
            return p1, p2, z1.detach(), z2.detach()

