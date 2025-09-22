#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adapted from Wang et al https://github.com/WangYueFt/dgcnn

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    else:
        k = idx.shape[2]
    device = x.get_device()

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

# The default model is
# conv_channel_sizes: [64, 64, 128, 256]
# emb_dims xxx
# linear_sizes: [512 256]
# num_outputs
# k: 20
# dropout
class DGCNN_param(nn.Module):
    def __init__(self, args: dict):
        super(DGCNN_param, self).__init__()
        self.conv_channel_sizes = args["conv_channel_sizes"]
        self.linear_sizes = args["linear_sizes"]

        self.input_dims = 2 * args["input_dims"]
        self.conv_emb_dims = args["emb_dims"]
        self.output_dim = args["num_outputs"]

        self.args = args
        self.k = args['k']

        # Add Convolutional Layers
        self.conv_list = nn.ModuleList()

        def create_conv_layer(input_channel, output_channel):
            conv = nn.Sequential(nn.Conv2d(input_channel, output_channel, kernel_size=1, bias=True),
                                 nn.BatchNorm2d(output_channel),
                                 nn.LeakyReLU(negative_slope=0.2))
            return conv

        prev_output_channel_size = self.input_dims
        for output_channel_size in self.conv_channel_sizes:
            conv = create_conv_layer(input_channel=prev_output_channel_size, output_channel=output_channel_size)
            self.conv_list.append(conv)
            prev_output_channel_size = output_channel_size * 2
        # Last layer: concatenate all prev inputs and convert to embedding size
        concatenated_size = sum(self.conv_channel_sizes)
        self.last_conv_layer = nn.Sequential(nn.Conv1d(concatenated_size, self.conv_emb_dims, kernel_size=1, bias=True),
                                 nn.BatchNorm1d(self.conv_emb_dims),
                                 nn.LeakyReLU(negative_slope=0.2))

        # Add linear layers
        self.linear_bn_list = nn.ModuleList()
        self.linear_list = nn.ModuleList()
        self.linear_dropout_list = nn.ModuleList()

        def add_linear_layer(input_size, output_size, bias=False):
            self.linear_list.append(nn.Linear(input_size, output_size, bias=bias))
            self.linear_bn_list.append(nn.BatchNorm1d(output_size))
            self.linear_dropout_list.append(nn.Dropout(p=args['dropout']))

        # First layer:
        add_linear_layer(self.conv_emb_dims * 2, self.linear_sizes[0], bias=False)
        prev_layer_size = self.linear_sizes[0]
        for output_layer_size in self.linear_sizes[1:]:
            add_linear_layer(input_size=prev_layer_size, output_size=output_layer_size, bias=True)
            prev_layer_size = output_layer_size
        # Last Layer
        self.last_linear_layer = nn.Linear(prev_layer_size, self.output_dim)

    def forward(self, x):
        # x input is as batch x points x features. Convert to batch x features x points
        x = torch.permute(x, (0, 2, 1))
        batch_size = x.size(0)

        # First do convolutions
        conv_outputs = []
        for conv_layer in self.conv_list:
            x = get_graph_feature(x, k=self.k)
            x = conv_layer(x)
            x = x.max(dim=-1, keepdim=False)[0]
            conv_outputs.append(x)

        # Apply last convolution
        x = torch.cat(conv_outputs, dim=1)
        x = self.last_conv_layer(x)

        # Convert to linear inputs
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        # Do linear layers
        for i in range(len(self.linear_list)):
            x = F.leaky_relu(self.linear_bn_list[i](self.linear_list[i](x)), negative_slope=0.2)
            x = self.linear_dropout_list[i](x)

        x = self.last_linear_layer(x)
        x = torch.tanh(x)

        return x
