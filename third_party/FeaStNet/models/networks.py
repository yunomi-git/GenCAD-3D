# Adapted from Verma et al, https://github.com/sw-gong/FeaStNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import FeaStConv
import numpy as np

class FeaStNet(torch.nn.Module):
    def __init__(self, args: dict): #in_channels, out_dim, heads, t_inv=True, last_layer=None):
        # args: in_channels, out_dim, heads, t_inv, last_layer
        super(FeaStNet, self).__init__()

        self.in_channels = args["in_channels"]
        self.num_outputs = args["num_outputs"]
        self.heads = args["heads"]
        self.t_inv = args["t_inv"]
        self.outputs_at = args["outputs_at"]

        self.conv_dims = args["conv_dims"] # default 16, 32, 64
        self.lin_dims = args["lin_dims"] #default 128 258

        self.lin_dims = args["lin_dims"] #default 128 258

        self.initialization = "normal"
        if "initialization" in args:
            self.initialization = args["initialization"]

        # self.with_bn = False
        # if "with_bn" in args:
        #     self.with_bn = args["with_bn"]

        self.conv_list = nn.ModuleList()
        self.linear_list = nn.ModuleList()

        self.fc0 = nn.Linear(self.in_channels, self.conv_dims[0])

        # Create feast convolution layers
        prev_output_channel_size = self.conv_dims[0]
        for output_channel_size in self.conv_dims[1:] + [self.lin_dims[0]]:
            self.conv_list.append(FeaStConv(prev_output_channel_size,
                                            output_channel_size,
                                            heads=self.heads, t_inv=self.t_inv))
            prev_output_channel_size = output_channel_size

        # Create final linear layers
        for output_channel_size in self.lin_dims[1:] + [self.num_outputs]:
            self.linear_list.append(nn.Linear(prev_output_channel_size, output_channel_size))
            prev_output_channel_size = output_channel_size

        self.reset_parameters()


    def reset_parameters(self):
        for conv in self.conv_list:
            conv.reset_parameters(self.initialization)

    def forward(self, verts, edges, mass=None):
        # Assume batch size 1

        x = verts # n x 3
        edge_index = edges # 2 x n

        if len(x.shape) == 3:
            x = x[0]
        if len(edge_index.shape) == 3:
            edge_index = edge_index[0]

        x = F.elu(self.fc0(x))
        for conv in self.conv_list:
            x = F.elu(conv(x, edge_index))

        for lin in self.linear_list[:-1]:
            x = F.elu(lin(x))
            x = F.dropout(x, training=self.training, p=0.2)

        x = self.linear_list[-1](x)

        # Gather based on output
        if self.outputs_at == 'vertices':
            x_out = x

        else: # self.outputs_at == 'global_mean':
            # Produce a single global mean ouput.
            if mass is not None:
                mass = mass[0]
                x_out = torch.sum(x * mass.unsqueeze(-1), dim=-2) / torch.sum(mass, dim=-1, keepdim=True)
                # clip outputs
                # x_out = torch.clip(x_out, min=-1E8, max=1E8)
                x_out = x_out.unsqueeze(0)
            else:
                x_out = torch.mean(x, dim=0, keepdim=True) # n x out_dim -> out_dim

        x_out = F.tanh(x_out)

        return x_out



class FeaStNetResidual(torch.nn.Module):
    def __init__(self, args: dict): #in_channels, out_dim, heads, t_inv=True, last_layer=None):
        # args: in_channels, out_dim, heads, t_inv, last_layer
        super(FeaStNetResidual, self).__init__()

        self.in_channels = args["in_channels"]
        self.num_outputs = args["num_outputs"]
        self.heads = args["heads"]
        self.t_inv = args["t_inv"]
        self.outputs_at = args["outputs_at"]

        self.conv_emb_dims = args["conv_emb_dims"]
        self.conv_dims = args["conv_dims"] # default 16, 32, 64
        self.lin_dims = args["lin_dims"] #default 128 258

        self.conv_list = nn.ModuleList()
        self.linear_list = nn.ModuleList()

        # self.fc0 = nn.Linear(self.in_channels, self.conv_dims[0])

        # Create feast convolution layers
        prev_output_channel_size = self.in_channels
        for output_channel_size in self.conv_dims[0:]:
            self.conv_list.append(FeaStConv(prev_output_channel_size,
                                            output_channel_size,
                                            heads=self.heads, t_inv=self.t_inv))
            prev_output_channel_size = output_channel_size

        # Create concatenated layer
        concatenated_size = sum(self.conv_dims)
        self.last_conv_layer = nn.Sequential(nn.Conv1d(concatenated_size, self.conv_emb_dims, kernel_size=1, bias=True),
                                             nn.LeakyReLU(negative_slope=0.2))
        prev_output_channel_size = self.conv_emb_dims * 2

        # Create final linear layers
        for output_channel_size in self.lin_dims[0:] + [self.num_outputs]:
            self.linear_list.append(nn.Linear(prev_output_channel_size, output_channel_size, bias=True))
            prev_output_channel_size = output_channel_size

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.conv_list:
            conv.reset_parameters()

    def forward(self, verts, edges, mass=None):
        # Assume batch size 1
        batch_size = 1

        x = verts # n x 3
        edge_index = edges # 2 x n

        if len(x.shape) == 3:
            x = x[0]
        if len(edge_index.shape) == 3:
            edge_index = edge_index[0]

        # Convolution layers
        conv_outputs = []
        for conv in self.conv_list:
            x = F.elu(conv(x, edge_index))
            conv_outputs.append(x)

        # Apply last convolution
        x = torch.cat(conv_outputs, dim=1)
        x = self.last_conv_layer(x.T)

        # Convert to linear inputs
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        # Linear layers
        for lin in self.linear_list[:-1]:
            x = F.elu(lin(x))
            x = F.dropout(x, training=self.training, p=0.2)

        # Last layer should not use dropout
        x = self.linear_list[-1](x)
        x_out = F.tanh(x)

        return x_out

