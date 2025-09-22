import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import torch.nn as nn

def normal(tensor, mean, std):
    if tensor is not None:
        tensor.data.normal_(mean, std)

def xavier(tensor, gain):
    if tensor is not None:
        torch.nn.init.xavier_uniform_(tensor, gain=gain)
        # tensor.data.xavier_uniform_(gain=gain)

class FeaStConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=8,
                 bias=True,
                 t_inv=True,
                 **kwargs):
        super(FeaStConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.t_inv = t_inv

        self.weight = Parameter(torch.Tensor(in_channels,
                                             heads * out_channels))
        self.u = Parameter(torch.Tensor(in_channels, heads))
        self.c = Parameter(torch.Tensor(heads))
        if not self.t_inv:
            self.v = Parameter(torch.Tensor(in_channels, heads))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self, type="normal"):
        if type == "xavier":
            print("initialize with xavier!")
            xavier(self.weight, gain=1.0)
            xavier(self.u, gain=1.0)
            normal(self.c, mean=0, std=0.1)
            normal(self.bias, mean=0, std=0.1)
            if not self.t_inv:
                xavier(self.v, gain=1.0)
        else:
            normal(self.weight, mean=0, std=0.1)
            normal(self.u, mean=0, std=0.1)
            normal(self.c, mean=0, std=0.1)
            normal(self.bias, mean=0, std=0.1)
            if not self.t_inv:
                normal(self.v, mean=0, std=0.1)

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate(edge_index, x=x, num_nodes=x.size(0))

    def message(self, x_i, x_j):
        # dim: x_i, [E, F_in];
        if self.t_inv:
            # with translation invariance
            q = torch.mm((x_i - x_j), self.u) + self.c  #[E, heads]
        else:
            q = torch.mm(x_i, self.u) + torch.mm(x_j, self.v) + self.c
        q = F.softmax(q, dim=1)  #[E, heads]

        x_j = torch.mm(x_j, self.weight).view(-1, self.heads,
                                              self.out_channels)
        return (x_j * q.view(-1, self.heads, 1)).sum(dim=1)

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
