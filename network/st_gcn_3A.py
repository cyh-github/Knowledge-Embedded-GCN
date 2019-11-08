import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pickle as pk

from network.utils.tgcn_correlation import ConvTemporalGraphical as ConvTemporalGraphical_self
from network.utils.tgcn import ConvTemporalGraphical
from network.utils.graph import Graph
from network.utils.graph_knowledge import k_Graph

class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, t_length, temporal_kernel, channel_base, graph_args, graph_args_k, edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        
        #load knowledge graph
        self.k_graph = k_Graph(**graph_args_k)
        K_A = torch.tensor(self.k_graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('K_A', K_A)


        T_length = t_length
        channel1 = channel_base
        channel2 = channel1*2
        channel3 = channel1*4

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = temporal_kernel
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, channel1, kernel_size, T_length, 1, residual=False, **kwargs0),
            st_gcn(channel1, channel1, kernel_size, T_length, 1, **kwargs),
            st_gcn(channel1, channel1, kernel_size, T_length, 1, **kwargs),
            st_gcn(channel1, channel1, kernel_size, T_length, 1, **kwargs),
            st_gcn(channel1, channel2, kernel_size, T_length, 2, **kwargs),
            st_gcn(channel2, channel2, kernel_size, int(T_length/2), 1, **kwargs),
            st_gcn(channel2, channel2, kernel_size, int(T_length/2), 1, **kwargs),
            st_gcn(channel2, channel3, kernel_size, int(T_length/2), 2, **kwargs),
            st_gcn(channel3, channel3, kernel_size, int(T_length/4), 1, **kwargs),
            st_gcn(channel3, channel3, kernel_size, int(T_length/4), 1, **kwargs),
        ))


        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
            self.edge_importance_k = nn.ParameterList([
                nn.Parameter(torch.ones(self.K_A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
            self.edge_importance_k = [1] * len(self.st_gcn_networks)



        # fcn for prediction
        self.fcn = nn.Conv2d(channel3, num_class, kernel_size=1)

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)


        # forwad
        for gcn, importance, importance_k in zip(self.st_gcn_networks, self.edge_importance,self.edge_importance_k):
            x, _ = gcn(x, self.A * importance, self.K_A* importance_k)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance, importance_k in zip(self.st_gcn_networks, self.edge_importance,self.edge_importance_k):
            x, self_A = gcn(x, self.A * importance, self.K_A* importance_k)

        #####predict
        # global pooling
        x_pred = F.avg_pool2d(x, x.size()[2:])
        x_pred = x_pred.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x_pred = self.fcn(x_pred)
        output = x_pred.view(x.size(0), -1)


        ####extract features
        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1) # N,C,T,V,M


        return output, feature

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_length,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        #self.k_org = nn.Parameter()

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
        self.gcn_self = ConvTemporalGraphical_self(in_channels, out_channels,
                                         kernel_size[1], t_length)


        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, A, K_A):

        res = self.residual(x)
        x_nature, _ = self.gcn(x, A)
        x_given, _ = self.gcn(x, K_A)
        x_learn, A_learn = self.gcn_self(x)
        x_3A = (x_nature + x_given + x_learn) / 3 
        x_3A = self.tcn(x_3A) + res


        return self.relu(x_3A), A_learn
