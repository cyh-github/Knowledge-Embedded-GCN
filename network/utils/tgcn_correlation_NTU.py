# The based unit of graph convolutional networks.

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ConvTemporalGraphical(nn.Module):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

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
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True,
                 num_subset=3):
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        inter_channels = out_channels // 4
        self.inter_c = inter_channels
        self.num_subset = num_subset
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * 3,
                kernel_size=(t_kernel_size, 1),
                padding=(t_padding, 0),
                stride=(t_stride, 1)),
            nn.BatchNorm2d(out_channels*3)
            )

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        self.soft = nn.Softmax(-2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()


    def forward(self, x):

        N, C, T, V = x.size()
        #A = torch.zeros(N,self.num_subset,V,V).cuda()
        y = None
        for i in range(self.num_subset):
            x1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            x2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            a = self.soft(torch.matmul(x1, x2) / x1.size(-1)) #N,V,V

            #thres = float(torch.mean(a))
            a = F.threshold(a,0.02,0)

            x_temp = x.view(N, C*T, V)
            z = self.conv_d[i](torch.matmul(x_temp, a).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)


        '''
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, 3, kc // 3, t, v)
        x = torch.einsum('nkctv,nkvw->nctw', (x, A))
        '''

        return self.relu(y), a
