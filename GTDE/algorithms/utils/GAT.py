import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros


class DenseGATConv(nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GATConv`."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.0,
            bias: bool = True,
            add_self_loops: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.lin = Linear(in_channels, heads * out_channels, bias=False,
                          weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.empty(1, 1, heads, out_channels))
        self.att_dst = Parameter(torch.empty(1, 1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def forward(self, x: Tensor, adj: Tensor, mask: Optional[Tensor] = None):
        r"""Forward pass.

        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            adj (torch.Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
                The adjacency tensor is broadcastable in the batch dimension,
                resulting in a shared adjacency matrix for the complete batch.
            mask (torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x  # [B, N, F]
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj  # [B, N, N]

        H, C = self.heads, self.out_channels
        B, N, _ = x.size()

        if self.add_self_loops:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1.0

        x = self.lin(x).view(B, N, H, C)  # [B, N, H, C]

        alpha_src = torch.sum(x * self.att_src, dim=-1)  # [B, N, H]
        alpha_dst = torch.sum(x * self.att_dst, dim=-1)  # [B, N, H]

        alpha = alpha_src.unsqueeze(1) + alpha_dst.unsqueeze(2)  # [B, N, N, H]

        # Weighted and masked softmax:
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = alpha.masked_fill(adj.unsqueeze(-1) == 0, float('-inf'))
        alpha = alpha.softmax(dim=2)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.matmul(alpha.movedim(3, 1), x.movedim(2, 1))
        out = out.movedim(1, 2)  # [B,N,H,C]

        if self.concat:
            out = out.reshape(B, N, H * C)
        else:
            out = out.mean(dim=2)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(-1, N, 1).to(x.dtype)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class GAT(torch.nn.Module):
    def __init__(self, input_dim, GAT_dim, hidden_dim, heads):
        super(GAT, self).__init__()
        self.gat1 = DenseGATConv(input_dim, GAT_dim, heads=heads)
        self.gat2 = DenseGATConv(GAT_dim * heads, hidden_dim, heads=1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.elu = nn.ELU()

    def forward(self, x, edge_index):
        gat1 = self.gat1(x, edge_index)
        gat2 = self.gat2(self.elu(gat1), edge_index)
        return self.norm(self.elu(gat2))
