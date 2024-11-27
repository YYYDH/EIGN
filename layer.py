import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, GCNConv, GINConv
from torch_geometric.utils import add_self_loops, degree
import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.typing import (
    Adj,
)
from torch_geometric.utils import (
    spmm,
)


# class EdgeGNNConv(MessagePassing):

#     def __init__(
#             self,
#             channels: int,
#             **kwargs,
#     ):
#         kwargs.setdefault('aggr', 'add')
#         super().__init__(**kwargs)
#         self._lin = nn.Linear(channels, channels, bias=False)
#         self._bias = nn.Parameter(torch.empty(channels))
#         self.reset_parameters()

#     def reset_parameters(self):
#         self._lin.reset_parameters()
#         self._bias.data.zero_()

#     def forward(
#             self,
#             x: torch.Tensor,
#             edge_index: torch.Tensor,
#             diff: torch.Tensor
#     ):
#         # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
#         h = x

#         x = self._lin(x)

#         # row, col = edge_index
#         # deg = degree(col, x.size(0), dtype=x.dtype)
#         # deg_inv_sqrt = deg.pow(-0.5)
#         # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#         # norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

#         out = self.propagate(edge_index, x=x, diff=diff)
#         out = out * h
#         out = out + self._bias

#         return out

#     def message(self, x_j, diff):
#         return diff.view(-1, 1) * x_j

#     def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
#         return spmm(adj_t, x, reduce=self.aggr)


class DegGNNConv(MessagePassing):

    def __init__(
            self,
            channels: int,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self._lin = nn.Linear(channels, channels, bias=False)
        self._bias = nn.Parameter(torch.empty(channels))
        self.reset_parameters()

    def reset_parameters(self):
        self._lin.reset_parameters()
        self._bias.data.zero_()

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor
    ):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        h = x

        x = self._lin(x)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = out * h
        out = out + self._bias

        return out

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)


class DGNN(nn.Module):

    def __init__(
            self,
            architecture: list,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._gcn = GCNConv(architecture[0], architecture[1])
        self._batchnorm1 = nn.BatchNorm1d(architecture[1])
        # self._ecn = EdgeGNNConv(architecture[1])
        # self._batchnorm2 = nn.BatchNorm1d(architecture[1])
        # self._dropout2 = nn.Dropout(p=dropout)
        self._dcn = DegGNNConv(architecture[1])
        self._batchnorm3 = nn.BatchNorm1d(architecture[1])
        self._dropout3 = nn.Dropout(p=dropout)
        self._output_layer = nn.Linear(architecture[1] * 2, architecture[-1])

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor
    ):
        h = self._gcn(x, edge_index)
        h = self._batchnorm1(h)
        # x = self._ecn(h, edge_index, diff)
        # x = self._batchnorm2(x)
        # x = self._dropout2(x)
        # xs.append(x)
        x = self._dcn(h, edge_index)
        x = self._batchnorm3(x)
        x = self._dropout3(x)

        xc = torch.concat([h, x], dim=-1)  # 把这些层得到的x逐一concat

        y_hat = self._output_layer(xc)

        return y_hat
    
    
class EdgeUpdate(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim):
        super(EdgeUpdate, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_feature_dim, edge_feature_dim),
            nn.ReLU(),
            nn.Linear(edge_feature_dim, edge_feature_dim)
        )
        # 新增参数用于生成自适应邻接权重
        self.adaptive_weight = nn.Linear(edge_feature_dim, edge_feature_dim)

    def forward(self, x_i, x_j, e_ij, adj_matrix=None, prob_remove=0.1):
        # 拼接节点特征并更新边特征
        edge_input = torch.cat([x_i, x_j], dim=-1)
        updated_edge = self.mlp(edge_input) + e_ij
        
        # 计算基于边特征的权重
        if adj_matrix is not None:
            # 生成边的自适应权重
            adaptive_weights = self.adaptive_weight(updated_edge)
            mask = torch.rand(updated_edge.shape[0]) > prob_remove  # 依据删除概率生成掩码
            
            # 应用掩码并归一化边特征
            updated_edge = updated_edge * mask.unsqueeze(-1) * adj_matrix
            updated_edge = F.normalize(updated_edge, p=2, dim=-1)
        
        return updated_edge


class NodeWithEdgeUpdate(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim):
        super(NodeWithEdgeUpdate, self).__init__()
        self.edge_update = EdgeUpdate(node_feature_dim, edge_feature_dim)

    def forward(self, x, edge_index, edge_attr, adj_matrix=None, prob_remove=0.1):
        # 获取边的两端节点特征
        row, col = edge_index
        x_i, x_j = x[row], x[col]

        # 更新边特征，传入邻接矩阵和边移除概率
        edge_attr = self.edge_update(x_i, x_j, edge_attr, adj_matrix, prob_remove)
        return edge_attr