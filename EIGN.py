# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_add_pool, GINEConv, GINConv, GCNConv

from appnp import APPNP
from layer import DGNN, NodeWithEdgeUpdate


class GIN(nn.Module):
    def __init__(self, node_dim, hidden_dim, in_drop_rate, out_drop_rate):
        super().__init__()
        hidden_list = [hidden_dim * 2, hidden_dim * 2, hidden_dim]
        gin_dim = hidden_dim
        self.lin_node = nn.Sequential(Linear(node_dim, hidden_dim), nn.SiLU())
        self.encoder_inter = Encoder(node_dim, hidden_dim)

        self.mlp_encode = nn.Sequential(
            nn.Linear(hidden_dim, gin_dim),
            nn.Dropout(in_drop_rate),
            nn.LeakyReLU(),
            nn.BatchNorm1d(gin_dim))

        self.edge_inter_attr = EdgeAttr(gin_dim)
        self.edge_inter_update = NodeWithEdgeUpdate(gin_dim, gin_dim)
        self.edge_intra_attr = EdgeAttr(gin_dim)
        self.edge_intra_update = NodeWithEdgeUpdate(gin_dim, gin_dim)

        self.gin1 = GINEConv(nn.Sequential(
            nn.Linear(gin_dim, gin_dim),
            nn.Dropout(in_drop_rate),
            nn.LeakyReLU(),
            nn.BatchNorm1d(gin_dim)))
        self.dgnn1 = DGNN([hidden_dim, hidden_dim, hidden_dim, hidden_dim])
        self.lin1 = nn.Sequential(Linear(hidden_dim * 2, hidden_dim), nn.SiLU())

        self.gin3 = GINEConv(nn.Sequential(
            nn.Linear(gin_dim, gin_dim),
            nn.Dropout(in_drop_rate),
            nn.LeakyReLU(),
            nn.BatchNorm1d(gin_dim)))
        self.dgnn3 = DGNN([hidden_dim, hidden_dim, hidden_dim, hidden_dim])
        self.lin3 = nn.Sequential(Linear(hidden_dim * 2, hidden_dim), nn.SiLU())

        self.gin4 = GINConv(nn.Sequential(
            nn.Linear(gin_dim, gin_dim),
            nn.Dropout(in_drop_rate),
            nn.LeakyReLU(),
            nn.BatchNorm1d(gin_dim)))

        self.fc = FC(gin_dim, hidden_list, out_drop_rate)
        self.lin_out = nn.Linear(hidden_dim * 2, 1)

    def forward(self, data):
        x_raw, edge_index_inter, edge_index_intra, edge_index_aug, pos = \
            data.x, data.edge_index_inter, data.edge_index_intra, data.edge_index_aug, data.pos
        
        # print(edge_index_intra)

        edge_weight = torch.norm((pos[edge_index_inter[0]] - pos[edge_index_inter[1]]), p=2, dim=1)
        # edge_weight = nn.Sigmoid()(edge_weight)
        x_inter, xg = self.encoder_inter(x_raw, edge_index_inter, data.batch, edge_weight)
        x_raw = self.lin_node(x_raw)
        # x = x_raw
        x = self.mlp_encode(x_inter + x_raw)

        edge_attr_inter = self.edge_inter_attr(pos, edge_index_inter)
        edge_attr_inter = self.edge_inter_update(x, edge_index_inter, edge_attr_inter)
        edge_attr_intra = self.edge_inter_attr(pos, edge_index_intra)
        edge_attr_intra = self.edge_inter_update(x, edge_index_intra, edge_attr_intra)

        x_inter1 = self.gin1(x, edge_index_inter, edge_attr_inter)
        x_inter2 = self.dgnn1(x, edge_index_inter)

        x_inter = torch.concat([x_inter1, x_inter2], dim=-1)
        x_inter = self.lin1(x_inter)

        x_intra1 = self.gin3(x, edge_index_intra, edge_attr_intra)
        x_intra2 = self.dgnn3(x, edge_index_intra)
        
        x_intra = torch.concat([x_intra1, x_intra2], dim=-1)
        x_intra = self.lin3(x_intra)

        x_mask = self.gin4(x, edge_index_aug)

        x_c = x_inter + x_intra + x_mask
        # x_c = x_intra + x_inter
        x = self.fc(x_c)

        # 多个池化层组合，随后融合
        pooled_output = global_add_pool(x, data.batch)
        x = torch.concat([pooled_output, xg], dim=-1)
        x = self.lin_out(x)

        return x.view(-1)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.SiLU()
        )
        self.propagate_inter = APPNP(K=1, alpha=0.1)
        self.pool = global_add_pool
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = self.linear(x)
        x_psc = F.normalize(x, p=2, dim=1) * 1.8
        x = self.propagate_inter(x_psc, edge_index, edge_weight=edge_weight)
        xg = self.pool(x, batch)
        x = self.dropout(x)
        return x, xg


class EdgeAttr(nn.Module):
    def __init__(self, hidden_dim):
        super(EdgeAttr, self).__init__()
        self.mlp_diff_pos = nn.Sequential(nn.Linear(16, hidden_dim), nn.Sigmoid())
        # self.mlp_deg = nn.Sequential(nn.Linear(16, hidden_dim), nn.Sigmoid())
        # self.linc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, pos, edge_index):
        coord_diff = pos[edge_index[0]] - pos[edge_index[1]]
        diff_feat = self.mlp_diff_pos(
            _rbf(torch.norm(coord_diff, p=2, dim=1), D_min=0., D_max=6., D_count=16, device=pos.device))

        return diff_feat


class FC(nn.Module):
    def __init__(self, in_dim, hidden_list, dropout):
        super(FC, self).__init__()
        self.predict = nn.ModuleList()
        for hidden_dim in hidden_list:
            self.predict.append(nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                # nn.Dropout(dropout),
                nn.LeakyReLU(),
                nn.BatchNorm1d(hidden_dim)))
            in_dim = hidden_dim
        self.predict.append(nn.Linear(in_dim, in_dim))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)

        return h


#  定义RBF层
def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count).to(device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF
