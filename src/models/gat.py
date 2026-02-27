"""
MolGuard — GAT (Graph Attention Network) Modeli
Dikkat mekanizmalı GNN / GNN with attention mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class MolGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=4,
                 num_heads=8, num_tasks=12, dropout=0.3):
        super(MolGAT, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.node_encoder = nn.Linear(input_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads,
                                       heads=num_heads, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tasks)
        )

    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        x = self.node_encoder(x.float())

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch_idx)
        return self.prediction_head(x)
