"""
MolGuard — Multi-Task GNN
Çok görevli GNN — tüm veri setlerini birlikte öğrenir
Learns from all datasets jointly with shared backbone + task-specific heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class MolMultiTaskGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=4,
                 num_heads=8, dropout=0.3,
                 num_tasks_tox=12, num_tasks_sider=27,
                 num_tasks_bbbp=1, num_tasks_clintox=2):
        super(MolMultiTaskGNN, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        # Paylaşılan omurga / Shared backbone
        self.node_encoder = nn.Linear(input_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads,
                                       heads=num_heads, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Görev kafaları / Task heads
        def make_head(n):
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, n)
            )

        self.head_toxicity = make_head(num_tasks_tox)
        self.head_sider = make_head(num_tasks_sider)
        self.head_bbbp = make_head(num_tasks_bbbp)
        self.head_clintox = make_head(num_tasks_clintox)

    def get_embedding(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        x = self.node_encoder(x.float())
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch_idx)

    def forward(self, batch, task='toxicity'):
        emb = self.get_embedding(batch)
        heads = {
            'toxicity': self.head_toxicity,
            'sider': self.head_sider,
            'bbbp': self.head_bbbp,
            'clintox': self.head_clintox,
        }
        return heads[task](emb)
