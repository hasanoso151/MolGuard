"""
MolGuard — Veri Seti Yükleyici / Dataset Loader
Tüm OGB moleküler veri setlerini yükler ve hazırlar
Loads and prepares all OGB molecular datasets
"""

import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader

# PyTorch uyumluluk / Compatibility
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])


class MolGuardDataLoader:
    """
    MolGuard veri seti yöneticisi / MolGuard dataset manager
    Tüm veri setlerini yükler ve DataLoader'ları hazırlar
    Loads all datasets and prepares DataLoaders
    """

    DATASETS = {
        'toxicity': 'ogbg-moltox21',
        'sider': 'ogbg-molsider',
        'bbbp': 'ogbg-molbbbp',
        'clintox': 'ogbg-molclintox',
    }

    def __init__(self, root='data/raw/', batch_size=64):
        self.root = root
        self.batch_size = batch_size
        self.datasets = {}
        self.loaders = {}

    def load_dataset(self, task_name):
        """Tek bir veri seti yükle / Load a single dataset"""
        if task_name not in self.DATASETS:
            raise ValueError(f"Bilinmeyen görev / Unknown task: {task_name}")

        dataset = PygGraphPropPredDataset(
            name=self.DATASETS[task_name], root=self.root
        )
        split_idx = dataset.get_idx_split()

        self.datasets[task_name] = dataset
        self.loaders[task_name] = {
            'train': DataLoader(dataset[split_idx['train']],
                              batch_size=self.batch_size, shuffle=True),
            'valid': DataLoader(dataset[split_idx['valid']],
                              batch_size=self.batch_size, shuffle=False),
            'test': DataLoader(dataset[split_idx['test']],
                             batch_size=self.batch_size, shuffle=False),
        }

        return dataset

    def load_all(self):
        """Tüm veri setlerini yükle / Load all datasets"""
        for task_name in self.DATASETS:
            print(f"  ⏳ {task_name} yükleniyor / loading...")
            self.load_dataset(task_name)
            print(f"  ✅ {task_name}: {len(self.datasets[task_name])} molekül")
        return self.datasets, self.loaders

    def get_info(self):
        """Veri seti bilgilerini döndür / Return dataset info"""
        info = {}
        for name, ds in self.datasets.items():
            info[name] = {
                'size': len(ds),
                'num_tasks': ds.num_tasks,
                'node_feat_dim': ds[0].x.shape[1],
                'edge_feat_dim': ds[0].edge_attr.shape[1],
            }
        return info
