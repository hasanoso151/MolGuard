"""
MolGuard — GNNExplainer Modülü / GNNExplainer Module
Moleküler tahminlerin açıklanması / Explaining molecular predictions
"""

import torch
import numpy as np


class MolExplainer:
    """
    GNN tahminlerini açıklar / Explains GNN predictions
    Hangi atomlar ve bağlar önemli olduğunu gösterir
    Shows which atoms and bonds are important
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def get_node_importance(self, data):
        """
        Düğüm önem skorlarını hesapla
        Compute node importance scores using gradient-based method
        """
        self.model.eval()
        data = data.to(self.device)
        data.x.requires_grad_(True)

        # İleri geçiş / Forward pass
        batch_vec = torch.zeros(
            data.x.shape[0], dtype=torch.long).to(self.device)

        x = self.model.node_encoder(data.x.float())
        for i in range(self.model.num_layers):
            x = self.model.convs[i](x, data.edge_index)
            x = self.model.bns[i](x)
            x = torch.relu(x)

        # Düğüm gömmelerinin normu / Norm of node embeddings
        importance = torch.norm(x, dim=1).detach().cpu().numpy()
        importance = (importance - importance.min()) / \
                     (importance.max() - importance.min() + 1e-8)

        return importance

    def find_safer_alternative(self, mol_idx, dataset,
                                embeddings, tox_scores, top_k=3):
        """
        Daha güvenli alternatif bul / Find safer alternative
        Embedding uzayında benzer ama daha az toksik moleküller
        Similar but less toxic molecules in embedding space
        """
        from sklearn.metrics.pairwise import cosine_similarity

        target_emb = embeddings[mol_idx:mol_idx+1]
        target_tox = tox_scores[mol_idx]

        similarities = cosine_similarity(target_emb, embeddings)[0]
        safer_mask = tox_scores < target_tox

        safe_sim = similarities.copy()
        safe_sim[~safer_mask] = -1
        safe_sim[mol_idx] = -1

        top_indices = np.argsort(safe_sim)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if safe_sim[idx] > 0:
                results.append({
                    'index': idx,
                    'similarity': similarities[idx],
                    'toxicity': tox_scores[idx],
                    'reduction': (target_tox - tox_scores[idx]) / target_tox * 100
                })
        return results
