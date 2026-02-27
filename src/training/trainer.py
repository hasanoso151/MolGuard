"""
MolGuard — Eğitim Yöneticisi / Training Manager
Erken durdurma ve metrik takibi ile eğitim döngüsü
Training loop with early stopping and metric tracking
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import time, os


class Trainer:
    def __init__(self, model, device, evaluator, lr=0.001,
                 weight_decay=1e-5, patience=15):
        self.model = model
        self.device = device
        self.evaluator = evaluator
        self.optimizer = Adam(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
        self.patience = patience

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(batch)
            y = batch.y.float()
            is_valid = ~torch.isnan(y)
            loss = F.binary_cross_entropy_with_logits(
                pred[is_valid], y[is_valid])
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def evaluate(self, loader):
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                pred = self.model(batch)
                y_true.append(batch.y.cpu().numpy())
                y_pred.append(pred.cpu().numpy())
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        return self.evaluator.eval(
            {'y_true': y_true, 'y_pred': y_pred})['rocauc']

    def train(self, train_loader, valid_loader, test_loader,
              epochs=100, save_path='checkpoints/best.pt'):
        best_valid = 0
        patience_counter = 0
        history = {'loss': [], 'valid_auc': [], 'test_auc': []}

        for epoch in range(1, epochs + 1):
            loss = self.train_one_epoch(train_loader)
            v_auc = self.evaluate(valid_loader)
            t_auc = self.evaluate(test_loader)

            history['loss'].append(loss)
            history['valid_auc'].append(v_auc)
            history['test_auc'].append(t_auc)

            if v_auc > best_valid:
                best_valid = v_auc
                best_test = t_auc
                patience_counter = 0
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        return {'best_valid': best_valid, 'best_test': best_test,
                'history': history}
