import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List, Dict, Optional


class StructureAwarePathContrastiveLearning(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        temperature: float = 0.1,
        dropout_rate: float = 0.05,
        use_replace: bool = True,
        use_extend: bool = True,
        use_delete: bool = True,
        use_shuffle: bool = True
    ):
        super(StructureAwarePathContrastiveLearning, self).__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.dropout_rate = dropout_rate
        self.path_dropout = nn.Dropout(p=dropout_rate)
        self.use_replace = use_replace
        self.use_extend = use_extend
        self.use_delete = use_delete
        self.use_shuffle = use_shuffle
        self._update_perturb_types()
        self.path_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def _update_perturb_types(self):
        self.perturb_types = []
        if self.use_replace:
            self.perturb_types.append('replace')
        if self.use_extend:
            self.perturb_types.append('extend')
        if self.use_delete:
            self.perturb_types.append('delete')
        if self.use_shuffle:
            self.perturb_types.append('shuffle')

    def get_edge_features(self, grid_outputs: torch.Tensor, i: int, j: int) -> torch.Tensor:
        return grid_outputs[i, j]

    def get_path_representation(
        self,
        grid_outputs: torch.Tensor,
        path_indices: List[int]
    ) -> torch.Tensor:
        if len(path_indices) < 1:
            return torch.zeros(
                self.hidden_dim,
                device=grid_outputs.device,
                dtype=grid_outputs.dtype
            )
        edge_features = []
        for t in range(len(path_indices)):
            curr = path_indices[t]
            nxt = path_indices[(t + 1) % len(path_indices)]
            edge_feat = self.get_edge_features(grid_outputs, curr, nxt)
            edge_features.append(edge_feat)
        if len(edge_features) == 0:
            return torch.zeros(
                self.hidden_dim,
                device=grid_outputs.device,
                dtype=grid_outputs.dtype
            )
        edge_stack = torch.stack(edge_features, dim=0)
        path_repr = edge_stack.mean(dim=0)
        path_repr = self.path_encoder(path_repr)
        return path_repr

    def construct_positive_view(self, anchor: torch.Tensor) -> torch.Tensor:
        positive = self.path_dropout(anchor)
        return positive

    def _replace_perturb(self, true_path: List[int], seq_len: int) -> Optional[List[int]]:
        if len(true_path) < 2:
            return None
        available = [i for i in range(seq_len) if i not in true_path]
        if not available:
            return None
        neg_path = true_path.copy()
        replace_idx = random.randint(0, len(neg_path) - 1)
        new_token = random.choice(available)
        neg_path[replace_idx] = new_token
        return neg_path

    def _extend_perturb(self, true_path: List[int], seq_len: int) -> Optional[List[int]]:
        if len(true_path) >= seq_len:
            return None
        available = [i for i in range(seq_len) if i not in true_path]
        if not available:
            return None
        neg_path = true_path.copy()
        insert_pos = random.randint(0, len(neg_path))
        new_token = random.choice(available)
        neg_path.insert(insert_pos, new_token)
        return neg_path

    def _delete_perturb(self, true_path: List[int]) -> Optional[List[int]]:
        if len(true_path) < 3:
            return None
        neg_path = true_path.copy()
        delete_idx = random.randint(0, len(neg_path) - 1)
        del neg_path[delete_idx]
        if len(neg_path) < 2:
            return None
        return neg_path

    def _shuffle_perturb(self, true_path: List[int]) -> Optional[List[int]]:
        if len(true_path) < 3:
            return None
        original = true_path.copy()
        neg_path = true_path.copy()
        for _ in range(10):
            random.shuffle(neg_path)
            if neg_path != original:
                return neg_path
        return None

    def construct_negative_paths(
        self,
        true_path: List[int],
        seq_len: int,
        num_negatives: int = 3,
        perturb_weights: Optional[Dict[str, float]]=None
    ) -> List[List[int]]:
        negatives = []
        if len(true_path) < 2:
            return negatives
        if not self.perturb_types:
            return negatives
        if perturb_weights is None:
            perturb_weights = {pt: 1.0 for pt in self.perturb_types}
        valid_types = []
        valid_weights = []
        for pt in self.perturb_types:
            weight = perturb_weights.get(pt, 0.0)
            if weight > 0:
                valid_types.append(pt)
                valid_weights.append(weight)
        if not valid_types:
            return negatives
        seen = set()
        max_attempts = max(num_negatives * 5, 10)
        attempts = 0
        while len(negatives) < num_negatives and attempts < max_attempts:
            attempts += 1
            perturb_type = random.choices(valid_types, weights=valid_weights, k=1)[0]
            neg_path = None
            if perturb_type == 'replace':
                neg_path = self._replace_perturb(true_path, seq_len)
            elif perturb_type == 'extend':
                neg_path = self._extend_perturb(true_path, seq_len)
            elif perturb_type == 'delete':
                neg_path = self._delete_perturb(true_path)
            elif perturb_type == 'shuffle':
                neg_path = self._shuffle_perturb(true_path)
            if neg_path is None:
                continue
            neg_key = tuple(neg_path)
            if neg_key == tuple(true_path):
                continue
            if neg_key in seen:
                continue
            seen.add(neg_key)
            negatives.append(neg_path)
        return negatives

    def extract_entity_paths(
        self,
        grid_outputs: torch.Tensor,
        entities: List[List[Dict]],
        seq_lengths: List[int]
    ) -> List[Dict]:
        batch_paths = []
        for batch_idx, (batch_entities, seq_len) in enumerate(zip(entities, seq_lengths)):
            if not batch_entities:
                continue
            batch_grid = grid_outputs[batch_idx]
            for entity in batch_entities:
                indices = entity.get('index', [])
                valid_indices = [idx for idx in indices if 0 <= idx < seq_len]
                if len(valid_indices) < 2:
                    continue
                path_repr = self.get_path_representation(batch_grid, valid_indices)
                batch_paths.append({
                    'feature': path_repr,
                    'indices': valid_indices,
                    'batch_idx': batch_idx,
                    'entity_info': entity,
                    'seq_len': seq_len
                })
        return batch_paths

    def compute_infonce_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: List[torch.Tensor]
    ) -> torch.Tensor:
        if len(negatives) == 0:
            return torch.tensor(0.0, device=anchor.device, dtype=anchor.dtype)
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative_features = torch.stack([F.normalize(neg, dim=-1) for neg in negatives], dim=0)
        pos_sim = torch.sum(anchor * positive, dim=-1)
        neg_sim = torch.matmul(negative_features, anchor.unsqueeze(-1)).squeeze(-1)
        logits = torch.cat([pos_sim.unsqueeze(0), neg_sim], dim=0)
        logits = logits / self.temperature
        labels = torch.zeros(1, dtype=torch.long, device=anchor.device)
        loss = F.cross_entropy(logits.unsqueeze(0), labels)
        return loss

    def forward(
        self,
        grid_outputs: torch.Tensor,
        entities: List[List[Dict]],
        seq_lengths: List[int],
        num_negatives_per_path: int = 3,
        perturb_weights: Optional[Dict[str, float]]=None
    ) -> torch.Tensor:
        if grid_outputs.dim() != 4:
            return torch.tensor(0.0, device=grid_outputs.device, dtype=grid_outputs.dtype)
        if not self.perturb_types:
            return torch.tensor(0.0, device=grid_outputs.device, dtype=grid_outputs.dtype)
        all_paths = self.extract_entity_paths(grid_outputs, entities, seq_lengths)
        if len(all_paths) == 0:
            return torch.tensor(0.0, device=grid_outputs.device, dtype=grid_outputs.dtype)
        total_loss = torch.tensor(0.0, device=grid_outputs.device, dtype=grid_outputs.dtype)
        valid_pairs = 0
        for path_info in all_paths:
            anchor_feat = path_info['feature']
            positive_feat = self.construct_positive_view(anchor_feat)
            true_indices = path_info['indices']
            batch_idx = path_info['batch_idx']
            seq_len = path_info['seq_len']
            batch_grid = grid_outputs[batch_idx]
            neg_paths = self.construct_negative_paths(
                true_path=true_indices,
                seq_len=seq_len,
                num_negatives=num_negatives_per_path,
                perturb_weights=perturb_weights
            )
            if not neg_paths:
                continue
            neg_features = []
            for neg_path in neg_paths:
                neg_feat = self.get_path_representation(batch_grid, neg_path)
                neg_features.append(neg_feat)
            if not neg_features:
                continue
            loss = self.compute_infonce_loss(
                anchor=anchor_feat,
                positive=positive_feat,
                negatives=neg_features
            )
            total_loss = total_loss + loss
            valid_pairs += 1
        if valid_pairs == 0:
            return torch.tensor(0.0, device=grid_outputs.device, dtype=grid_outputs.dtype)
        contrastive_loss = total_loss / valid_pairs
        return contrastive_loss
