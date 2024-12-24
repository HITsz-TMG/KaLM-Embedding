import logging
import random
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class CustomMatryoshkaLoss(nn.Module):
    def __init__(
        self,
        loss: nn.Module,
        matryoshka_dims: List[int],
        matryoshka_weights: Optional[List[Union[float, int]]] = None,
        n_dims_per_step: int = -1,
        temperature: float = 1.0,
        use_inbatch_neg: bool = False,
        use_expaned_neg: bool = False
    ) -> None:
        super().__init__()
        self.loss = loss
        self.matryoshka_dims = matryoshka_dims
        if matryoshka_weights is None:
            matryoshka_weights = [1] * len(matryoshka_dims)
        self.matryoshka_weights = matryoshka_weights
        self.n_dims_per_step = n_dims_per_step
        self.temperature = temperature
        self.use_inbatch_neg = use_inbatch_neg
        self.use_expaned_neg = use_expaned_neg
        logger.info(f"matryoshka_dims: {matryoshka_dims}")
        logger.info(f"matryoshka_weights: {matryoshka_weights}")
        logger.info(f"n_dims_per_step: {n_dims_per_step}")
        logger.info(f"temperature: {temperature}")
        logger.info(f"use_inbatch_neg: {use_inbatch_neg}")
        logger.info(f"use_expaned_neg: {use_expaned_neg}")

    def forward(self, q_reps: Tensor, p_reps: Tensor) -> Tensor:
        dim_indices = range(len(self.matryoshka_dims))
        if self.n_dims_per_step > 0 and self.n_dims_per_step < len(dim_indices):
            dim_indices = random.sample(dim_indices, self.n_dims_per_step)

        loss = 0.0
        target = None
        group_size = p_reps.size(0) // q_reps.size(0)
        for idx in dim_indices:
            dim = self.matryoshka_dims[idx]
            weight = self.matryoshka_weights[idx]
            q_reps_shrinked = self.shrink(dim, q_reps)
            p_reps_shrinked = self.shrink(dim, p_reps)

            if self.use_inbatch_neg:
                scores = self.compute_similarity(
                    q_reps_shrinked, p_reps_shrinked) / self.temperature  # B B*G
                scores = scores.view(q_reps_shrinked.size(0), -1)
                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * group_size

                if self.use_expaned_neg:
                    scores_q_reps = self.compute_similarity(
                        q_reps_shrinked, q_reps_shrinked) / self.temperature
                    scores_q_reps = scores_q_reps.masked_fill_(
                        torch.eye(
                            scores_q_reps.size(0),
                            dtype=torch.int,
                            device=scores_q_reps.device).bool(),
                        torch.finfo(
                            scores_q_reps.dtype).min)
                    scores_p_reps = self.compute_similarity(
                        p_reps_shrinked[target], p_reps_shrinked[target]) / self.temperature
                    scores_p_reps = scores_p_reps.masked_fill_(
                        torch.eye(
                            scores_p_reps.size(0),
                            dtype=torch.int,
                            device=scores_p_reps.device).bool(),
                        torch.finfo(
                            scores_p_reps.dtype).min)

                    scores = torch.cat([
                        scores, 
                        scores_q_reps.view(q_reps_shrinked.size(0), -1), 
                        scores_p_reps.view(q_reps_shrinked.size(0), -1)
                        ], dim=-1)

                loss += weight * self.loss(scores, target)
            else:
                scores = self.compute_similarity(q_reps_shrinked[:, None, :,], p_reps_shrinked.view(
                    q_reps_shrinked.size(0), group_size, -1)).squeeze(1) / self.temperature  # B G
                scores = scores.view(q_reps_shrinked.size(0), -1)
                target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
                loss += weight * self.loss(scores, target)
        return loss

    def get_config_dict(self) -> Dict[str, Any]:
        return {
            "loss": self.loss.__class__.__name__,
            "matryoshka_dims": self.matryoshka_dims,
            "matryoshka_weights": self.matryoshka_weights,
            "n_dims_per_step": self.n_dims_per_step,
        }

    def shrink(self, dim, tensor: Tensor) -> Tensor:
        tensor_dim = tensor.shape[-1]
        if dim > tensor_dim:
            raise ValueError(
                f"Dimension {dim} in matryoshka_dims cannot be greater than the model's embedding dimension: {tensor_dim}"
            )
        tensor = tensor[..., : dim]
        tensor = F.normalize(tensor, p=2, dim=-1)
        return tensor

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))
