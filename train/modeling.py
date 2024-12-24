import copy
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist
from peft import LoraConfig, TaskType, get_peft_model
from torch import Tensor, nn
from transformers import AutoModel
from transformers.file_utils import ModelOutput

from .losses import CustomMatryoshkaLoss

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class BiEncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 model_name: str = None,
                 normlized: bool = False,
                 sentence_pooling_method: str = 'cls',
                 use_lora: bool = False,
                 lora_target_modules: List[str] = None,
                 lora_rank=8,
                 lora_alpha=16,
                 lora_dropout=0,
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 use_inbatch_neg: bool = True,
                 use_expaned_neg: bool = True,
                 use_matryoshka: bool = False,
                 matryoshka_dims: List[int] = None,
                 matryoshka_weights: Optional[List[Union[float, int]]] = None,
                 n_dims_per_step: int = -1,
                 query_instruction_for_retrieval: Optional[str] = None,
                 passage_instruction_for_retrieval: Optional[str] = None,
                 ):
        super().__init__()
        try:
            self.model = AutoModel.from_pretrained(
                model_name, attn_implementation="flash_attention_2", trust_remote_code=True, use_cache=False)
        except BaseException:
            logger.info("Loading flash_attention_2 failed. Load in normal setting.")
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_cache=False)

        if use_lora:
            peft_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=lora_target_modules,
            )
            self.model.enable_input_require_grads()
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.passage_instruction_for_retrieval = passage_instruction_for_retrieval

        self.use_lora = use_lora
        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.use_inbatch_neg = use_inbatch_neg
        self.use_expaned_neg = use_expaned_neg
        self.config = self.model.config

        if not normlized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError(
                    'Distributed training has not been initialized for representation all gather.')
            #     logger.info("Run in a single GPU, set negatives_cross_device=False")
            #     self.negatives_cross_device = False
            # else:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        self.use_matryoshka = use_matryoshka
        self.matryoshka_loss = None
        if self.use_matryoshka:
            logger.info(f"use_matryoshka: {self.use_matryoshka}")
            self.matryoshka_loss = CustomMatryoshkaLoss(self.cross_entropy,
                                                        matryoshka_dims=matryoshka_dims,
                                                        matryoshka_weights=matryoshka_weights,
                                                        n_dims_per_step=n_dims_per_step,
                                                        temperature=temperature,
                                                        use_inbatch_neg=use_inbatch_neg,
                                                        use_expaned_neg=use_expaned_neg)

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]
        elif self.sentence_pooling_method == 'lasttoken':
            left_padding = (mask[:, -1].sum() == mask.shape[0])
            if left_padding:
                return hidden_state[:, -1]
            else:
                sequence_lengths = mask.sum(dim=1) - 1
                batch_size = hidden_state.shape[0]
                return hidden_state[torch.arange(
                    batch_size, device=hidden_state.device), sequence_lengths]
        else:
            raise NotImplementedError

    def encode(self, features):
        if features is None:
            return None
        psg_out = self.model(**features, return_dict=True)
        p_reps = self.sentence_embedding(psg_out.last_hidden_state, features['attention_mask'])
        if self.normlized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def forward(self, query: Dict[str, Tensor] = None,
                passage: Dict[str, Tensor] = None, teacher_score: Tensor = None):
        if self.use_matryoshka:
            origin_normlized = self.normlized
            self.normlized = False
            q_reps = self.encode(query)
            p_reps = self.encode(passage)
            self.normlized = origin_normlized
        else:
            q_reps = self.encode(query)
            p_reps = self.encode(passage)

        if self.training:
            if self.negatives_cross_device and self.use_inbatch_neg:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            if not self.use_matryoshka:
                group_size = p_reps.size(0) // q_reps.size(0)

                if self.use_inbatch_neg:
                    scores = self.compute_similarity(q_reps, p_reps) / self.temperature  # B B*G
                    scores = scores.view(q_reps.size(0), -1)

                    target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                    target = target * group_size

                    if self.use_expaned_neg:
                        scores_q_reps = self.compute_similarity(q_reps, q_reps) / self.temperature
                        scores_q_reps = scores_q_reps.masked_fill_(
                            torch.eye(
                                scores_q_reps.size(0),
                                dtype=torch.int,
                                device=scores_q_reps.device).bool(),
                            torch.finfo(scores_q_reps.dtype).min
                            )
 
                        scores_p_reps = self.compute_similarity(p_reps[target], p_reps[target]) / self.temperature
                        scores_p_reps = scores_p_reps.masked_fill_(
                            torch.eye(
                                scores_p_reps.size(0),
                                dtype=torch.int,
                                device=scores_p_reps.device).bool(),
                            torch.finfo(scores_p_reps.dtype).min
                            )

                        scores = torch.cat([
                            scores, 
                            scores_q_reps.view(q_reps.size(0), -1), 
                            scores_p_reps.view(q_reps.size(0), -1)], 
                            dim=-1)

                    loss = self.compute_loss(scores, target)
                else:
                    scores = self.compute_similarity(q_reps[:, None, :,], p_reps.view(
                        q_reps.size(0), group_size, -1)).squeeze(1) / self.temperature  # B G

                    scores = scores.view(q_reps.size(0), -1)
                    target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
                    loss = self.compute_loss(scores, target)
            else:
                scores = self.compute_similarity(q_reps, p_reps)
                loss = self.matryoshka_loss(q_reps, p_reps)
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        device = self.model.device
        model = copy.deepcopy(self.model.to('cpu'))
        if self.use_lora:
            model = model.merge_and_unload()
        model.save_pretrained(output_dir)

        self.model.to(device)
