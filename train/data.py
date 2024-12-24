import math
import os
import os.path
import random
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Iterator, Sized, Optional

import datasets
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

from .arguments import DataArguments


logger = logging.getLogger(__name__)


class TrainDatasetForEmbedding(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer
    ):
        args.task_column = "task_name" if args.task_column is None else args.task_column

        if os.path.isdir(args.train_data):
            train_datasets = []
            for task in os.listdir(args.train_data):
                task_path = os.path.join(args.train_data, task)

                if os.path.isdir(task_path):
                    temp_dataset = datasets.load_dataset('json', data_dir=task_path, split='train', num_proc=int(os.cpu_count()))
                else:
                    temp_dataset = datasets.load_dataset('json', data_files=task_path, split='train', num_proc=int(os.cpu_count()))

                if len(temp_dataset) > args.max_example_num_per_dataset:
                    temp_dataset = temp_dataset.select(
                        sorted(random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset)))
                
                task_column = [task] * len(temp_dataset)
                temp_dataset = temp_dataset.add_column(args.task_column, task_column)

                logger.info(f'Loading Task: {task} === {len(temp_dataset)}')

                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train', num_proc=int(os.cpu_count()))

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        query = self.dataset[item]['query']
        if self.args.query_instruction_for_retrieval is not None:
            query = self.args.query_instruction_for_retrieval + query

        passages = []

        assert isinstance(self.dataset[item]['pos'], list)
        pos = random.choice(self.dataset[item]['pos'])
        passages.append(pos)

        if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
            num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['neg']))
            negs = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)
        passages.extend(negs)

        if self.args.passage_instruction_for_retrieval is not None:
            passages = [self.args.passage_instruction_for_retrieval+p for p in passages]
        return query, passages
    
    def _get_task_indices(self) -> List[int]:
        task_indices_dict = defaultdict(list)
        for i, example in enumerate(self.dataset):
            dataset_name = example.get(self.args.task_column, "unamed")
            task_indices_dict[dataset_name].append(i)
        return list(task_indices_dict.values())


@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 32
    passage_max_len: int = 128

    def padding_score(self, teacher_score):
        group_size = None
        for scores in teacher_score:
            if scores is not None:
                group_size = len(scores)
                break
        if group_size is None:
            return None

        padding_scores = [100.0] + [0.0] * (group_size - 1)
        new_teacher_score = []
        for scores in teacher_score:
            if scores is None:
                new_teacher_score.append(padding_scores)
            else:
                new_teacher_score.append(scores)
        return new_teacher_score

    def __call__(self, features):
        query = [f[0] for f in features]
        passage = [f[1] for f in features]

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])

        q_collated = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            passage,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        return {"query": q_collated, "passage": d_collated}


class DistributedInTaskSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Sampler used when training on multiple datasets to ensure each 
    batch only contains samples from one dataset for the majority of cases.
    See https://github.com/ContextualAI/gritlm/blob/a122855d6578a4f0980ea20340d5c9e1dd59d8c4/gritlm/training/data.py#L284
    """

    def __init__(self, batch_size: int = 8, homogeneous_ratio: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.total_batch_size = self.batch_size * self.num_replicas
        self.homogeneous_ratio = homogeneous_ratio

    def __iter__(self) -> Iterator[int]:
        task_indices = self.dataset._get_task_indices()

        # deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            for i, sub_indices in enumerate(task_indices):
                task_indices[i] = torch.tensor(sub_indices)[torch.randperm(len(sub_indices), generator=g)].tolist()

        # Create batches with only samples from one dataset
        batch_indices = [list(torch.split(torch.tensor(sub_indices), self.total_batch_size)) for sub_indices in task_indices]
        # Create separate batches from the remaining samples
        incomplete_indices = []
        for b in batch_indices:
            if len(b[-1]) < self.total_batch_size:
                incomplete_indices.append(b.pop())

        if incomplete_indices:
            # Randomly permute the incomplete indices
            order = torch.randperm(len(incomplete_indices), generator=g).tolist()
            incomplete_indices = torch.cat([incomplete_indices[i] for i in order])
            # Then split again into groups of four & drop the last one if it is incomplete
            mixed_batches = list(torch.split(incomplete_indices, self.total_batch_size))

            # if len(mixed_batches[-1]) < self.total_batch_size:
            #     if not self.drop_last:
            #         pass
            #     else:
            #         mixed_batches.pop()

            if self.drop_last and len(mixed_batches[-1]) % self.num_replicas != 0:
                last_size = math.ceil((len(mixed_batches[-1]) - self.num_replicas) / self.num_replicas) * self.num_replicas
            else:
                last_size = math.ceil(len(mixed_batches[-1]) / self.num_replicas) * self.num_replicas

            if not self.drop_last:
                # add extra samples to make it evenly divisible
                flat_task_indices = [i for indices in task_indices for i in indices]
                padding_size = last_size - len(mixed_batches[-1])
                if padding_size <= len(flat_task_indices):
                    mixed_batches[-1] = torch.cat((mixed_batches[-1], torch.tensor(flat_task_indices[:padding_size])), dim=0)
                else:
                    mixed_batches[-1] = torch.cat((mixed_batches[-1], torch.tensor(flat_task_indices * math.ceil(padding_size / len(flat_task_indices)))[:padding_size]), dim=0)
                logger.info(f"Pad the last batch with {padding_size} samples. If you want to avoid this, please set 'dataloader_drop_last=True'.")
            else:
                # remove tail of data to make it evenly divisible.
                mixed_batches[-1] = mixed_batches[-1][:last_size]

            # Merge all batches to look like [...tensor([259, 273, 284, 289]), tensor([262, 280, 295, 258]), ...]
            batch_indices = sum(batch_indices, []) + mixed_batches
            logger.info(f"Using global batch size {self.total_batch_size} created {len(batch_indices) - len(mixed_batches)} single-dataset batches & {len(mixed_batches)} mixed dataset batches.")
        else:
            batch_indices = sum(batch_indices, [])
            logger.info(f"Using global batch size {self.total_batch_size} created {len(batch_indices)} single-dataset batches.")

        # Randomly permute the order of all batches, then merge them to look like tensor([...259, 273, 284, 289, 262, 280, 295, 258...])
        if self.shuffle:
            if len(batch_indices[-1]) < self.total_batch_size:
                order = torch.randperm(len(batch_indices) - 1, generator=g).tolist() + [len(batch_indices) - 1]
                logger.info(f"Put the incomplete batch with {len(batch_indices[-1])} samples at the last step. If you want to avoid this, try to manually cut the dataset to fit in total batch size {self.total_batch_size}.")
            else:
                order = torch.randperm(len(batch_indices), generator=g).tolist()
            
            batch_indices = self._shuffle_last_elements(batch_indices, int(self.total_batch_size * (1.0 - self.homogeneous_ratio)))
        else:
            order = [i for i in range(len(batch_indices))]
            logger.info("homogeneous_ratio does not work when self.shuffle=False.")

        indices = [int(i) for i in torch.cat([batch_indices[i] for i in order]).tolist()]

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
    
    def _shuffle_last_elements(self, nested_list, m):
        if m <= 0:
            return nested_list
        
        # Step 1: extract the last m elements from each sublist
        candidate_pool = []
        for tensor in nested_list:
            candidate_pool.append(tensor[-m:])
        candidate_pool = torch.cat(candidate_pool)
        
        # Step 2: shuffe the candidate pool
        indices = torch.randperm(candidate_pool.size(0))
        candidate_pool = candidate_pool[indices]
        
        # Step 3: insert the shullfed m elements into the sublist
        start_idx = 0
        for i, tensor in enumerate(nested_list):
            end_idx = start_idx + m
            nested_list[i][-m:] = candidate_pool[start_idx:end_idx]
            start_idx = end_idx
        
        return nested_list