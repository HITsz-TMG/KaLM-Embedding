import os
from dataclasses import dataclass, field
from typing import Optional, List

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_lora : bool = field(
        default=False, metadata={"help": "Whether to use LoRA for finetuning"}
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None, metadata={"help": "target_modules in LoraConfig"}
    )
    lora_rank: Optional[int] = field(
        default=8, metadata={"help": "r in LoraConfig"}
    )
    lora_alpha: Optional[int] = field(
        default=16, metadata={"help": "lora_alpha in LoraConfig"}
    )
    lora_dropout: Optional[float] = field(
        default=0, metadata={"help": "lora_dropout in LoraConfig"}
    )



@dataclass
class DataArguments:
    train_data: str = field(
        default=None, metadata={"help": "Path to train data"}
    )
    train_group_size: int = field(default=8)

    query_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    passage_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    max_example_num_per_dataset: int = field(
        default=100000000, metadata={"help": "the max number of examples for each dataset"}
    )

    query_instruction_for_retrieval: str= field(
        default=None, metadata={"help": "instruction for query"}
    )
    passage_instruction_for_retrieval: str = field(
        default=None, metadata={"help": "instruction for passage"}
    )

    task_column: str = field(
        default="task_name", 
        metadata={"help": "the column name of each task data. you can also split the tasks by inputing train_data with different task files or directories"}
    )

    def __post_init__(self):
        if not os.path.exists(self.train_data):
            raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")

@dataclass
class RetrieverTrainingArguments(TrainingArguments):
    negatives_cross_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    temperature: Optional[float] = field(default=0.02)
    fix_position_embedding: bool = field(default=False, metadata={"help": "Freeze the parameters of position embeddings"})
    sentence_pooling_method: str = field(default='cls', metadata={"help": "the pooling method, should be cls or mean or lasttoken"})
    normlized: bool = field(default=True)
    use_inbatch_neg: bool = field(default=True, metadata={"help": "use passages in the same batch as negatives"})
    use_expaned_neg: bool = field(default=False, metadata={"help": "use cross queries and passages as negatives"})
    sample_intask_neg: bool = field(default=True, metadata={"help": "Whether to sample negative examples from the same task."})
    sample_intask_neg_ratio: float = field(default=1.0, metadata={"help": "The ratio of negative examples from the same task in one batch."})
    use_matryoshka: bool = field(default=False)
    matryoshka_dims: Optional[List[int]] = field(default=None)
    matryoshka_weights: Optional[List[float]] = field(default=None)
    n_dims_per_step: Optional[int] = field(default=-1)
    