"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

from __future__ import annotations

import os
import functools
import logging
import argparse

import sklearn
import torch
import torch.multiprocessing as mp
from sentence_transformers import SentenceTransformer

from mteb import MTEB, get_tasks

from utils import DRESModel, get_task_instruct_by_task_name
from tasks.en import TASK_LIST_EN
from tasks.zh import TASK_LIST_ZH
from tasks.fr import TASK_LIST_FR
from tasks.pl import TASK_LIST_PL
from tasks.ru import TASK_LIST_RU
from tasks.multilingual import TASK_LIST_MULTILINGUAL
from tasks.long import TASK_LIST_LONG


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")


def main_worker(rank, args, task_queue, languages):
    os.environ['RANK'] = str(rank)
    # torch.distributed.init_process_group(backend="hccl",rank=rank)

    try:
        import torch_npu
        if torch.npu.is_available():  
            model = SentenceTransformer(args.model_name_or_path, trust_remote_code=args.trust_remote_code, truncate_dim=args.truncate_dim, device=f"npu:{rank}")
        elif torch.cuda.is_available():
            model = SentenceTransformer(args.model_name_or_path, trust_remote_code=args.trust_remote_code, truncate_dim=args.truncate_dim, device=f"cuda:{rank}")
        else:
            raise ValueError("No GPU available")
    except:
        model = SentenceTransformer(args.model_name_or_path, trust_remote_code=args.trust_remote_code, truncate_dim=args.truncate_dim, device=f"cuda:{rank}")
    
    model.max_seq_length = args.max_seq_length
    model.encode = functools.partial(model.encode, normalize_embeddings=True, batch_size=args.batch_size, show_progress_bar=True,)
    model = DRESModel(model)

    while not task_queue.empty():
        task = task_queue.get()
        
        if args.use_instruct:
            instruct = get_task_instruct_by_task_name(task)
            model.model.prompts[task] = instruct
            print(f"Running task: {task} with instruct: {instruct}")
        else:
            print(f"Running task: {task} without instruct")
        
        if args.eval_lang == "en":
            eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
        else:
            eval_splits = None

        tasks = get_tasks(tasks=[task], languages=languages)
        evaluation = MTEB(tasks=tasks)
        evaluation.run(model, output_folder=args.output_dir, eval_splits=eval_splits, encode_kwargs={'batch_size': args.batch_size}, verbosity=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default=None, type=str)
    parser.add_argument('--trust_remote_code', action="store_true")
    parser.add_argument('--use_instruct', action="store_true")
    parser.add_argument('--eval_lang', default="en", type=str)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--max_seq_length', default=512, type=int)
    parser.add_argument('--truncate_dim', default=None, type=int)
    parser.add_argument('--master_port', default=12355, type=int)
    parser.add_argument('--output_dir', default=None, type=str)
    args = parser.parse_args()

    print(args)

    if args.eval_lang == "en":
        tasks = TASK_LIST_EN
        languages = ["en", "eng-Latn"]
    elif args.eval_lang == "zh":
        tasks = TASK_LIST_ZH
        languages = ["zh", "zh-CN", "cmn-Hans", "cmo-Hans"]
    elif args.eval_lang == "fr":
        tasks = TASK_LIST_FR
        languages = ["fr", "fra-Latn"]
    elif args.eval_lang == "pl":
        tasks = TASK_LIST_PL
        languages = ["pl", "pol-Latn"]
    elif args.eval_lang == "ru":
        tasks = TASK_LIST_RU
        languages = ["ru", "rus-Cyrl"]
    elif args.eval_lang == "multilingual":
        tasks = TASK_LIST_MULTILINGUAL
        languages = None
    elif args.eval_lang == "long":
        tasks = TASK_LIST_LONG
        languages = None
    else:
        raise ValueError

    mp_manager = mp.Manager()
    task_queue = mp_manager.Queue()
    for task in tasks:
        task_queue.put(task)

    try:
        import torch_npu
        if torch.npu.is_available():  
            world_size = torch_npu.npu.device_count()
        elif torch.cuda.is_available():
            world_size = torch.cuda.device_count()
        else:
            raise ValueError("No GPU available")
    except:
        world_size = torch.cuda.device_count()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.master_port)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    mp.spawn(main_worker, nprocs=world_size, args=(args, task_queue, languages), join=True)