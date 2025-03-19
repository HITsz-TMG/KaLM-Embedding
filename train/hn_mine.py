import os
import argparse
import logging
import json
import random
import numpy as np
from typing import cast, List, Union
from dataclasses import dataclass, field

import torch
import faiss
from tqdm import tqdm
from transformers import HfArgumentParser, AutoTokenizer, AutoModel, is_torch_npu_available


logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s  %(filename)s : %(levelname)s  %(message)s", 
    datefmt="%Y-%m-%d %A %H:%M:%S") 
logger = logging.getLogger(__name__)


@dataclass
class HNMineArguments:
    model_name_or_path: str = field(
        default=None, 
        metadata={"help": "Path to the model for hard negative mining."}
    )
    pooling_method: str = field(
        default="cls", 
        metadata={"help": "The pooling method of the embedding model."}
    )
    input_file: str = field(
        default=None, 
        metadata={"help": "The input file path for hard negative mining."}
    )
    candidate_pool: str = field(
        default=None, 
        metadata={"help": "The candidate document file path for negative pool."}
    )
    output_file: str = field(
        default=None, 
        metadata={"help": "The output file path for saving."}
    )
    range_for_sampling: str = field(
        default="10-210", 
        metadata={"help": "The negative sampling range, e.g. 10-100"}
    )
    use_gpu_for_searching: bool = field(
        default=False, 
        metadata={"help": "Whether to use GPU for faiss searching. faiss-gpu neeeds to be installed."}
    )
    negative_number: int = field(
        default=15, 
        metadata={"help": "The number of mined negative samples."}
    )
    filter_topk: int = field(
        default=None, 
        metadata={"help": "The top-k threshold for ranking consistency filtering."}
    )
    query_instruction_for_retrieval: str = field(
        default="", 
        metadata={"help": "The query instruction for retrieval."}
    )
    passages_instruction_for_retrieval: str = field(
        default="", 
        metadata={"help": "The passage instruction for retrieval."}
    )
    batch_size: int = field(
        default=256, 
        metadata={"help": "The batch size for model encoding."}
    )


class FlagModel:
    def __init__(
            self,
            model_name_or_path: str = None,
            pooling_method: str = 'cls',
            normalize_embeddings: bool = True,
            query_instruction_for_retrieval: str = None,
            passages_instruction_for_retrieval: str = None,
            use_fp16: bool = False
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.passages_instruction_for_retrieval = passages_instruction_for_retrieval
        self.normalize_embeddings = normalize_embeddings
        self.pooling_method = pooling_method

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif is_torch_npu_available():
            self.device = torch.device("npu")
        else:
            self.device = torch.device("cpu")
            use_fp16 = False
        if use_fp16: self.model.half()
        self.model = self.model.to(self.device)

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            print(f"----------using {self.num_gpus}*GPUs----------")
            self.model = torch.nn.DataParallel(self.model)
    
    def encode_queries(self, queries: Union[List[str], str],
                       batch_size: int=256,
                       max_length: int=512) -> np.ndarray:
        '''
        This function will be used for retrieval task
        if there is a instruction for queries, we will add it to the query text
        '''
        if self.query_instruction_for_retrieval is not None:
            if isinstance(queries, str):
                input_texts = self.query_instruction_for_retrieval + queries
            else:
                input_texts = ['{}{}'.format(self.query_instruction_for_retrieval, q) for q in queries]
        else:
            input_texts = queries
        return self.encode(input_texts, batch_size=batch_size, max_length=max_length)
    
    def encode_corpus(self,
                      corpus: Union[List[str], str],
                      batch_size: int=256,
                      max_length: int=512) -> np.ndarray:
        '''
        This function will be used for retrieval task
        encode corpus for retrieval task
        if there is a instruction for corpus, we will add it to the corpus text
        '''
        if self.passages_instruction_for_retrieval is not None:
            if isinstance(corpus, str):
                input_texts = self.passages_instruction_for_retrieval + corpus
            else:
                input_texts = ['{}{}'.format(self.passages_instruction_for_retrieval, q) for q in corpus]
        else:
            input_texts = corpus
        return self.encode(input_texts, batch_size=batch_size, max_length=max_length)


    @torch.no_grad()
    def encode(self, sentences: Union[List[str], str], batch_size: int=256, max_length: int=512) -> np.ndarray:
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus
        self.model.eval()

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings", disable=len(sentences)<256):
            sentences_batch = sentences[start_index:start_index + batch_size]
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
            ).to(self.device)
            last_hidden_state = self.model(**inputs, return_dict=True).last_hidden_state
            embeddings = self.pooling(last_hidden_state, inputs['attention_mask'])
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)
            all_embeddings.append(embeddings.cpu().numpy())

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        if input_was_string:
            return all_embeddings[0]
        return all_embeddings


    def pooling(self,
                last_hidden_state: torch.Tensor,
                attention_mask: torch.Tensor=None):
        if self.pooling_method == 'cls':
            return last_hidden_state[:, 0]
        elif self.pooling_method == 'mean':
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'lasttoken':
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                return last_hidden_state[torch.arange(
                    batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise NotImplementedError


def create_index(embeddings, use_gpu):
    index = faiss.IndexFlatIP(len(embeddings[0]))
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if use_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        index = faiss.index_cpu_to_all_gpus(index, co=co)
    index.add(embeddings)
    return index


def batch_search(index,
                 query,
                 topk: int = 200,
                 batch_size: int = 64):
    all_scores, all_inxs = [], []
    for start_index in tqdm(range(0, len(query), batch_size), desc="Batches", disable=len(query) < 256):
        batch_query = query[start_index:start_index + batch_size]
        batch_scores, batch_inxs = index.search(np.asarray(batch_query, dtype=np.float32), k=topk)
        all_scores.extend(batch_scores.tolist())
        all_inxs.extend(batch_inxs.tolist())
    return all_scores, all_inxs


def get_corpus(candidate_pool):
    corpus = []
    for line in open(candidate_pool):
        line = json.loads(line.strip())
        corpus.append(line['text'])
    return corpus


def find_knn_neg(model, input_file, candidate_pool, output_file, sample_range, negative_number, filter_topk, batch_size, use_gpu):
    corpus = []
    queries = []
    train_data = []
    for line in open(input_file):
        line = json.loads(line.strip())
        train_data.append(line)
        corpus.extend(line['pos'])
        if 'neg' in line:
            corpus.extend(line['neg'])
        queries.append(line['query'])

    if candidate_pool is not None and candidate_pool != "" and candidate_pool.lower() != "none":
        if not isinstance(candidate_pool, list):
            candidate_pool = get_corpus(candidate_pool)
        corpus = list(set(candidate_pool))
    else:
        corpus = list(set(corpus))

    print(f'inferencing embedding for corpus (number={len(corpus)})--------------')
    p_vecs = model.encode(corpus, batch_size=batch_size)
    p_vecs_dict = {p: p_v for p_v, p in zip(p_vecs, corpus)}

    print(f'inferencing embedding for queries (number={len(queries)})--------------')
    q_vecs = model.encode_queries(queries, batch_size=batch_size)

    print('create index and search------------------')
    index = create_index(p_vecs, use_gpu=use_gpu)
    all_scores, all_inxs = batch_search(index, q_vecs, topk=sample_range[-1])
    assert len(all_inxs) == len(train_data)

    dump_data = []
    for i, data in enumerate(train_data):
        query = data['query']
        data['neg'] = data['neg'][:negative_number]
        
        q_v = q_vecs[i]
        p_v_list = np.array([p_vecs_dict[p] for p in data['pos']])
        pos_scores = np.einsum('i,ji->j', q_v, p_v_list)
        pos_score = max(pos_scores)
        all_p_scores = np.array(all_scores[i])

        if filter_topk is not None and filter_topk > 0 and len(all_p_scores) > filter_topk and pos_score < all_p_scores[filter_topk]:
            continue

        inxs = all_inxs[i][sample_range[0]:sample_range[1]]
        filtered_inx = []
        for inx in inxs:
            if inx == -1: break
            if corpus[inx] not in data['pos'] and corpus[inx] != query:
                filtered_inx.append(inx)

        if len(filtered_inx) > negative_number - len(data['neg']):
            filtered_inx = random.sample(filtered_inx, negative_number - len(data['neg']))
        data['neg'].extend([corpus[inx] for inx in filtered_inx])
        
        dump_data.append(data)
    
    print(f"Data num: {len(dump_data)} / {len(train_data)}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for data in dump_data:
            if len(data['neg']) < negative_number:
                data['neg'].extend(random.sample(corpus, negative_number - len(data['neg'])))
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = HfArgumentParser(HNMineArguments)
    args, = parser.parse_args_into_dataclasses()
    logger.info("Hard Negative Mining Parameters %s", args)

    sample_range = args.range_for_sampling.split('-')
    sample_range = [int(x) for x in sample_range]

    model = FlagModel(model_name_or_path=args.model_name_or_path, 
                      pooling_method=args.pooling_method,
                      query_instruction_for_retrieval=args.query_instruction_for_retrieval, 
                      passages_instruction_for_retrieval=args.passages_instruction_for_retrieval,
                      use_fp16=False)

    find_knn_neg(model,
                 input_file=args.input_file,
                 candidate_pool=args.candidate_pool,
                 output_file=args.output_file,
                 sample_range=sample_range,
                 negative_number=args.negative_number,
                 filter_topk=args.filter_topk,
                 batch_size=args.batch_size,
                 use_gpu=args.use_gpu_for_searching)
