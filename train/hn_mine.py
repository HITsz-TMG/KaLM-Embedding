import argparse
import json
import random
import numpy as np

import torch
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, is_torch_npu_available
from FlagEmbedding import FlagModel


class FlagModelv2(FlagModel):
    def __init__(
            self,
            model_name_or_path: str = None,
            pooling_method: str = 'cls',
            normalize_embeddings: bool = True,
            query_instruction_for_retrieval: str = None,
            use_fp16: bool = True
    ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="BAAI/bge-base-en", type=str)
    parser.add_argument('--pooling_method', default="cls", type=str)
    parser.add_argument('--input_file', default=None, type=str)
    parser.add_argument('--candidate_pool', default=None, type=str)
    parser.add_argument('--output_file', default=None, type=str)
    parser.add_argument('--range_for_sampling', default="10-210", type=str, help="range to sample negatives")
    parser.add_argument('--use_gpu_for_searching', action='store_true', help='use faiss-gpu')
    parser.add_argument('--negative_number', default=15, type=int, help='the number of negatives')
    parser.add_argument('--filter_topk', default=None, type=int, help='filter the sample by ranking consisteny in top-k')
    parser.add_argument('--query_instruction_for_retrieval', default="")

    return parser.parse_args()


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


def find_knn_neg(model, input_file, candidate_pool, output_file, sample_range, negative_number, filter_topk, use_gpu):
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

    if candidate_pool is not None:
        if not isinstance(candidate_pool, list):
            candidate_pool = get_corpus(candidate_pool)
        corpus = list(set(candidate_pool))
    else:
        corpus = list(set(corpus))

    print(f'inferencing embedding for corpus (number={len(corpus)})--------------')
    p_vecs = model.encode(corpus, batch_size=256)
    p_vecs_dict = {p: p_v for p_v, p in zip(p_vecs, corpus)}

    print(f'inferencing embedding for queries (number={len(queries)})--------------')
    q_vecs = model.encode_queries(queries, batch_size=256)

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

        if filter_topk is not None and len(all_p_scores) > filter_topk and pos_score < all_p_scores[filter_topk]:
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

    with open(output_file, 'w') as f:
        for data in dump_data:
            if len(data['neg']) < negative_number:
                data['neg'].extend(random.sample(corpus, negative_number - len(data['neg'])))
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    args = get_args()
    sample_range = args.range_for_sampling.split('-')
    sample_range = [int(x) for x in sample_range]

    model = FlagModelv2(args.model_name_or_path, query_instruction_for_retrieval=args.query_instruction_for_retrieval)

    find_knn_neg(model,
                 input_file=args.input_file,
                 candidate_pool=args.candidate_pool,
                 output_file=args.output_file,
                 sample_range=sample_range,
                 negative_number=args.negative_number,
                 filter_topk=args.filter_topk,
                 use_gpu=args.use_gpu_for_searching)
