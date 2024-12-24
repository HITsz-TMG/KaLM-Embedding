# KaLM-Embedding



## :sparkles: Overview

Code for training and evaluation of our [KaLM-Embedding](https://huggingface.co/collections/HIT-TMG/kalm-embedding-67316afa4c56f4fc1f58764b) models.



## :computer: Usage

### :rainbow: Environment

```
conda env create -f environment.yaml
conda activate kalm
```


### :fire: Hard-negative Mining with Filtering
```
bash ./scripts/hn_mine.sh
```
You can customize the `filter_topk` parameter to set the threshold of ranking consistency filtering.


### :fire: Train
```
bash ./scripts/train.sh
```


### :bar_chart: Evaluation
We have provided a code for evaluating MTEB using multiple GPUs, which allocates each task from the task set to a single GPU in a queue-based manner, thereby enhancing evaluation efficiency.
```
bash ./scripts/eval_mteb.sh
```


## Acknowledgements

Specifically, our training code was forked from [FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding/tree/1.1/FlagEmbedding/baai_general_embedding/finetune). We have made modifications to suit our specific needs, but the core functionality and structure are derived from their excellent work.
Please check out their repository for more details!



## :scroll: License

This repository respects to MIT license.