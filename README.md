# KaLM-Embedding



## :sparkles: Overview

Code for training and evaluation of our [KaLM-Embedding](https://huggingface.co/collections/HIT-TMG/kalm-embedding-67316afa4c56f4fc1f58764b) models.
For a more comprehensive understanding of the technical details, please refer to our paper [KaLM-Embedding: Superior Training Data Brings A Stronger Embedding Model](https://arxiv.org/abs/2501.01028).



## :computer: Usage

### :rainbow: Environment

```
conda env create -f environment.yaml
conda activate kalm
```


### :pick: Hard-negative Mining with Filtering
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


## :loudspeaker: Acknowledgements

Specifically, our training code was forked from [FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding/tree/1.1/FlagEmbedding/baai_general_embedding/finetune). We have made modifications to suit our specific needs, but the core functionality and structure are derived from their excellent work.
Please check out their repository for more details!



## :link: Citation
Please cite the repo if you use the model or code in this repo.

```
@article{hu2025kalm,
  title={KaLM-Embedding: Superior Training Data Brings A Stronger Embedding Model},
  author={Hu, Xinshuo and Shan, Zifei and Zhao, Xinping and Sun, Zetian and Liu, Zhenyu and Li, Dongfang and Ye, Shaolin and Wei, Xinyuan and Chen, Qian and Hu, Baotian and others},
  journal={arXiv preprint arXiv:2501.01028},
  year={2025}
}
```



## :scroll: License

This repository respects to MIT license.