import logging
from typing import Dict, Union

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, WordEmbeddings

from mteb.model_meta import ModelMeta
from mteb.evaluation.evaluators import model_encode

from tasks.zh import (
    TASK_LIST_ZH, TASK_LIST_CLUSTERING_ZH, TASK_LIST_STS_ZH, TASK_LIST_CLASSIFICATION_ZH, TASK_LIST_PAIR_CLASSIFICATION_ZH, TASK_LIST_RERANKING_ZH, TASK_LIST_RETRIEVAL_ZH,
)
from tasks.en import (
    TASK_LIST_EN, TASK_LIST_CLUSTERING_EN, TASK_LIST_STS_EN, TASK_LIST_CLASSIFICATION_EN, TASK_LIST_PAIR_CLASSIFICATION_EN, TASK_LIST_RERANKING_EN, TASK_LIST_RETRIEVAL_EN, TASK_LIST_SUMMARIZATION_EN
)
from tasks.fr import (
    TASK_LIST_FR, TASK_LIST_CLUSTERING_FR, TASK_LIST_STS_FR, TASK_LIST_CLASSIFICATION_FR, TASK_LIST_PAIR_CLASSIFICATION_FR, TASK_LIST_RERANKING_FR, TASK_LIST_RETRIEVAL_FR, TASK_LIST_SUMMARIZATION_FR
)
from tasks.pl import (
    TASK_LIST_PL, TASK_LIST_CLUSTERING_PL, TASK_LIST_STS_PL, TASK_LIST_CLASSIFICATION_PL, TASK_LIST_PAIR_CLASSIFICATION_PL, TASK_LIST_RERANKING_PL, TASK_LIST_RETRIEVAL_PL, TASK_LIST_SUMMARIZATION_PL
)
from tasks.ru import (
    TASK_LIST_RU, TASK_LIST_CLUSTERING_RU, TASK_LIST_STS_RU, TASK_LIST_CLASSIFICATION_RU, TASK_LIST_PAIR_CLASSIFICATION_RU, TASK_LIST_RERANKING_RU, TASK_LIST_RETRIEVAL_RU, TASK_LIST_SUMMARIZATION_RU, TASK_LIST_MULTILABEL_CLASSIFICATION_RU
)

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("utils")


class DRESModel:
    """Dense Retrieval Exact Search (DRES) requires an encode_queries & encode_corpus method.
    This class converts a model with just an .encode method into DRES format.
    https://github.com/embeddings-benchmark/mteb/blob/d9153a7870140cb802ebed649db4ee7d8d013437/mteb/evaluation/evaluators/RetrievalEvaluator.py#L331C1-L408C81
    """

    mteb_model_meta: Union[ModelMeta, None]

    def __init__(self, model, **kwargs):
        self.model = model
        self.use_sbert_model = isinstance(model, SentenceTransformer)
        self.save_corpus_embeddings = kwargs.get("save_corpus_embeddings", False)
        self.corpus_embeddings = {}

    def encode_queries(
        self, queries: list[str], *, prompt_name: str, batch_size: int, **kwargs
    ):
        if self.use_sbert_model:
            if isinstance(self.model._first_module(), Transformer):
                logger.info(
                    f"Queries will be truncated to {self.model.get_max_seq_length()} tokens."
                )
            elif isinstance(self.model._first_module(), WordEmbeddings):
                logger.warning(
                    "Queries will not be truncated. This could lead to memory issues. In that case please lower the batch_size."
                )

        return model_encode(
            queries,
            model=self.model,
            prompt_name=prompt_name,
            batch_size=batch_size,
            **kwargs,
        )

    def encode_corpus(
        self,
        corpus: list[dict[str, str]],
        prompt_name: str,
        batch_size: int,
        request_qid: Union[str, None] = None,
        **kwargs,
    ):
        if (
            request_qid
            and self.save_corpus_embeddings
            and len(self.corpus_embeddings) > 0
        ):
            return self.corpus_embeddings[request_qid]

        if isinstance(corpus, dict):
            sentences = [
                (corpus["title"][i] + " " + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        elif isinstance(corpus[0], dict):
            sentences = [
                (doc["title"] + " " + doc["text"]).strip()
                if "title" in doc
                else doc["text"].strip()
                for doc in corpus
            ]
        else:
            sentences = corpus

        corpus_embeddings = model_encode(
            sentences,
            model=self.model,
            prompt_name=None,  # Remove prompt for encode_corpus
            batch_size=batch_size,
            **kwargs,
        )

        if self.save_corpus_embeddings and request_qid:
            self.corpus_embeddings[request_qid] = corpus_embeddings
        return corpus_embeddings

    def encode(self, sentences: list[str], prompt_name: str, **kwargs):
        return self.encode_queries(sentences, prompt_name=prompt_name, **kwargs)


def get_default_instruct(task_description: str) -> str:
    if not task_description:
        return ''

    return 'Instruct: {} \n Query: '.format(task_description)


def get_task_instruct_by_task_name(task_name: str) -> str:
    if task_name in (
        TASK_LIST_CLASSIFICATION_EN + 
        TASK_LIST_CLASSIFICATION_ZH + 
        TASK_LIST_CLASSIFICATION_FR + 
        TASK_LIST_CLASSIFICATION_PL +
        TASK_LIST_CLASSIFICATION_RU +
        TASK_LIST_MULTILABEL_CLASSIFICATION_RU):
        task_name_to_instruct: Dict[str, str] = {
            'AmazonCounterfactualClassification': 'Given an Amazon review, judge whether it is counterfactual.',
            'AmazonPolarityClassification': 'Classifying Amazon reviews into positive or negative sentiment',
            'AmazonReviewsClassification': 'Classifying the given Amazon review into its appropriate rating category',
            'Banking77Classification': 'Given a online banking query, find the corresponding intents',
            'EmotionClassification': 'Classifying the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise',
            'ImdbClassification': 'Classifying the sentiment expressed in the given movie review text from the IMDB dataset',
            'MassiveIntentClassification': 'Given a user utterance as query, find the user intents',
            'MassiveScenarioClassification': 'Given a user utterance as query, find the user scenarios',
            'MTOPDomainClassification': 'Classifying the intent domain of the given utterance in task-oriented conversation',
            'MTOPIntentClassification': 'Classifying the intent of the given utterance in task-oriented conversation',
            'ToxicConversationsClassification': 'Classifying the given comments as either toxic or not toxic',
            'TweetSentimentExtractionClassification': 'Classifying the sentiment of a given tweet as either positive, negative, or neutral',
            # C-MTEB eval instructions
            'TNews': 'Categorizing the given news title',
            'IFlyTek': 'Given an App description text, find the appropriate fine-grained category',
            'MultilingualSentiment': 'Classifying sentiment of the customer review into positive, or negative',
            'JDReview': 'Classifying sentiment of the customer review for iPhoneinto positive or negative',
            'OnlineShopping': 'Classifying sentiment of the customer reviewinto positive or negative',
            'Waimai': 'Classify the customer review from a food takeaway platform into positive or negative',
            # MTEB-fr eval instructions
            'MasakhaNEWSClassification': 'Classifying the category of french news.',
            # MTEB-pl eval instructions
            "CBD":"Classifying the sentiment of polish tweet reviews",
            "PolEmo2.0-IN": "Classifying the sentiment of in-domain (medicine and hotels) online reviews",
            "PolEmo2.0-OUT":"Classifying the sentiment of out-of-domain (products and school) online reviews",
            "AllegroReviews": "Classifying the sentiment of reviews from e-commerce marketplace Allegro",
            "PAC": "Classifying the sentence into one of the two types: \"BEZPIECZNE_POSTANOWIENIE_UMOWNE\" and \"KLAUZULA_ABUZYWNA\"",
            # MTEB-ru eval instructions
            "GeoreviewClassification": "Classifying the sentiment of Russian reviews.",
            "HeadlineClassification": "Classifying the topic of Russian headlines.",
            "InappropriatenessClassification": "Detecting inappropriate messages on sensitive topics",
            "KinopoiskClassification": "Classifying the sentiment of Kinopoisk reviews.",
            "RuReviewsClassification": "Classifying the sentiment of Russian product reviews.",
            "RuSciBenchGRNTIClassification": "Classifying the topic of Russian scientific papers.",
            "RuSciBenchOECDClassification": "Classifying the topic of Russian scientific papers.",
            "CEDRClassification": "Classification of sentences by emotions.",
            "SensitiveTopicsClassification": "Detecting inappropriate messages on sensitive topics.",
        }
        return get_default_instruct(task_name_to_instruct[task_name])

    if task_name in (
        TASK_LIST_CLUSTERING_EN + 
        TASK_LIST_CLUSTERING_ZH + 
        TASK_LIST_CLUSTERING_FR + 
        TASK_LIST_CLUSTERING_PL + 
        TASK_LIST_CLUSTERING_RU):
        task_name_to_instruct: Dict[str, str] = {
            'ArxivClusteringP2P': 'Identify the main and secondary category of Arxiv papers based on the titles and abstracts',
            'ArxivClusteringS2S': 'Identify the main and secondary category of Arxiv papers based on the titles',
            'BiorxivClusteringP2P': 'Identify the main category of Biorxiv papers based on the titles and abstracts',
            'BiorxivClusteringS2S': 'Identify the main category of Biorxiv papers based on the titles',
            'MedrxivClusteringP2P': 'Identify the main category of Medrxiv papers based on the titles and abstracts',
            'MedrxivClusteringS2S': 'Identify the main category of Medrxiv papers based on the titles',
            'RedditClustering': 'Identify the topic or theme of Reddit posts based on the titles',
            'RedditClusteringP2P': 'Identify the topic or theme of Reddit posts based on the titles and posts',
            'StackExchangeClustering': 'Identify the topic or theme of StackExchange posts based on the titles',
            'StackExchangeClusteringP2P': 'Identify the topic or theme of StackExchange posts based on the given paragraphs',
            'TwentyNewsgroupsClustering': 'Identify the topic or theme of the given news articles',
            # C-MTEB eval instructions
            'CLSClusteringS2S': 'Identify the main category of scholar papers based on the titles',
            'CLSClusteringP2P': 'Identify the main category of scholar papers based on the titles and abstracts',
            'ThuNewsClusteringS2S': 'Identify the topic or theme of the given news articles based on the titles',
            'ThuNewsClusteringP2P': 'Identify the topic or theme of the given news articles based on the titles and contents',
            # MTEB-fr eval instructions
            "AlloProfClusteringP2P": "Identify the main category of Allo Prof document based on the titles and descriptions",
            "AlloProfClusteringS2S": "Identify the main category of Allo Prof document based on the titles",
            "HALClusteringS2S": "Identify the main category of academic passage based on the titles and contents",
            "MasakhaNEWSClusteringP2P": "Identify the topic or theme of the given news articles based on the titles and contents",
            "MasakhaNEWSClusteringS2S": "Identify the topic or theme of the given news articles based on the titles",
            "MLSUMClusteringP2P": "Identify the topic or theme of the given articles based on the titles and contents",
            "MLSUMClusteringS2S":  "Identify the topic or theme of the given articles based on the titles",
            # MTEB-pl eval instructions
            "EightTagsClustering": "Identify of headlines from social media posts in Polish  into 8 categories: film, history, food, medicine, motorization, work, sport and technology",
            # MTEB-ru eval instructions
            "GeoreviewClusteringP2P": "Identify the topic or theme of the Russian reviews.",
            "RuSciBenchGRNTIClusteringP2P": "Identify the topic or theme of the Russian articles.",
            "RuSciBenchOECDClusteringP2P": "Identify the topic or theme of the Russian articles.",
        }
        return get_default_instruct(task_name_to_instruct[task_name])

    if task_name in (TASK_LIST_RETRIEVAL_EN + TASK_LIST_RETRIEVAL_ZH + TASK_LIST_RETRIEVAL_FR + TASK_LIST_RETRIEVAL_PL + TASK_LIST_RETRIEVAL_RU):
        return get_default_instruct("Given a query, retrieve documents that answer the query.")
    
    if task_name in (TASK_LIST_RERANKING_EN + TASK_LIST_RERANKING_ZH + TASK_LIST_RERANKING_FR + TASK_LIST_RERANKING_PL + TASK_LIST_RERANKING_RU):
        return get_default_instruct("Given a query, retrieve documents that answer the query.")

    logging.warning(f"No instruction config for task {task_name}, use none instruction.")
    return get_default_instruct(None) 
