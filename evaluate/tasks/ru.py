
# MTEB for Russian
TASK_LIST_CLASSIFICATION_RU = [
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "GeoreviewClassification",
    "HeadlineClassification",
    "InappropriatenessClassification",
    "KinopoiskClassification",
    "RuReviewsClassification",
    "RuSciBenchGRNTIClassification",
    "RuSciBenchOECDClassification",
]

TASK_LIST_MULTILABEL_CLASSIFICATION_RU = [
    "CEDRClassification",
    "SensitiveTopicsClassification",
]

TASK_LIST_CLUSTERING_RU  = [
    "GeoreviewClusteringP2P",
    "RuSciBenchGRNTIClusteringP2P",
    "RuSciBenchOECDClusteringP2P"
]

TASK_LIST_PAIR_CLASSIFICATION_RU  = [
    "TERRa",
]

TASK_LIST_RERANKING_RU  = [
    "RuBQReranking",
    "MIRACLReranking",
]

TASK_LIST_RETRIEVAL_RU = [
    "RiaNewsRetrieval",
    "RuBQRetrieval",
    "MIRACLRetrieval",
]

TASK_LIST_STS_RU= [
    "STS22",
    "RUParaPhraserSTS",
    "RuSTSBenchmarkSTS",
]

TASK_LIST_SUMMARIZATION_RU = [
]

TASK_LIST_RU = (
    TASK_LIST_RETRIEVAL_RU
    + TASK_LIST_STS_RU
    + TASK_LIST_RERANKING_RU
    + TASK_LIST_CLUSTERING_RU
    + TASK_LIST_PAIR_CLASSIFICATION_RU
    + TASK_LIST_CLASSIFICATION_RU
    + TASK_LIST_MULTILABEL_CLASSIFICATION_RU
    + TASK_LIST_SUMMARIZATION_RU
)
