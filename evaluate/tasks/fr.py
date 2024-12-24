
# MTEB for French
TASK_LIST_CLASSIFICATION_FR = [
    "AmazonReviewsClassification",
    "MasakhaNEWSClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
]

TASK_LIST_CLUSTERING_FR = [
    "AlloProfClusteringP2P",
    "AlloProfClusteringS2S",
    "HALClusteringS2S",
    "MLSUMClusteringP2P",
    "MLSUMClusteringS2S",
    "MasakhaNEWSClusteringP2P",
    "MasakhaNEWSClusteringS2S",
]

TASK_LIST_PAIR_CLASSIFICATION_FR = [
    "OpusparcusPC",
    "PawsXPairClassification",
]

TASK_LIST_RERANKING_FR = [
    "AlloprofReranking",
    "SyntecReranking",
]

TASK_LIST_RETRIEVAL_FR = [
    "BSARDRetrieval",
    "AlloprofRetrieval",
    "SyntecRetrieval",
    "XPQARetrieval",
    "MintakaRetrieval",
]

TASK_LIST_STS_FR = [
    "SICKFr",
    "STS22",
    "STSBenchmarkMultilingualSTS",
]

TASK_LIST_SUMMARIZATION_FR = [
    "SummEvalFr",
]

TASK_LIST_FR = (
    TASK_LIST_RETRIEVAL_FR
    + TASK_LIST_STS_FR
    + TASK_LIST_RERANKING_FR
    + TASK_LIST_CLUSTERING_FR
    + TASK_LIST_PAIR_CLASSIFICATION_FR
    + TASK_LIST_CLASSIFICATION_FR
    + TASK_LIST_SUMMARIZATION_FR
)
