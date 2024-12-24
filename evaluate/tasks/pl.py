
# MTEB for Polish
TASK_LIST_CLASSIFICATION_PL = [
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "CBD",
    "PolEmo2.0-IN",
    "PolEmo2.0-OUT",
    "AllegroReviews",
    "PAC"
]

TASK_LIST_CLUSTERING_PL = [
    "EightTagsClustering",
]

TASK_LIST_PAIR_CLASSIFICATION_PL = [
    "SICK-E-PL",
    "PpcPC",
    "CDSC-E",
    "PSC",
]

TASK_LIST_RERANKING_PL = [
]

TASK_LIST_RETRIEVAL_PL = [
    "MSMARCO-PL",
    "HotpotQA-PL",
    "NQ-PL",
    "DBPedia-PL",
    "NFCorpus-PL",
    "Quora-PL",
    "TRECCOVID-PL",
    "SCIDOCS-PL",
    "SciFact-PL",
    "FiQA-PL",
    "ArguAna-PL",
]

TASK_LIST_STS_PL = [
    "STS22",
    "SICK-R-PL",
    "CDSC-R",
]

TASK_LIST_SUMMARIZATION_PL = [
]

TASK_LIST_PL = (
    TASK_LIST_RETRIEVAL_PL
    + TASK_LIST_STS_PL
    + TASK_LIST_RERANKING_PL
    + TASK_LIST_CLUSTERING_PL
    + TASK_LIST_PAIR_CLASSIFICATION_PL
    + TASK_LIST_CLASSIFICATION_PL
    + TASK_LIST_SUMMARIZATION_PL
)