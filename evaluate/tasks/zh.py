

# CMTEB for Chinese
TASK_LIST_CLASSIFICATION_ZH = [
    "AmazonReviewsClassification",
    "IFlyTek",
    "JDReview",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MultilingualSentiment",
    "OnlineShopping",
    "TNews",
    "Waimai",
] # 9

TASK_LIST_CLUSTERING_ZH = [
    "CLSClusteringP2P",
    "CLSClusteringS2S",
    "ThuNewsClusteringP2P",
    "ThuNewsClusteringS2S",
] # 4

TASK_LIST_PAIR_CLASSIFICATION_ZH = [
    "Cmnli",
    "Ocnli",
] # 2

TASK_LIST_RERANKING_ZH = [
    "CMedQAv1-reranking",
    "CMedQAv2-reranking",
    "MMarcoReranking",
    "T2Reranking",
] # 4

TASK_LIST_RETRIEVAL_ZH = [
    "CmedqaRetrieval",
    "CovidRetrieval",
    "DuRetrieval",
    "EcomRetrieval",
    "MedicalRetrieval",
    "MMarcoRetrieval",
    "T2Retrieval",
    "VideoRetrieval",
] # 8

TASK_LIST_STS_ZH = [
    "AFQMC",
    "ATEC",
    "BQ",
    "LCQMC",
    "PAWSX",
    "QBQTC",
    "STS22",
    "STSB",
] # 8

TASK_LIST_ZH = (
    TASK_LIST_RETRIEVAL_ZH
    + TASK_LIST_RERANKING_ZH
    + TASK_LIST_STS_ZH
    + TASK_LIST_PAIR_CLASSIFICATION_ZH
    + TASK_LIST_CLUSTERING_ZH
    + TASK_LIST_CLASSIFICATION_ZH
)
