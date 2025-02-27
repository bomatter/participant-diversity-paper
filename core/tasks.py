import torch.nn as nn


TASK_LABELS = {
    "normality": "normality",
    "dementia": "dementia_label",
    "sleep_stage": "sleep_stage",
}

LABEL_MAPPINGS = {
    "normality": {"abnormal": 0, "normal": 1},
    "dementia": {"normal": 0, "mci": 1, "dementia": 2},
    "sleep_stage": {"wake": 0, "nonrem1": 1, "nonrem2": 2, "nonrem3": 3, "rem": 4},
}    

TASK_METRICS = {
    "normality": {
        "BinaryAccuracy": {},
        "BinaryPrecision": {},
        "BinaryRecall": {},
        "BinaryF1Score": {},
        "BinaryAUROC": {},
    },
    "dementia": {
        "MulticlassAccuracy": {"num_classes": 3},
        "MulticlassPrecision": {"num_classes": 3},
        "MulticlassRecall": {"num_classes": 3},
        "MulticlassF1Score": {"num_classes": 3},
        "MulticlassAUROC": {"num_classes": 3},
    },
    "sleep_stage": {
        "MulticlassAccuracy": {"num_classes": 5},
        "MulticlassPrecision": {"num_classes": 5},
        "MulticlassRecall": {"num_classes": 5},
        "MulticlassF1Score": {"num_classes": 5},
        "MulticlassAUROC": {"num_classes": 5},
    },
}

TASK_LOSS_FUNCTIONS = {
    "normality": nn.BCEWithLogitsLoss,
    "dementia": nn.CrossEntropyLoss,
    "sleep_stage": nn.CrossEntropyLoss,
}