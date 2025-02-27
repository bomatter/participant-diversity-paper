import torch
from torch import nn

import pandas as pd

import torchmetrics
from torchmetrics import MetricCollection
from torchmetrics.wrappers import MetricTracker

from core.tasks import TASK_METRICS, TASK_LOSS_FUNCTIONS, TASK_LABELS, LABEL_MAPPINGS


def get_class_weights(dataset, task, device="cpu"):
    if task in ["dementia"]:
        label = TASK_LABELS[task]
        class_distribution = dataset.info[label].value_counts(normalize=True)
        n_classes = dataset.info[label].nunique()

        class_weights = 1.0 / (class_distribution * n_classes)
        class_weights = class_weights.iloc[class_weights.index.map(LABEL_MAPPINGS[task])]  # Reorder to match label mappings
        class_weights = torch.tensor(class_weights.values, dtype=torch.float, device=device)  # Convert to tensor

        return class_weights
    else:
        return None


def build_loss_function(task, weight=None):
    if task not in TASK_LOSS_FUNCTIONS:
        raise ValueError(f"Task {task} not recognized.")
    return TASK_LOSS_FUNCTIONS[task](weight=weight)


def build_metrics(
        metrics: dict,
        names: list[str] = None,
        prefixes: list[str] = [],
        device: str = "cpu",
    ):
        """ Adds metrics to the tracker. """
        new_metrics = []
        for metric_name, args in metrics.items():
            if hasattr(torchmetrics.regression, metric_name):
                metric = getattr(torchmetrics.regression, metric_name)(**args)
            elif hasattr(torchmetrics.classification, metric_name):
                metric = getattr(torchmetrics.classification, metric_name)(**args)
            elif hasattr(torchmetrics.aggregation, metric_name):
                metric = getattr(torchmetrics.aggregation, metric_name)(**args)
            else:
                raise ValueError(
                    f"Metric {metric_name} not available in torchmetrics.regression, "
                    + "torchmetrics.classification, or torchmetrics.aggregation."
                )
            metric.persistent(True)
            new_metrics.append(metric)

        if names is not None:
            new_metrics = {name: metric for name, metric in zip(names, new_metrics)}

        if prefixes:
            new_metrics = [MetricTracker(MetricCollection(new_metrics, prefix=p)) for p in prefixes]
            for metric in new_metrics:
                metric.to(device)
                metric.increment()
        else:
            new_metrics = MetricTracker(MetricCollection(new_metrics))
            new_metrics.to(device)
            new_metrics.increment()

        return new_metrics


class Tracker:
    """Wrapper class to track train and validation metrics together."""
    
    def __init__(self, metrics: dict, n_batches_per_epoch: int, device: str = "cpu", wandb=None):
        
        self.epoch = 0
        self.batch = 0
        self.n_batches_per_epoch = n_batches_per_epoch

        self.metrics_train, self.metrics_val = build_metrics(
            metrics, prefixes=["train", "val"], device=device
        )
        self.loss_train, self.loss_val = build_metrics(
            {"MeanMetric": {}}, names=["Loss"], prefixes=["train", "val"], device=device
        )

        # Track best validation loss for early stopping
        self.evaluations_since_improvement = 0
        self.best_val_loss = float("inf")

        # Note: torchmetrics currently uses an unreliable way of determining whether to apply a sigmoid.
        # It is determined on a per-batch basis and checks if the predictions are floats outside of [0, 1].
        # For consistent behaviour, we always apply sigmoid for binary classification.
        self.apply_sigmoid = any([
            hasattr(torchmetrics.classification, m)
            and isinstance(
                getattr(torchmetrics.classification, m)(**args),
                torchmetrics.classification.stat_scores.BinaryStatScores
            ) for m, args in metrics.items()
        ])

        self.wandb = wandb


    def report_train_step(self, pred, target, loss, epoch=None, batch=None):
        assert batch is None or batch == self.batch, (
            "Inconsistent batch number. Expected {}, got {}. Make sure you call update every batch.".format(self.batch, batch)
        )
        assert epoch is None or epoch == self.epoch, (
            "Inconsistent epoch number. Expected {}, got {}. Make sure you call update every batch.".format(self.epoch, epoch)
        )

        if self.apply_sigmoid:
            pred = nn.functional.sigmoid(pred)

        self.metrics_train.update(pred, target)
        self.loss_train.update(loss)

        self.batch = (self.batch + 1) % self.n_batches_per_epoch
        self.epoch += 1 if self.batch == 0 else 0


    def report_val_step(self, pred, target, loss):
        if self.apply_sigmoid:
            pred = nn.functional.sigmoid(pred)

        self.metrics_val.update(pred, target)
        self.loss_val.update(loss)


    def report_val_done(self):
        """Call this method after all validation batches for an evaluation have been processed."""
        self._notify_wandb()

        current_val_loss = self.loss_val.compute()["valLoss"]
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.evaluations_since_improvement = 0
        else:
            self.evaluations_since_improvement += 1

        self.metrics_train.increment()
        self.metrics_val.increment()
        self.loss_train.increment()
        self.loss_val.increment()


    def _notify_wandb(self):
        if self.wandb is not None:
            self.wandb.log(self._compute_metrics())


    def _compute_metrics(self):
        # Collect metrics and losses into a dictionary
        metrics_dict = {
            "epoch": self.epoch,
            "batch": self.batch,
            "batch_global": self.epoch * self.n_batches_per_epoch + self.batch,
            **self.metrics_train.compute(),
            **self.metrics_val.compute(),
            **self.loss_train.compute(),
            **self.loss_val.compute()
        }
        metrics_dict = {k: v.tolist() if k not in ["epoch", "batch", "batch_global"] else v for k, v in metrics_dict.items()}
        return metrics_dict
    

def compute_participant_level_accuracy(
    sample_predictions,  # DataFrame with sample predictions
    dataset: str  # name of the dataset, i.e. "CAUEEG" or "TUAB"
):
    """Compute participant-level accuracy from sample-level predictions."""

    if dataset == "CAUEEG":
        def get_predicted_label(row):
            if row['predicted_dementia_0'] == max(row['predicted_dementia_0'], row['predicted_dementia_1'], row['predicted_dementia_2']):
                return 'normal'
            elif row['predicted_dementia_1'] == max(row['predicted_dementia_0'], row['predicted_dementia_1'], row['predicted_dementia_2']):
                return 'mci'
            else:
                return 'dementia'

        sample_predictions['predicted_label'] = sample_predictions.apply(get_predicted_label, axis=1)

        # To get the majority prediction
        def get_mode(series):
            return series.mode().iloc[0]

        # Group by the specified columns and compute the majority predicted label
        recording_predictions = (
            sample_predictions
            .groupby(['_record_id', 'dataset', 'task', 'model', 'n_participants', 'n_segments', 'seed'])
            .apply(lambda df: pd.Series({
                'predicted_label': get_mode(df['predicted_label']),
                'dementia_label': df['dementia_label'].iloc[0]
            }))
            .reset_index()
        )

        recording_predictions["correct"] = recording_predictions["predicted_label"] == recording_predictions["dementia_label"]

        metrics = (
            recording_predictions
            .groupby(['dataset', 'task', 'model', 'n_participants', 'n_segments', 'seed'])
            .agg({'correct': 'mean'})
            .reset_index()
            .rename(columns={'correct': 'participant_level_accuracy'})
        )
    elif dataset == "TUAB":
        sample_predictions['thresholded_prediction'] = sample_predictions['predicted_normality'] > 0
        recording_predictions = (
            sample_predictions
            .groupby(['_record_id', 'dataset', 'task', 'model', 'n_participants', 'n_segments', 'seed'])
            .agg({'predicted_normality': 'mean', 'thresholded_prediction': 'mean', 'normality': 'first'})
            .reset_index()
            .rename(columns={'predicted_normality': 'mean_predicted_logit', 'thresholded_prediction': 'mean_predicted_normality'})
        )

        recording_predictions['predicted_normality_decision_average'] = recording_predictions['mean_predicted_normality'] > 0.5
        recording_predictions['normality_bool'] = recording_predictions['normality'] == 'normal'
        recording_predictions["correct_decision_average"] = recording_predictions["predicted_normality_decision_average"] == recording_predictions["normality_bool"]

        metrics = (
            recording_predictions
            .groupby(['dataset', 'task', 'model', 'n_participants', 'n_segments', 'seed'])
            .agg({'correct_decision_average': 'mean'})
            .reset_index()
            .rename(columns={'correct_decision_average': 'participant_level_accuracy'})
        )
    else:
        raise ValueError(f"Participant-level accuracy computation is not implemented for the {dataset} dataset.")

    return metrics


def add_participant_level_accuracy_to_metrics(metrics, sample_predictions, dataset):
    if dataset in ["CAUEEG", "TUAB"]:
        participant_level_accuracy = compute_participant_level_accuracy(sample_predictions, dataset)
        return metrics.merge(participant_level_accuracy, on=['dataset', 'task', 'model', 'n_participants', 'n_segments', 'seed'])
    else:
        print(f"Participant-level accuracy computation is not implemented for the {dataset} dataset. Returning the original metrics.")
        return metrics


def build_train_val_metric_tracker(task, n_batches_per_epoch, device="cpu", wandb=None):
    if task not in TASK_METRICS:
        raise ValueError(f"Task {task} not recognized.")
    return Tracker(TASK_METRICS[task], n_batches_per_epoch=n_batches_per_epoch, device=device, wandb=wandb)
    

def build_test_metric_tracker(task, device="cpu"):
    if task not in TASK_METRICS:
        raise ValueError(f"Task {task} not recognized.")
    return build_metrics(TASK_METRICS[task], device=device)