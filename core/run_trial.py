import argparse
import pandas as pd

from math import ceil
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from core.datasets import build_dataset, sample_subset
from core.models import build_model
from core.metrics import get_class_weights, build_loss_function, build_train_val_metric_tracker, build_test_metric_tracker, add_participant_level_accuracy_to_metrics
from core.utils import construct_output_dir
from core.user_config import user_config


def run_trial(config, output_root=None, use_wandb=False):
    """Run a single trial including model training and evaluation on the test set."""

    print(f"Running trial with the following config:\n{config}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize datasets & dataloaders
    dataset_train, dataset_val, dataset_test = build_dataset(config, split=True)
    dataset_train = sample_subset(
        dataset=dataset_train,
        n_participants=config["n_participants"],
        n_segments=config["n_segments"],
        seed=config["seed"]
    )

    dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=config["batch_size"], num_workers=2, pin_memory=True)
    dataloader_val = DataLoader(dataset_val, shuffle=False, batch_size=config["batch_size"], num_workers=2, pin_memory=True)
    dataloader_test = DataLoader(dataset_test, shuffle=False, batch_size=config["batch_size"], num_workers=2, pin_memory=True)

    # Initialize model
    model = build_model(config, device=device)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 1e-3))

    # Initialize loss function
    class_weights = get_class_weights(dataset=dataset_train, task=config["task"], device=device)
    loss_function = build_loss_function(config["task"], weight=class_weights)

    # Initialize train and validation metric tracker
    if use_wandb:
        import wandb
        output_dir = construct_output_dir(output_root, config)
        output_dir.mkdir(parents=True, exist_ok=True)
        wandb_run = wandb.init(
            config=config,
            project="participant-diversity-paper",
            group="-".join([config["dataset"], config["task"], config["model"]]),
            dir=output_dir,
            mode="offline",
        )
        tracker = build_train_val_metric_tracker(
            config["task"], n_batches_per_epoch=len(dataloader_train), device=device, wandb=wandb_run
        )
    else:
        tracker = build_train_val_metric_tracker(
            config["task"], n_batches_per_epoch=len(dataloader_train), device=device
        )
    
    # Train
    global_batch = 0
    early_stopping_triggered = False
    best_model_state = None
    model.train()
    batches_per_epoch = ceil(len(dataset_train) / config["batch_size"])
    max_epochs = ceil(config["max_batches"] / batches_per_epoch)
    for epoch in tqdm(range(max_epochs), position=0, desc="Epochs ", leave=True):
        if early_stopping_triggered:
            break

        for batch, (X, y) in tqdm(enumerate(dataloader_train), position=1, desc="Batches", leave=True):
            if global_batch >= config["max_batches"]:
                break

            X, y = X.to(device), y.to(device)
            out = model(X)
            
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients
            optimizer.step()

            tracker.report_train_step(pred=out.detach(), target=y, loss=loss.item(), epoch=epoch, batch=batch)
            global_batch += 1

            # Evaluate model
            if global_batch % config["evaluation_interval"] == 0:
                model.eval()
                with torch.no_grad():
                    for X, y in dataloader_val:
                        X, y = X.to(device), y.to(device)
                        out = model(X)
                        loss = loss_function(out, y)
                        tracker.report_val_step(pred=out, target=y, loss=loss.item())
                tracker.report_val_done()
                model.train()

                # Early stopping
                if config["early_stopping_patience"] is not None:
                    if tracker.evaluations_since_improvement == 0:
                        # Backup model state
                        best_model_state = model.state_dict()
                    elif tracker.evaluations_since_improvement >= config["early_stopping_patience"]:
                        print(f"Early stopping after {global_batch} batches.")
                        early_stopping_triggered = True
                        break

    if use_wandb:
        wandb_run.finish()

    # Evaluate on test set    
    test_metrics = build_test_metric_tracker(config["task"], device)
    sample_info = dataset_test.info.copy()
    target = config["task"]

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        for batch, (X, y) in tqdm(enumerate(dataloader_test), desc="Batches", leave=True):
            X, y = X.to(device), y.to(device)
            out = model(X)
            test_metrics.update(out, y)

            # Log sample predictions
            istart = batch * config["batch_size"]
            istop = istart + len(out) - 1
            if out.dim() == 2 and out.shape[1] > 1:  # Multiple outputs (e.g. multi-class classification)
                for i in range(out.shape[1]):
                    sample_info.loc[istart:istop, f"predicted_{target}_{i}"] = out[:, i].cpu().numpy()
            else:
                sample_info.loc[istart:istop, f"predicted_{target}"] = out.cpu().numpy()

    info = {
        "dataset": config["dataset"],
        "task": config["task"],
        "model": config["model"],
        "n_participants": config["n_participants"],
        "n_segments": config["n_segments"],
        "seed": config["seed"],
        "n_batches_trained": global_batch - (early_stopping_triggered * config["evaluation_interval"] * config["early_stopping_patience"]),
        "augmentation": config["augmentation"],
    }

    metrics_dict = test_metrics.compute_all()
    metrics_dict = {k: [v.tolist() for v in value] for k, value in metrics_dict.items()}
    metrics_df = pd.DataFrame(metrics_dict).assign(**info)
    samples_df = sample_info.assign(**info)

    if config["dataset"] in ["TUAB", "CAUEEG"]:
        metrics_df = add_participant_level_accuracy_to_metrics(metrics_df, samples_df, config["dataset"])
    
    # Save metrics and sample predictions
    if output_root is not None:
        output_dir = construct_output_dir(output_root, config)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(output_dir / "metrics.csv")
        samples_df.to_csv(output_dir / "sample_predictions.csv")

    return metrics_df, samples_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trial with specified configuration.")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to use.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to use.")
    parser.add_argument("--task", type=str, required=True, help="Task to perform.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--max_batches", type=int, default=50000, help="Maximum number of training batches.")
    parser.add_argument("--evaluation_interval", type=int, default=500, help="Interval for evaluation in batches.")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience in evaluations.")
    parser.add_argument("--augmentation", type=str, default=None, help="Augmentation to use.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a model checkpoint.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--n_participants", type=int, required=True, help="Number of participants.")
    parser.add_argument("--n_segments", type=int, required=True, help="Number of segments.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data subsampling.")
    parser.add_argument("--experiment_name", type=str, default="baseline", help="Name of the experiment.")
    parser.add_argument("--output_root", type=str, default=user_config["output_root"], help="Output directory. The configuration in user_config.yml will be used by default.")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights and Biases for tracking.")
    args = parser.parse_args()

    # Convert the string "None" to the actual None type
    if args.augmentation == "None":
        args.augmentation = None

    if args.checkpoint == "None":
        args.checkpoint = None

    config = dict(
        model=args.model,
        dataset=args.dataset,
        task=args.task,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        evaluation_interval=args.evaluation_interval,
        early_stopping_patience=args.early_stopping_patience,
        augmentation=args.augmentation,
        checkpoint=args.checkpoint,
        learning_rate=args.learning_rate,
        n_participants=args.n_participants,
        n_segments=args.n_segments,
        seed=args.seed,
        experiment_name=args.experiment_name,
    )

    run_trial(config, output_root=args.output_root, use_wandb=args.use_wandb)