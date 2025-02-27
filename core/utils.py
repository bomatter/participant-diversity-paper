import subprocess

from pathlib import Path


def construct_output_dir(output_root, config):
    output_dir = (
        Path(output_root)
        / config["experiment_name"] / config["dataset"] / config["task"] / config["model"]
        / f"{config['n_participants']}_participants-{config['n_segments']}_segments"
        / f"seed_{config['seed']}"
    )
    return output_dir


def ensure_list(*args):
    """
    Ensures that each argument is a list. Arguments that are not already lists are wrapped in a list.
    """
    converted_args = [arg if isinstance(arg, list) else [arg] for arg in args]
    if len(converted_args) == 1:
        return converted_args[0]
    else:
        return converted_args


def create_trial_configs(
    dataset,
    task,
    model,
    batch_size=64,
    max_batches=50000,
    evaluation_interval=500,
    early_stopping_patience=5,
    augmentation=None,
    checkpoint=None,
    learning_rate=None,  # If None, the model-specific default learning rate is used
    n_participants=[25, 50, 100, 200, 400, 800, 1600],
    n_segments=[5, 10, 20, 40, 80, 160, 320],
    seed=[42, 43, 44, 45, 46],
    experiment_name="baseline",
):
    model = ensure_list(model)
    n_participants = ensure_list(n_participants)
    n_segments = ensure_list(n_segments)
    seed = ensure_list(seed)

    trial_configs = []
    for m in model:
        for np in n_participants:
            for ns in n_segments:
                for s in seed:

                    if learning_rate is None:
                        if m == "LaBraM":
                            if checkpoint is not None:  # Fine-tuning
                                if dataset == "TUAB":
                                    learning_rate = 1e-6
                                elif dataset == "CAUEEG":
                                    learning_rate = 5e-6
                                elif dataset == "PhysioNet":
                                    learning_rate = 1e-5
                                else:
                                    raise ValueError(f"Fine-tuning lr not set for dataset: {dataset}")
                            else:
                                if dataset == "TUAB":
                                    learning_rate = 1e-4
                                elif dataset == "CAUEEG":
                                    learning_rate = 1e-3
                                elif dataset == "PhysioNet":
                                    learning_rate = 1e-3
                                else:
                                    raise ValueError(f"Fine-tuning lr not set for dataset: {dataset}")
                        else:
                            learning_rate = 1e-3

                    trial_configs.append(
                        dict(
                            model=m,
                            dataset=dataset,
                            task=task,
                            batch_size=batch_size,
                            max_batches=max_batches,
                            evaluation_interval=evaluation_interval,
                            early_stopping_patience=early_stopping_patience,
                            augmentation=augmentation,
                            checkpoint=checkpoint,
                            learning_rate=learning_rate,
                            n_participants=np,
                            n_segments=ns,
                            seed=s,
                            experiment_name=experiment_name,
                        )
                    )
                    
    return trial_configs


def submit_slurm_job(
    command, 
    cpus=2,
    mem="16G",
    gpus=1, 
    time_limit="08:00:00",
    partition=None,
    job_name="job",
    log_dir="logs",
):
    """Convenience function to submit Slurm jobs."""
    sbatch_command = [
        "sbatch",
        f"--time={time_limit}",
        f"--cpus-per-task={cpus}",
        f"--mem={mem}",
        f"--gres=gpu:{gpus}",
        f"--job-name={job_name}",
        f"--output={log_dir}/{job_name}.log",
    ]
    
    if partition:
        sbatch_command.append(f"--partition={partition}")

    sbatch_command.extend(["--wrap", command])

    result = subprocess.run(sbatch_command, capture_output=True, text=True)
    
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]  # Extract job ID from sbatch output
        return job_id
    else:
        error_message = result.stderr.strip()
        raise RuntimeError(f"Error submitting job {job_name}: {error_message}")