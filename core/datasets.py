import logging
import warnings
import shutil
import subprocess
import json

from typing import Any, Callable
from warnings import warn
from pathlib import Path
from copy import deepcopy

import torch
import pandas as pd
import numpy as np

import mne
from mne import make_fixed_length_epochs
from mne_bids import find_matching_paths, read_raw_bids, get_bids_path_from_fname

from torcheeg import transforms
from torcheeg.datasets import CSVFolderDataset

from core.user_config import user_config
from core.tasks import TASK_LABELS, LABEL_MAPPINGS
from core import augmentations


AUGMENTATION_DEFAULTS = {
    "AmplitudeScaling": {"AmplitudeScaling": {"min_scaling_factor": 0.5, "max_scaling_factor": 2}},
    "FrequencyShift": {"FrequencyShift": {"sfreq": 100, "max_shift": 0.3}},
    "PhaseRandomization": {"PhaseRandomization": {"max_phase_shift": 1, "channel_wise": True}},
    "PhaseRandomizationJoint": {"PhaseRandomization": {"max_phase_shift": 1, "channel_wise": False}},
}


def map_epochs_to_layout(epochs: mne.Epochs, layout: list[str]) -> mne.Epochs:
    """
    Maps the channels of an epochs EEG object to a specified layout, i.e.
    a specified list of channels.

    The output epochs object will have the channels in the order specified in
    layout. If a channel in layout is not present in epochs, it will be added
    as a dummy channel filled with zeros.

    Example:
    epochs = mne.read_epochs("path/to/epochs.fif")
    layout = ["C3", "C4", "F3", "F4", "O1", "O2"]
    epochs = map_epochs_to_layout(epochs, layout)
    """
        
    # Create an empty array to store the data in the new order
    data = np.zeros((epochs.get_data().shape[0], len(layout), epochs.get_data().shape[2]))

    # Iterate over the new layout and fill the data array
    for i, ch_name in enumerate(layout):
        if ch_name in epochs.ch_names:
            data[:, i, :] = epochs.get_data(picks=ch_name).squeeze()
        else:
            warn(f"Channel {ch_name} not found in epochs, adding zeros")

    # Create a new mne.EpochsArray object with the new layout
    info = mne.create_info(layout, epochs.info["sfreq"], ch_types="eeg")
    events = np.copy(epochs.events)
    epochs = mne.EpochsArray(data, info, events=events, event_id=epochs.event_id)

    return epochs
    

def epochs_from_raw_eeg(
    file_path: str | Path,  # path to the raw EEG file
    epoch_duration: float | None = 1,  # in seconds
    epoch_overlap: float | None = 0,  # in seconds
    mne_raw_transforms: list[dict[str, dict[str, Any]]] | None = None,
    mne_epochs_transforms: list[dict[str, dict[str, Any]]] | None = None,
    layout: list[str] | None = None,  # list of channel names
    verbose: bool = False,
    **kwargs
):
    """
    Load raw EEG data and create epochs (mne.Epochs) from it.
    mne_raw_transforms and mne_epochs_transforms can be used to specify processing
    steps using mne.io.Raw and mne.Epochs instance methods. They should be specified
    as dicts, where the keys are the method names and the values are dictionaries of
    keyword arguments to pass to the method.

    The `layout` argument can be used to specify a specific list of channels to extract from
    the raw data and their ordering. If a channel in layout is not present in
    the raw data, it will be added as a dummy channel filled with zeros. This is useful
    to ensure a consistent channel layout across recordings or datasets.

    Example:
    mne_raw_transforms = {
        "filter": {"l_freq": 0.5, "h_freq": 40},
        "resample": {"sfreq": 100},
    }
    
    epochs = epochs_from_raw_eeg(
        "path/to/raw.eeg",
        epoch_duration=5,
        epoch_overlap=2.5,
        mne_raw_transforms=mne_raw_transforms,
    )
    """
    try:
        # Load raw data
        raw = read_raw_bids(get_bids_path_from_fname(file_path), verbose=verbose).load_data()

        # Apply raw transforms
        if mne_raw_transforms is not None:
            for method_name, args in mne_raw_transforms.items():
                if hasattr(raw, method_name):
                    method = getattr(raw, method_name)
                    raw = method(**args)
                else:
                    raise ValueError(f"Unknown mne_raw_transform {method_name}")

        # Create epochs
        try:
            epochs = make_fixed_length_epochs(
                raw, duration=epoch_duration, overlap=epoch_overlap,
                reject_by_annotation=False, verbose=verbose
            )
        except ValueError as e:
            if "No events produced" in str(e):
                warnings.warn(f"No events produced for {file_path}, returning None.", RuntimeWarning)
                return None
            else:
                raise

        # Apply epochs transforms
        if mne_epochs_transforms is not None:
            for method_name, args in mne_epochs_transforms.items():
                if hasattr(epochs, method_name):
                    method = getattr(epochs, method_name)
                    epochs = method(**args)
                else:
                    raise ValueError(f"Unknown mne_epochs_transform {method_name}")

        if layout is not None:
            epochs = map_epochs_to_layout(epochs, layout)

        return epochs

    except Exception as e:
        # Add file_path to error message
        raise type(e)(f"Error processing {file_path}: {e}").with_traceback(e.__traceback__)
    

class BIDSDataset(CSVFolderDataset):
    """
    Dataset class for BIDS-compliant datasets.

    This class is a subclass of the `torcheeg.datsets.CSVFolderDataset` class that
    provides a more convenient constructor for data in BIDS format and advanced
    preprocessing options from mne-python. It builds on its parent class' functionality
    to restore directly from existing data in the io_path directory but it will first
    verify if the metadata of the existing data matches the parameters provided to the
    constructor.

    Rather than requiring the user to create a CSV file with metadata for all files,
    this class allows the user to specify the BIDS root directory along with optional
    specifications of bids entities to filter the data.

    The additional arguments raw_transforms and epochs_transforms allow the user to
    specify `mne.io.Raw` and `mne.Epochs` instance methods to be applied.
    """

    def __init__(
        self,
        bids_root: str | Path,
        entity_selection: dict | None = None,
        epoch_duration: float | None = 1,  # in seconds
        epoch_overlap: float | None = 0,  # in seconds
        mne_raw_transforms: dict[str, dict[str, Any]] | None = None,
        mne_epochs_transforms: dict[str, dict[str, Any]] | None = None,
        scaling_factor: float = 1e6,  # scaling factor for EEG data, defaults to 1e6 to convert from V to uV
        layout: list[str] | None = None,  # list of channel names
        online_transform: Callable | dict[str, dict[str, Any]] | None = None,
        target: str | list | None = None,  # specify a column in the participants.tsv or *_scans.tsv file as the target
        target_mapping: dict[str, Any] | None = None,  # mapping for target values, e.g. {"control": 0, "patient": 1}
        target_dtype: str = "float32",  # data type for the targets, supports all torch data types
        io_path: str | None = None,
        io_size: int = 1048576,
        io_mode: str = 'lmdb',
        num_worker: int = 0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Dataset class for BIDS-compliant datasets. Creates a dataset from the EEG recordings
        in the specified `bids_root` directory. The `entity_selection` argument allows the user
        to only select a subset of the data based on the BIDS entities (e.g. subjects, sessions,
        runs, etc.). The `entity_selection` arguments are passed to the `mne_bids.find_matching_paths`
        function, so the same arguments are supported.

        A folder is created in the `io_path` directory to store the processed dataset in an efficient
        format for fast loading. If `io_path` is not specified, a random directory will be created.
        If the `io_path` already exists, the dataset will attempt to restore directly from the existing
        data (after verification of its metadata to ensure that it matches the parameters provided to the
        constructor).

        The additional arguments raw_transforms and epochs_transforms allow the user to
        specify `mne.io.Raw` and `mne.Epochs` instance methods to be applied.
        These are specified as dicts, where the key is the method name and the values are
        dictionaries of keyword arguments to pass to the method.

        Example:
        ```Python
        mne_raw_transforms = {
            "filter": {"l_freq": 0.5, "h_freq": 40},
            "resample": {"sfreq": 100},
        }

        dataset = BIDSDataset(
            bids_root="path/to/bids",
            entity_selection={"subjects": ["01", "02"], "sessions": "01"},
            epoch_duration=2,
            epoch_overlap=1,
            mne_raw_transforms=mne_raw_transforms,
            io_path="outputs/"
        )
        ```

        """
        # Try to get git commit id
        try:
            git_commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        except:
            git_commit_id = None

        self.metadata = {
            "bids_root": str(bids_root),
            "entity_selection": deepcopy(entity_selection),
            "epoch_duration": epoch_duration,
            "epoch_overlap": epoch_overlap,
            "mne_raw_transforms": deepcopy(mne_raw_transforms),
            "mne_epochs_transforms": deepcopy(mne_epochs_transforms),
            "scaling_factor": scaling_factor,
            "layout": layout,
            "git_commit_id": git_commit_id,
            **kwargs
        }

        if io_path is not None:
            io_path = Path(io_path)

        if not (io_path is not None and io_path.exists() and (io_path / "data.csv").exists()):
            # Add defaults if entity selection is not specified
            if entity_selection is None:
                entity_selection = {}
            if "extensions" not in entity_selection:
                entity_selection["extensions"] = [".vhdr", ".edf", ".bdf", ".set"]
            if "datatypes" not in entity_selection:
                entity_selection["datatypes"] = "eeg"
            if "suffixes" not in entity_selection:
                entity_selection["suffixes"] = "eeg"

            # Find matching bids paths
            bids_root = Path(bids_root)
            bids_paths = find_matching_paths(bids_root, **entity_selection)

            # Hot fix for potential duplicates due mne-bids bug: https://github.com/mne-tools/mne-bids/issues/1127
            bids_paths = list({bids_path.fpath: bids_path for bids_path in bids_paths}.values())

            # Read participants.tsv
            participants_tsv = pd.read_csv(bids_root / "participants.tsv", sep="\t")

            # Create csv
            data = []
            for bp in bids_paths:
                # Construct trial_id from session and run
                trial_id = bp.session if bp.session is not None else "0"
                if bp.run is not None:
                    trial_id += f"_{bp.run}"

                # Extract subject info from participants.tsv
                subject_info = participants_tsv[participants_tsv['participant_id'] == "sub-" + bp.subject].to_dict('records')[0]

                # Extract recording info from scans.tsv
                if bp.session is not None:
                    scants_tsv_path = bp.root / f"sub-{bp.subject}" / f"ses-{bp.session}" / f"sub-{bp.subject}_ses-{bp.session}_scans.tsv"
                else:
                    scants_tsv_path = bp.root / f"sub-{bp.subject}" / f"sub-{bp.subject}_scans.tsv"

                if scants_tsv_path.exists():
                    scans_tsv = pd.read_csv(scants_tsv_path, sep="\t")
                    recording_info = scans_tsv[scans_tsv['filename'] == f"eeg/{bp.basename}"].to_dict('records')[0]
                else:
                    recording_info = {}

                data.append(
                    dict(
                        subject = bp.subject,
                        trial_id = trial_id,
                        file_path = bp.fpath,
                        **subject_info,
                        **recording_info
                    )
                )

            df = pd.DataFrame(data)

            # Save csv temporarily for initialization of CSVFolderDataset.
            # We can only move it to the io_path after the super().__init__ call
            # because the CSVFolderDataset constructor uses the io_path to check
            # if the dataset already exists.
            csv_path = Path("temp_data.csv")
            df.to_csv(csv_path, index=False)

        else:  # Dataset already exists
            # Verify if metadata is compatible
            self.load_and_verfiy_metadata(io_path, self.metadata)

            # Restore existing dataset
            logging.info(f"Found existing dataset at {io_path}. Restoring dataset...")
            csv_path = io_path / "data.csv"

        offline_transform = transforms.Compose([
            transforms.Lambda(
                targets=["eeg"],
                lambd=lambda eeg: torch.tensor(eeg.squeeze(), dtype=torch.float32)
            ),
            transforms.Lambda(
                targets=["eeg"],
                lambd=lambda eeg: eeg * scaling_factor
            ),
        ])

        label_transform = [transforms.Select(key=target)]
        if target_mapping is not None:
            label_transform.append(transforms.Mapping(map_dict=target_mapping))
        if target_dtype is not None:
            dtype = getattr(torch, target_dtype)
            label_transform.append(transforms.Lambda(
                targets=["y"],
                lambd=lambda x: torch.tensor(x, dtype=dtype)
            ))
        label_transform = transforms.Compose(label_transform)

        if online_transform is not None and not callable(online_transform):
            online_transforms_parsed = []
            for tf, args in online_transform.items():
                if hasattr(transforms, tf):
                    online_transforms_parsed.append(getattr(transforms, tf)(**args))
                elif hasattr(augmentations, tf):
                    online_transforms_parsed.append(getattr(augmentations, tf)(**args))
                else:
                    raise ValueError(f"Unknown transform {tf}")
            online_transform = transforms.Compose(online_transforms_parsed)

        super().__init__(
            csv_path=csv_path,
            read_fn=epochs_from_raw_eeg,
            epoch_duration=epoch_duration,
            epoch_overlap=epoch_overlap,
            mne_raw_transforms=mne_raw_transforms,
            mne_epochs_transforms=mne_epochs_transforms,
            online_transform=online_transform,
            offline_transform=offline_transform,
            label_transform=label_transform,
            io_path=io_path,
            io_size=io_size,
            io_mode=io_mode,
            num_worker=num_worker,
            verbose=verbose,
            before_trial=locals().get("before_trial", None),  # hot fix for torcheeg bug
            layout=layout,
            **kwargs
        )

        # Remove samples with NaN values for the target to deal with missing labels
        if self.info[target].isnull().any().any():
            warn("Dataset contains samples with missing labels. These samples will be dropped.")
            self.info.dropna(subset=target, inplace=True)

        # Move csv and save metadata if dataset was newly created
        if str(csv_path) == "temp_data.csv":
            # Move csv file to io_path
            shutil.move(csv_path, io_path / "data.csv")

            # Save metadata
            with open(io_path / 'dataset_metadata.json', 'w') as f:
                json.dump(self.metadata, f)
            

    @staticmethod
    def load_and_verfiy_metadata(io_path, metadata):
        try:
            with open(io_path / 'dataset_metadata.json', 'r') as f:
                old_metadata = json.load(f)

            # Remove git_commit_id from metadata since we don't require it to match
            old_metadata.pop("git_commit_id", None)
            metadata.pop("git_commit_id", None)

            # Check if metadata is compatible
            if old_metadata != metadata:
                raise ValueError(
                    "The metadata of the existing dataset does not match the metadata of the new dataset. "
                    "Please specify a different io_path or remove the existing dataset."
                    f"Existing metadata: {old_metadata}"
                    f"New metadata: {metadata}"
                )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No metadata file found at {io_path}. Please specify a different io_path "
                + "or remove the existing dataset to create it from scratch.")
        
    
    @staticmethod
    def process_record(
        file: Any = None,
        offline_transform: None | Callable = None,
        read_fn: None | Callable = None,
        **kwargs
    ):
        """
        Minor adjustment to the process_record method of the CSVFolderDataset class
        to deal with files that contain no or only a single epoch.
        """

        trial_info = file
        file_path = trial_info['file_path']

        trial_samples = read_fn(file_path, **kwargs)
        if trial_samples is not None and len(trial_samples.events):
            events = [i[0] for i in trial_samples.events]
            events.append(events[-1] + trial_samples.times.size)

            write_pointer = 0
            for i, trial_signal in enumerate(trial_samples.get_data()):
                t_eeg = trial_signal
                if not offline_transform is None:
                    t = offline_transform(eeg=trial_signal)
                    t_eeg = t['eeg']

                clip_id = f'{file_path}_{write_pointer}'
                write_pointer += 1

                record_info = {
                    **trial_info, 'start_at': events[i],
                    'end_at': events[i + 1],
                    'clip_id': clip_id
                }

                yield {'eeg': t_eeg, 'key': clip_id, 'info': record_info}
        else:  # file contains no samples (can happen e.g. if file is too short)
            # Return an empty iterator
            return iter([])
    

def save_splits(
    io_path: str | Path,  # path to the io_path of the dataset
    split_metadata: dict,  # metadata for the split
    train_info: pd.DataFrame | None = None,  # dataframe with train split info
    val_info: pd.DataFrame | None = None,  # dataframe with val split info
    test_info: pd.DataFrame | None = None,  # dataframe with test split info
):
    """
    Save split metadata and info to the io_path of the dataset.

    Dataset info files (train.csv, val.csv, test.csv) are saved to the io_path along with
    a split_metadata.json file.

    Example:
    ```Python
    split_metadata = {
        "split_type": "train_val_test",
        "train_size": 0.8,
        "val_size": 0.1,
        "test_size": 0.1,
    }
    save_splits(
        io_path="/dataset/io_path/",
        split_metadata=split_metadata,
        train_info=train_info,
        val_info=val_info,
        test_info=test_info,
    )
    ```
    """
    io_path = Path(io_path)

    if not io_path.exists():
        raise FileNotFoundError(f"Directory {io_path} does not exist.")

    if train_info is not None:
        train_info.to_csv(io_path / "train.csv", index=False)
    if val_info is not None:
        val_info.to_csv(io_path / "val.csv", index=False)
    if test_info is not None:
        test_info.to_csv(io_path / "test.csv", index=False)

    with open(io_path / "split_metadata.json", "w") as f:
        json.dump(split_metadata, f)    


def load_and_verify_split_metadata(
    io_path: str | Path,  # path from which to load the split metadata
    metadata: dict  # metadata to verify against
):
    """Load split metadata from io_path and verify that it matches the provided metadata."""
    io_path = Path(io_path)

    try:
        with open(io_path / 'split_metadata.json', 'r') as f:
            old_metadata = json.load(f)

        # Check if metadata is compatible
        if old_metadata != metadata:
            raise ValueError(
                f"The existing split metadata in {io_path} does not match the split parameters that you specified. "
                "Call the splitting function without io_path to ignore the existing split or remove the existing "
                "split_metadata to create and save a new split."
            )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No split metadata file found at {io_path}. Please specify a different io_path "
            + "or remove the existing dataset to create it from scratch.")


def load_existing_splits(
    io_path: str | Path,  # path from which to load the splits
    verification_metadata: dict = None  # metadata to verify against
):
    """Load existing splits and (optionally) verify the metadata against provided metadata."""
    io_path = Path(io_path)

    if verification_metadata is not None:
        load_and_verify_split_metadata(io_path, verification_metadata)
    
    train_info = None
    val_info = None
    test_info = None

    if (io_path / "train.csv").exists():
        train_info = pd.read_csv(io_path / "train.csv")
    if (io_path / "val.csv").exists():
        val_info = pd.read_csv(io_path / "val.csv")
    if (io_path / "test.csv").exists():
        test_info = pd.read_csv(io_path / "test.csv")

    return train_info, val_info, test_info


def split_by_info_values(
    dataset,
    info_key: str,  # key / column name in the dataset info to split by
    train_values: list = [],  # values for samples to include in the training set
    val_values: list = [],  # values for samples to include in the validation set
    test_values: list = [],  # values for samples to include in the test set
    io_path: str | Path = None,  # if specified, splits will be restored (if available) or saved from/to this path
):
    """
    Splits a dataset into a training, a validation, and a test dataset
    based on a key (column name) in the dataset info and corresponding
    values for the different splits.
    """

    split_metadata = {
        "info_key": info_key,
        "train_values": train_values,
        "val_values": val_values,
        "test_values": test_values,
    }

    io_path = Path(io_path) if io_path is not None else None
    if io_path is not None and (io_path / "split_metadata.json").exists():
        train_info, val_info, test_info = load_existing_splits(io_path, split_metadata)
        logging.info(f"Restored existing splits from {io_path}.")
    else:
        dataset_info = dataset.info
        train_info = dataset_info[dataset_info[info_key].isin(train_values)]
        val_info = dataset_info[dataset_info[info_key].isin(val_values)]
        test_info = dataset_info[dataset_info[info_key].isin(test_values)]

        if io_path is not None:
            save_splits(
                io_path=io_path,
                split_metadata=split_metadata,
                train_info=train_info,
                val_info=val_info,
                test_info=test_info,
            )

    if train_info is None or len(train_info) == 0:
        train_dataset = None
    else:
        # Note: we need to use merge to respect potentially dropped samples when loading existing splits.
        # Samples may be dropped if the target contains NaN values.
        train_dataset = deepcopy(dataset)
        train_dataset.info = train_dataset.info.merge(train_info, how="inner")

    if val_info is None or len(val_info) == 0:
        val_dataset = None
    else:
        # Set new dataset info
        # Note: we need to use merge to respect potentially dropped samples when loading existing splits.
        # Samples may be dropped if the target contains NaN values.
        val_dataset = deepcopy(dataset)
        val_dataset.info = val_dataset.info.merge(val_info, how="inner")

    if test_info is None or len(test_info) == 0:
        test_dataset = None
    else:
        # Set new dataset info
        # Note: we need to use merge to respect potentially dropped samples when loading existing splits.
        # Samples may be dropped if the target contains NaN values.
        test_dataset = deepcopy(dataset)
        test_dataset.info = test_dataset.info.merge(test_info, how="inner")

    return train_dataset, val_dataset, test_dataset


def build_dataset(
    config: dict,
    split: bool = False  # if True, the dataset is split into train, val and test sets
):
    if isinstance(config["augmentation"], str):
        augmentation = AUGMENTATION_DEFAULTS.get(config["augmentation"], None)
    else:
        augmentation = config["augmentation"]
    
    if config["dataset"] == "TUAB" and config["task"] == "normality":
        if config["model"] == "LaBraM":
            dataset = BIDSDataset(
                bids_root=user_config["data"]["TUAB"]["bids_root"],
                io_path=str(Path(user_config["data"]["TUAB"]["deriv_root"]) / "lmdb_crop-15m_epochs-2s_labram_compatible"),
                scaling_factor=1e4,
                target=TASK_LABELS[config["task"]],
                target_mapping=LABEL_MAPPINGS.get(config["task"], None),
                epoch_duration=2,
                mne_raw_transforms={
                    "filter": {"l_freq": 0.1, "h_freq": 75},
                    "notch_filter": {"freqs": [50]},
                    "resample": {"sfreq": 200},
                    "crop": {"tmin": 30, "tmax": 930},
                    "pick": {"picks": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]},
                    "set_eeg_reference": {"ref_channels": "average"}
                },
                online_transform=augmentation,
            )
            if split:
                dataset = split_by_info_values(
                    dataset,
                    info_key="train_val_test_split",
                    train_values=["train"],
                    val_values=["val"],
                    test_values=["test"],
                    io_path=str(Path(user_config["data"][config["dataset"]]["deriv_root"]) / "lmdb_crop-15m_epochs-2s_labram_compatible")
                )
        else:
            dataset = BIDSDataset(
                bids_root=user_config["data"]["TUAB"]["bids_root"],
                io_path=str(Path(user_config["data"]["TUAB"]["deriv_root"]) / "lmdb_layout-1020_filtered-1to50Hz_crop-15m_epochs-2s"),
                target=TASK_LABELS[config["task"]],
                target_mapping=LABEL_MAPPINGS.get(config["task"], None),
                epoch_duration=2,
                mne_raw_transforms={
                    "filter": {"l_freq": 1, "h_freq": 50},
                    "resample": {"sfreq": 100},
                    "crop": {"tmin": 30, "tmax": 930},
                    "pick": {"picks": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]},
                    "set_eeg_reference": {"ref_channels": "average"}
                },
                online_transform=augmentation,
            )
            if split:
                dataset = split_by_info_values(
                    dataset,
                    info_key="train_val_test_split",
                    train_values=["train"],
                    val_values=["val"],
                    test_values=["test"],
                    io_path=str(Path(user_config["data"][config["dataset"]]["deriv_root"]) / "lmdb_layout-1020_filtered-1to50Hz_crop-15m_epochs-2s")
                )
    elif config["dataset"] == "PhysioNet" and config["task"] == "sleep_stage":
        if config["model"] == "LaBraM" or config["experiment_name"] == "LaBraM-Preproc":
            dataset = BIDSDataset(
                bids_root=user_config["data"]["PhysioNet"]["bids_root"],
                io_path=str(Path(user_config["data"]["PhysioNet"]["deriv_root"]) / "pickle_sleep_stages_eeg_epochs-15s_labram_compatible"),
                io_mode="pickle",
                scaling_factor=1e4,
                target=TASK_LABELS[config["task"]],
                target_mapping=LABEL_MAPPINGS.get(config["task"], None),
                target_dtype="int64",
                epoch_duration=15,
                mne_raw_transforms={
                    "filter": {"l_freq": 0.1, "h_freq": 75},
                    "notch_filter": {"freqs": [50]},
                    "resample": {"sfreq": 200},
                    "pick": {"picks": ["F3", "F4", "C3", "C4", "O1", "O2"]},
                },
                online_transform=augmentation,
            )
            if split:
                dataset = split_by_info_values(
                    dataset,
                    info_key="labeled_train_val_test_split",
                    train_values=["train"],
                    val_values=["val"],
                    test_values=["test"],
                    io_path=str(Path(user_config["data"][config["dataset"]]["deriv_root"]) / "pickle_sleep_stages_eeg_epochs-15s_labram_compatible")
                )
        else:
            dataset = BIDSDataset(
                bids_root=user_config["data"]["PhysioNet"]["bids_root"],
                io_path=str(Path(user_config["data"]["PhysioNet"]["deriv_root"]) / "pickle_sleep_stages_eeg_filtered-1to50Hz_epochs-30s"),
                io_mode="pickle",
                target=TASK_LABELS[config["task"]],
                target_mapping=LABEL_MAPPINGS.get(config["task"], None),
                target_dtype="int64",
                epoch_duration=30,
                mne_raw_transforms={
                    "filter": {"l_freq": 1, "h_freq": 50},
                    "resample": {"sfreq": 100},
                    "pick": {"picks": ["F3", "F4", "C3", "C4", "O1", "O2"]},
                    "set_eeg_reference": {"ref_channels": "average"}
                },
                online_transform=augmentation,
            )
            if split:
                dataset = split_by_info_values(
                    dataset,
                    info_key="labeled_train_val_test_split",
                    train_values=["train"],
                    val_values=["val"],
                    test_values=["test"],
                    io_path=str(Path(user_config["data"][config["dataset"]]["deriv_root"]) / "pickle_sleep_stages_eeg_filtered-1to50Hz_epochs-30s")
                )
    elif config["dataset"] == "CAUEEG" and config["task"] == "dementia":
        if config["model"] == "LaBraM":
            dataset = BIDSDataset(
                bids_root=user_config["data"]["CAUEEG"]["bids_root"],
                io_path=str(Path(user_config["data"]["CAUEEG"]["deriv_root"]) / "lmdb_dementia_crop-5m_epochs-2s_labram_compatible"),
                scaling_factor=1e4,
                target=TASK_LABELS[config["task"]],
                target_mapping=LABEL_MAPPINGS.get(config["task"], None),
                target_dtype="int64",
                epoch_duration=2,
                mne_raw_transforms={
                    "filter": {"l_freq": 0.1, "h_freq": 75},
                    "notch_filter": {"freqs": [50]},
                    "resample": {"sfreq": 200},
                    "crop": {"tmin": 0, "tmax": 300},
                    "pick": {"picks": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]},
                    "set_eeg_reference": {"ref_channels": "average"}
                },
                online_transform=augmentation,
            )
            if split:
                dataset = split_by_info_values(
                    dataset,
                    info_key="dementia_split_no_overlap",
                    train_values=["train"],
                    val_values=["val"],
                    test_values=["test"],
                    io_path=str(Path(user_config["data"][config["dataset"]]["deriv_root"]) / "lmdb_dementia_crop-5m_epochs-2s_labram_compatible")
                )
        else:
            dataset = BIDSDataset(
                bids_root=user_config["data"]["CAUEEG"]["bids_root"],
                io_path=str(Path(user_config["data"]["CAUEEG"]["deriv_root"]) / "lmdb_dementia_layout-1020_filtered-1to50Hz_crop-5m_epochs-2s"),
                target=TASK_LABELS[config["task"]],
                target_mapping=LABEL_MAPPINGS.get(config["task"], None),
                target_dtype="int64",
                epoch_duration=2,
                mne_raw_transforms={
                    "filter": {"l_freq": 1, "h_freq": 50},
                    "resample": {"sfreq": 100},
                    "crop": {"tmin": 0, "tmax": 300},
                    "pick": {"picks": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]},
                    "set_eeg_reference": {"ref_channels": "average"}
                },
                online_transform=augmentation,
            )
            if split:
                dataset = split_by_info_values(
                    dataset,
                    info_key="dementia_split_no_overlap",
                    train_values=["train"],
                    val_values=["val"],
                    test_values=["test"],
                    io_path=str(Path(user_config["data"][config["dataset"]]["deriv_root"]) / "lmdb_dementia_layout-1020_filtered-1to50Hz_crop-5m_epochs-2s")
                )

    return dataset


def sample_subset(dataset, n_participants, n_segments, seed):
    """Sample a subset of the dataset with a fixed number of participants and segments."""

    dataset = deepcopy(dataset)

    # Sample participants
    rng = np.random.default_rng(seed)
    sampled_subjects = rng.choice(dataset.info['subject'].unique(), size=n_participants, replace=False)
    dataset.info = dataset.info[dataset.info['subject'].isin(sampled_subjects)]

    # Sample segments
    sampled_segments = dataset.info.groupby('subject')["clip_id"].apply(
        lambda x: x.sample(n=n_segments, replace=False, random_state=seed)
    )
    dataset.info = dataset.info[dataset.info["clip_id"].isin(sampled_segments)]

    return dataset