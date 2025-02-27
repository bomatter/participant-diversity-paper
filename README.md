# Is Limited Participant Diversity Impeding EEG-based Machine Learning?

This repository contains the research code for the experiments in our paper *"Is Limited Participant Diversity Impeding EEG-based Machine Learning?"*



## Install

1. Create and activate conda/mamba environment

   ```
   mamba env create -f environment.yml
   mamba activate participant-diversity-paper
   ```

2. Install the package

   ```
   pip install -e .
   ```

3. Request access to the datasets and refer to the following repositories to harmonise and convert them to BIDS format.

   1. TUAB: https://github.com/bomatter/data-TUAB

   2. CAUEEG: https://github.com/bomatter/data-CAUEEG

   3. PhysioNet: https://github.com/bomatter/data-PhysioNet

4. Create a copy of the `user_config.example.yml` file and rename it to `user_config.yml`. Then open it and configure the paths to the dataset folders and the directory, where you want outputs to be saved.

5. To run experiments with the pretrained LaBraM model, download the labram-base.pth checkpoint to the root folder of this repository from the [official labram repo](https://github.com/935963004/LaBraM/tree/main/checkpoints).




## Run Experiments

Individual trials can be run with the `run_trial.py` script.

```
python core/run_trial.py \
    --model TCN --dataset TUAB --task normality \
    --n_participants 10 --n_segments 10 \
    --output_root results/
```

Alternatively, use the `run_experiments.ipynb` notebook to generate configurations (and submit slurm jobs) for all experiments to reproduce the paper results.