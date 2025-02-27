from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import braindecode

from core.labram import NeuralTransformer, load_state_dict, get_input_chans
from core.mAtt import E2R, SPDRectified, SPDTangentSpace, AttentionManifold


class TCN(braindecode.models.TCN):
    """TCN model with a mean pooling layer at the end."""
    
    def forward(self, x):
        x = super().forward(x)
        x = x.mean(dim=-1).squeeze(-1)
        return x
    

class mAtt(nn.Module):
    """
    A generic mAtt model where the number of EEG channels, outputs, and filters
    can be specified in the constructor.
    """

    def __init__(
        self,
        n_channels,
        n_outputs = 1,
        epochs = 5,
        n_kernels_conv1 = 100,
        n_kernels_conv2 = 50,
        kernel_size_conv2 = 11,
        embedding_dim = 25
    ):
        super().__init__()

        #FE
        self.conv1 = nn.Conv2d(1, n_kernels_conv1, (n_channels, 1))
        self.Bn1 = nn.BatchNorm2d(n_kernels_conv1)
        self.conv2 = nn.Conv2d(n_kernels_conv1, n_kernels_conv2, (1, kernel_size_conv2), padding="same")
        self.Bn2 = nn.BatchNorm2d(n_kernels_conv2)
        
        # E2R
        self.ract1 = E2R(epochs=epochs)
        # riemannian part
        self.att2 = AttentionManifold(n_kernels_conv2, embedding_dim)
        self.ract2  = SPDRectified()
        
        # R2E
        self.tangent = SPDTangentSpace(embedding_dim)
        self.flat = nn.Flatten()
        # fc
        self.linear = nn.Linear(((embedding_dim * (embedding_dim + 1)) // 2) * epochs, n_outputs, bias=True)

    def forward(self, x):
        """
        x: [batchsize x n_eeg_chnnels x n_timepoints] or [n_eeg_chnnels x n_timepoints]
        """
        
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # add batch and channel dimension
        elif x.dim() == 3:
            x = x.unsqueeze(1)  # add channel dimension

        x = self.conv1(x)
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)
        
        x = self.ract1(x)
        x, shape = self.att2(x)
        x = self.ract2(x)
        
        x = self.tangent(x)
        x = x.view(shape[0], shape[1], -1)
        x = self.flat(x)
        x = self.linear(x)
        x = x.squeeze(-1)
        return x


class LaBraM(NeuralTransformer):
    """
    A wrapper for the LaBraM model to simplify initialisation and for compatibility of the input format.
    """

    def __init__(self, channel_names, checkpoint=None, **kwargs):
        args = dict(
            patch_size=200,
            embed_dim=200,
            depth=12,
            num_heads=10,
            mlp_ratio=4,
            qk_norm=partial(nn.LayerNorm, eps=1e-6),
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=1,
            drop_rate=0,
            drop_path_rate=0.1,
            attn_drop_rate=0,
            drop_block_rate=None,
            use_mean_pooling=True,
            init_scale=0.001,
            use_rel_pos_bias=False,
            use_abs_pos_emb=True,
            init_values=0.1,
            qkv_bias=False,
        )
        args.update(kwargs)
        super().__init__(**args)

        self.input_chans = get_input_chans(channel_names)

        if checkpoint:
            if checkpoint.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    checkpoint, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(checkpoint, map_location='cpu')

            checkpoint_model = None
            for model_key in ["model", "module"]:
                if model_key in checkpoint:
                    checkpoint_model = checkpoint[model_key]
                    print("Load state_dict by model_key = %s" % model_key)
                    break
            if checkpoint_model is None:
                checkpoint_model = checkpoint
            if (checkpoint_model is not None):
                all_keys = list(checkpoint_model.keys())
                new_dict = OrderedDict()
                for key in all_keys:
                    if key.startswith('student.'):
                        new_dict[key[8:]] = checkpoint_model[key]
                    else:
                        pass
                checkpoint_model = new_dict

            state_dict = self.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            all_keys = list(checkpoint_model.keys())
            for key in all_keys:
                if "relative_position_index" in key:
                    checkpoint_model.pop(key)

            load_state_dict(self, checkpoint_model)

    
    def forward(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        '''
        x: [batchsize x n_eeg_chnnels x n_timepoints]
        input_chans: custom list of channel indices for positional encoding (only required if not as configured during initialization)
        '''

        if input_chans is None:
            input_chans = self.input_chans

        # Convert from [batchsize x n_eeg_chnnels x n_timepoints] to format expected by LaBram, i.e.
        # [batch size, number of electrodes, number of patches, patch size]
        # For example, for an EEG sample of 4 seconds with 64 electrodes, x will be [batch size, 64, 4, 200]
        x = x.reshape(x.shape[0], x.shape[1], -1, self.patch_size)

        x = self.forward_features(x, input_chans=input_chans, return_patch_tokens=return_patch_tokens, return_all_tokens=return_all_tokens, **kwargs)
        x = self.head(x).squeeze(-1)
        return x


def build_model(config, device):

    if config["dataset"] == "TUAB" or config["dataset"] == "CAUEEG":
        n_channels = 19
        n_timepoints = 400 if config["model"] == "LaBraM" else 200
        channel_names = [
            "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
            "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"
        ]
    elif config["dataset"] == "PhysioNet":
        n_channels = 6
        n_timepoints = 3000
        channel_names = ["F3", "F4", "C3", "C4", "O1", "O2"]
    else:
        raise ValueError(f"Unknown dataset '{config['dataset']}'")
    
    if config["task"] == "normality":
        n_outputs = 1
    elif config["task"] == "dementia":
        n_outputs = 3
    elif config["task"] == "sleep_stage":
        n_outputs = 5
    else:
        raise ValueError(f"Unknown task '{config['task']}'")

    if config["model"] == "TCN":
        model = TCN(
            n_chans=n_channels,
            n_outputs=n_outputs,
            n_times=n_timepoints,
            n_blocks=4,
            n_filters=64,
            kernel_size=5,
            drop_prob=0,
            add_log_softmax=False,
        ).to(device)
    elif config["model"] == "mAtt":
        model = mAtt(
            n_channels=n_channels,
            n_outputs=n_outputs,
        ).to(device)
    elif config["model"] == "LaBraM":
        model = LaBraM(
            EEG_size=n_timepoints,
            channel_names=channel_names,
            num_classes=n_outputs,
            checkpoint=config.get("checkpoint", None),
        ).to(device)
    else:
        raise ValueError(f"Unknown model '{config['model']}'")
    
    return model