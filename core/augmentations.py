import numpy as np

import torch
from torch.nn.functional import pad
from torch.fft import fft, ifft

from torcheeg.transforms import EEGTransform
from sklearn.utils import check_random_state


class AmplitudeScaling(EEGTransform):
    """Independently scale the amplitude of each EEG channel."""
    def __init__(self, min_scaling_factor=0.5, max_scaling_factor=2):
        super().__init__()
        self.min_scaling_factor = min_scaling_factor
        self.max_scaling_factor = max_scaling_factor

    def apply(self, eeg):
        scaling_factors = torch.rand(eeg.shape[0]) * (self.max_scaling_factor - self.min_scaling_factor) + self.min_scaling_factor
        eeg *= scaling_factors.unsqueeze(1)
        return eeg


class FrequencyShift(EEGTransform):
    """
    Random shift in the frequency domain.
    The shift is consistent across all EEG channels.
    
    Adapted from braindecode's FrequencyShift augmentation.
    Simplified under the preconditions that the input is a single EEG sample
    rather than a batch and that the same shift is applied to all channels.
    """
    def __init__(
        self,
        sfreq,
        max_shift=0.3,  # in Hz
        seed=None,
    ):
        super().__init__()
        
        self.sfreq = sfreq        
        self.max_shift = max_shift

        self.rng = check_random_state(seed)

    def apply(self, eeg):
        u = torch.as_tensor(self.rng.uniform(), device=eeg.device)
        shift = u * 2 * self.max_shift - self.max_shift
        eeg = self._frequency_shift(eeg, shift)
        return eeg

    def _frequency_shift(self, eeg, shift):
        """
        Shift the specified signal by the specified frequency.
        See https://gist.github.com/lebedov/4428122
        """
        # Pad the signal with zeros to prevent the FFT invoked by the transform
        # from slowing down the computation:
        n_channels, N_orig = eeg.shape[-2:]
        N_padded = 2 ** int(np.ceil(np.log2(np.abs(N_orig))))
        t = torch.arange(N_padded, device=eeg.device) / self.sfreq
        padded = pad(eeg, (0, N_padded - N_orig))

        N = padded.shape[-1]
        f = fft(padded, N, dim=-1)
        h = torch.zeros_like(f)
        if N % 2 == 0:
            h[..., 0] = h[..., N // 2] = 1
            h[..., 1 : N // 2] = 2
        else:
            h[..., 0] = 1
            h[..., 1 : (N + 1) // 2] = 2

        analytical = ifft(f * h, dim=-1)

        shift = shift.repeat(n_channels, N_padded)
        
        eeg = analytical * torch.exp(2j * np.pi * shift * t)
        eeg = eeg[..., :N_orig].real.float()

        return eeg


class PhaseRandomization(EEGTransform):
    """
    Randomly shift the phase of each frequency component in the EEG signal.
    Can be applied channel-wise or with the same phase shifts across EEG channels.

    Adapted from braindecode's ft_surrogate augmentation.
    Simplified under the preconditions that the input is a single EEG sample
    rather than a batch.
    """
    def __init__(
        self,
        max_phase_shift=1,  # between 0 and 1, will be interpreted as `max_phase_shift` * 2 * `pi`
        channel_wise=True,  # if True, apply different phase shifts to each channel
        seed=None,
    ):
        super().__init__()

        assert 0 <= max_phase_shift <= 1, "max_phase_shift must be in the range [0, 1]"

        self.max_phase_shift = max_phase_shift
        self.channel_wise = channel_wise

        self.rng = check_random_state(seed)

    def apply(self, eeg):
        f = fft(eeg.double(), dim=-1)
        random_phase = self._new_random_fft_phase(
            f.shape[-2] if self.channel_wise else 1,
            f.shape[-1],
            device=eeg.device,
        )

        if not self.channel_wise:
            random_phase = torch.tile(random_phase, (f.shape[-2], 1))

        f_shifted = f * torch.exp(self.max_phase_shift * random_phase)
        shifted = ifft(f_shifted, dim=-1)
        eeg = shifted.real.float()

        return eeg
    
    def _new_random_fft_phase(self, c, n, device):
        if n % 2:
            random_phase = torch.from_numpy(
                2j * np.pi * self.rng.random((c, (n - 1) // 2))
            ).to(device)
            return torch.cat(
                [
                    torch.zeros((c, 1), device=device),
                    random_phase,
                    -torch.flip(random_phase, [-1]),
                ],
                dim=-1,
            )
        else:
            random_phase = torch.from_numpy(
                2j * np.pi * self.rng.random((c, n // 2 - 1))
            ).to(device)
            return torch.cat(
                [
                    torch.zeros((c, 1), device=device),
                    random_phase,
                    torch.zeros((c, 1), device=device),
                    -torch.flip(random_phase, [-1]),
                ],
                dim=-1,
            )