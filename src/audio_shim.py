"""Import shims for audio I/O. Library module, imported by train_model only.

torchaudio and speechbrain cannot be installed alongside torch 1.13.1
(their surviving releases require torch>=2), but openWakeWord's data.py
only needs a tiny surface of each:
  - torchaudio.load / torchaudio.info (WAV I/O in augment_clips)
  - speechbrain.dataio.dataio.read_audio (unused in our path, provided anyway)
  - speechbrain.processing.signal_processing.reverberate (RIR augmentation)

This module injects lightweight equivalents into sys.modules BEFORE
openwakeword.data is imported. Sound semantics match the originals:
float32 tensors in [-1, 1], (channels, time) layout.
"""

import sys
import types


def _torchaudio_module():
    import math

    import torch
    import soundfile as sf

    mod = types.ModuleType("torchaudio")
    mod.__version__ = "0.13.1"

    class _Info:
        def __init__(self, num_frames, sample_rate, num_channels):
            self.num_frames = num_frames
            self.sample_rate = sample_rate
            self.num_channels = num_channels

    def load(path, **kwargs):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), sr

    def info(path, **kwargs):
        i = sf.info(str(path))
        return _Info(
            num_frames=i.frames,
            sample_rate=i.samplerate,
            num_channels=i.channels,
        )

    def list_audio_backends():
        return ["soundfile"]

    def set_audio_backend(backend):
        return None

    class Resample(torch.nn.Module):
        """FFT resampler. Matches torchaudio.transforms.Resample call shape."""

        def __init__(self, orig_freq=16000, new_freq=16000):
            super().__init__()
            self.orig_freq = orig_freq
            self.new_freq = new_freq

        def forward(self, waveform: torch.Tensor) -> torch.Tensor:
            if self.orig_freq == self.new_freq:
                return waveform
            n = waveform.shape[-1]
            new_n = max(1, int(round(n * self.new_freq / self.orig_freq)))
            spec = torch.fft.rfft(waveform, n=n, dim=-1)
            want_bins = new_n // 2 + 1
            if want_bins < spec.shape[-1]:
                spec = spec[..., :want_bins]
            else:
                spec = torch.nn.functional.pad(spec, (0, want_bins - spec.shape[-1]))
            return torch.fft.irfft(spec, n=new_n, dim=-1)

    class TimeStretch(torch.nn.Module):
        """Phase-vocoder time stretcher for complex spectrograms.

        Matches torchaudio.transforms.TimeStretch call shape as used by
        torch_pitch_shift (fixed_rate, n_freq, hop_length).
        """

        def __init__(self, hop_length=None, n_freq=201, fixed_rate=1.0):
            super().__init__()
            self.hop_length = hop_length or 256
            self.n_freq = n_freq
            self.fixed_rate = float(fixed_rate)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            rate = self.fixed_rate
            if rate == 1.0:
                return x
            freq, hop = self.n_freq, self.hop_length
            n_fft = 2 * (freq - 1)
            shape = x.shape
            xb = x.reshape(-1, freq, shape[-1])
            mag = xb.abs()
            phase = xb.angle()
            omega = (
                (2 * math.pi * hop / n_fft)
                * torch.arange(freq, device=xb.device, dtype=mag.dtype)
            ).view(1, freq, 1)
            dphi = phase[..., 1:] - phase[..., :-1] - omega
            dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
            inst = omega + dphi
            inst = torch.cat([inst, inst[..., -1:]], dim=-1)
            t_old, t_new = shape[-1], max(1, int(round(shape[-1] / rate)))
            pos = (torch.arange(t_new, device=xb.device, dtype=mag.dtype) * rate).clamp(
                0, t_old - 1 - 1e-6
            )
            i0 = pos.floor().long()
            frac = (pos - i0.float()).view(1, 1, t_new)
            i0c = i0.clamp(0, t_old - 1)
            i1c = (i0 + 1).clamp(0, t_old - 1)
            b = xb.shape[0]
            g0 = i0c.view(1, 1, t_new).expand(b, freq, t_new)
            g1 = i1c.view(1, 1, t_new).expand(b, freq, t_new)
            mag_new = mag.gather(-1, g0) * (1 - frac) + mag.gather(-1, g1) * frac
            adv = inst.gather(-1, g0) * (1 - frac) + inst.gather(-1, g1) * frac
            phase_new = phase[..., :1] + torch.cumsum(adv * rate, dim=-1)
            out = mag_new * torch.exp(1j * phase_new.to(mag.dtype))
            return out.reshape(shape[:-2] + (freq, t_new))

    transforms = types.ModuleType("torchaudio.transforms")
    transforms.Resample = Resample
    transforms.TimeStretch = TimeStretch

    mod.load = load
    mod.info = info
    mod.list_audio_backends = list_audio_backends
    mod.set_audio_backend = set_audio_backend
    mod.transforms = transforms
    return mod


def _reverberate(waveforms, rir_waveform, rescale_amp="avg"):
    """Convolve with an RIR, truncate to input length, rescale amplitude.

    Mirrors speechbrain.processing.signal_processing.reverberate for the
    (batch, time) + rescale_amp="avg" case used by openWakeWord.
    """
    import numpy as np
    import torch
    from scipy.signal import fftconvolve

    rir = rir_waveform[0] if rir_waveform.dim() > 1 else rir_waveform
    h = rir.detach().cpu().numpy().astype(np.float64)
    out = []
    for row in waveforms:
        x = row.detach().cpu().numpy().astype(np.float64)
        y = fftconvolve(x, h)[: x.shape[0]]
        if rescale_amp == "avg":
            denom = float(np.abs(y).mean())
            if denom > 0:
                y = y * (float(np.abs(x).mean()) / denom)
        out.append(torch.from_numpy(y.astype(np.float32)))
    return torch.stack(out)


def install() -> None:
    """Inject the shims. Idempotent; call before importing openwakeword.data."""
    if "torchaudio" in sys.modules and getattr(
        sys.modules["torchaudio"], "__shim__", False
    ):
        return
    ta = _torchaudio_module()
    sys.modules["torchaudio"] = ta
    sys.modules["torchaudio.transforms"] = ta.transforms
    ta.__shim__ = True

    import torch
    import soundfile as sf

    sb = types.ModuleType("speechbrain")
    sb.__path__ = []
    sb_dataio = types.ModuleType("speechbrain.dataio")
    sb_dataio.__path__ = []
    sb_dataio_dataio = types.ModuleType("speechbrain.dataio.dataio")

    def read_audio(path, **kwargs):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), sr

    sb_dataio_dataio.read_audio = read_audio

    sb_proc = types.ModuleType("speechbrain.processing")
    sb_proc.__path__ = []
    sb_sig = types.ModuleType("speechbrain.processing.signal_processing")
    sb_sig.reverberate = _reverberate

    sys.modules["speechbrain"] = sb
    sys.modules["speechbrain.dataio"] = sb_dataio
    sys.modules["speechbrain.dataio.dataio"] = sb_dataio_dataio
    sys.modules["speechbrain.processing"] = sb_proc
    sys.modules["speechbrain.processing.signal_processing"] = sb_sig
