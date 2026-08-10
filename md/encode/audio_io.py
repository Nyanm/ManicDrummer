"""Load an audio file as the mono 24kHz float32 waveform MuQ expects. librosa decodes opus via
soundfile (OGG/Opus), resampling to 24kHz and downmixing to mono in one call."""
import librosa
import numpy as np

from .. import config


def load_mono(str_audio_path: str) -> np.ndarray:
    """Decode an audio file to a mono float32 waveform at MuQ's sample rate (24kHz)"""
    wav, _sr = librosa.load(str_audio_path, sr=config.MUQ_SAMPLE_RATE_HZ, mono=True)
    return wav.astype(np.float32)
