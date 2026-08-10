"""Stereo mixing timeline + gain staging, ported from iidxOnKnitting src/audio/{mix,master}.rs.

Everything is float32 in -1.0..1.0 at a fixed rate (GITADORA is natively 48 kHz end to end).
The gain model is the audited one (m4f0s6 calibration, within 0.6 dB of the game's own masters):
  note gain = KEYSOUND_GAIN * (entry.volume/127) * (note.velocity/127), pan by unity-centre balance
  master    = soft-knee limiter, |x| <= 0.95 untouched, excess folded by tanh (never exceeds 1.0)
"""
import numpy as np

KEYSOUND_GAIN = 0.70       # keysound layer attenuation over the pre-mastered bed (-3.1 dB)
KNEE_THRESHOLD = 0.95      # where the soft knee starts
VOLUME_FULL_SCALE = 127.0  # both VA3 entry volume and per-note velocity are 0..127
_I16_FULL_SCALE = 32768.0


class Timeline:
    """A growable (frames, 2) float32 mixing buffer at a fixed rate"""

    def __init__(self, rate_hz: int, frames_hint: int = 0):
        self.rate_hz = rate_hz
        self.samples = np.zeros((frames_hint, 2), dtype=np.float32)

    @classmethod
    def from_stereo_i16(cls, rate_hz: int, bed: np.ndarray) -> "Timeline":
        timeline = cls(rate_hz)
        timeline.samples = bed.astype(np.float32) / _I16_FULL_SCALE
        return timeline

    def frames(self) -> int:
        return len(self.samples)

    def ensure_frames(self, frames: int):
        if frames > len(self.samples):
            grown = np.zeros((frames, 2), dtype=np.float32)
            grown[:len(self.samples)] = self.samples
            self.samples = grown

    def add_mono_i16(self, frame_start: int, pcm: np.ndarray, gain_left: float, gain_right: float):
        """Sum a mono int16 source in at frame_start, panned by per-channel gains"""
        self.ensure_frames(frame_start + len(pcm))
        scaled = pcm.astype(np.float32) / _I16_FULL_SCALE
        segment = self.samples[frame_start:frame_start + len(pcm)]
        segment[:, 0] += scaled * gain_left
        segment[:, 1] += scaled * gain_right

    def seconds(self) -> float:
        return len(self.samples) / self.rate_hz


def pan_gains(pan: int) -> tuple[float, float]:
    """GITADORA pan byte (0..127, 64 = centre) -> per-channel gains, unity-centre balance law
    (a centred sound keeps full level in both channels; panning only attenuates the opposite side)"""
    offset = (pan - 64.0) / 64.0
    return 1.0 - max(offset, 0.0), 1.0 + min(offset, 0.0)


def soft_knee_limit(samples: np.ndarray, threshold: float = KNEE_THRESHOLD) -> np.ndarray:
    """Fold everything above `threshold` smoothly towards full scale (in place); stateless tanh
    waveshaping -- no look-ahead, no pumping, output never exceeds full scale"""
    span = 1.0 - threshold
    magnitude = np.abs(samples)
    mask = magnitude > threshold
    samples[mask] = np.sign(samples[mask]) * (threshold + span * np.tanh((magnitude[mask] - threshold) / span))
    return samples


def peak_normalize(samples: np.ndarray) -> np.ndarray:
    """Scale the whole buffer so its peak is at most 1.0, only when it exceeds 1.0 (in place)"""
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    if peak > 1.0:
        samples *= 1.0 / peak
    return samples
